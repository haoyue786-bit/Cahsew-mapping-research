#!/usr/bin/env python3
"""Single-tile Stage4 FADA training with a guarded PlanetMap class prior."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fada import (
    FadaDomainMarginalView,
    FineGrainedDomainDiscriminator,
    class_knowledge,
    discriminator_loss as fada_discriminator_loss,
    resize_segmentation_logits,
    target_adversarial_loss as fada_target_adversarial_loss,
)
from planet_prior import PRIOR_LOCK_RELATIVE, resolve_planet_profile, validate_prior_lock
from protocol import (
    ANCHOR_CONFIG,
    BATCH_SIZE,
    MAX_EPOCH,
    MIN_EARLY_STOP_EPOCH,
    PARAMETER_BOUNDS,
    PROTOCOL_VERSION,
    REVIEW_INTERVAL,
    SEED,
    TARGET_TRAIN_COUNT,
    atomic_json,
    candidate_score,
    canonical_prior_config,
    canonical_training_config,
    epoch_batches,
    new_early_stop_state,
    read_lines,
    sha256,
    steps_per_epoch,
    target_distribution_valid,
    update_early_stop,
    write_status,
)
from geometry import inspect_geometry
from review_metrics import ReviewAccumulator, gradient_probe_steps
from architecture import feature_learning_parameters, make_model, model_parameter_groups
from agent_policy import (
    AGENT_MODEL,
    AGENT_REASONING_EFFORT,
    AGENT_SKILL_NAME,
    apply_agent_training_decision,
    build_initialization_request,
    build_review_request,
    build_selection_request,
    validate_agent_provenance,
    validate_initialization_decision,
    validate_review_decision,
    validate_selection_decision,
    parameter_payload,
)


ARCHIVE = Path("@WORKSPACE@/code/exp_convconnect/domain_adaptation_archive")
GAN_FILE = ARCHIVE / "03_agent_guide_gan/source/agent_strategy_tests_ivoire2019_tile0019e/followup_exp4tw_exp5_featuregan/gan_search/train_gan_variant.py"
GRL_CODE = ARCHIVE / "02_agent_guide_grl/source/agent_autonomous_da_ivoire2019_tile0019e/code"
MODEL_REGISTRY = Path(__file__).resolve().parents[1] / "references/model_registry.json"
AGENT_DECISION_SCHEMA = Path(__file__).resolve().parents[1] / "references/agent_decision.schema.json"
AGENT_CONTROLLER_SKILL = Path(__file__).resolve().parents[1] / ".codex/skills/cashew-fada-agent-controller/SKILL.md"
GPU_LEASE_ROOT = ARCHIVE / ".agent_guide_gan_gpu_leases"
# The request should normally set this explicitly. A conservative
# single-slot fallback keeps older manifests safe while allowing any positive
# multi-slot value (2, 4, 8, ...) when the machine has the memory/throughput.
DEFAULT_MAX_WORKERS_PER_GPU = 1
RUNTIME_HASHES = {
    GAN_FILE: "2362c66badb38851089703293a012a0f1e7f5247f9a9a52b37d2e2cedeb97436",
    ARCHIVE / "03_agent_guide_gan/source/agent_strategy_tests_ivoire2019_tile0019e/shared/code/common.py": "83d1877fbad9c788bf0aa365b0a6ab5c430f1dbe5324ad6a428af7af895e2306",
    ARCHIVE / "03_agent_guide_gan/source/agent_strategy_tests_ivoire2019_tile0019e/followup_exp4tw_exp5_featuregan/exp05_agent_feature_gan/discriminator/feature_gan.py": "57326b39235b193121ad5a3723856b395fb877a27ef1b84152e98c63394eb72e",
    ARCHIVE / "03_agent_guide_gan/source/agent_strategy_tests_ivoire2019_tile0019e/followup_exp4tw_exp5_featuregan/gan_search/baseline_audit/strict_heldout_probe.py": "b455a05d8decf354ff80758f89a9c82d633693c5f8bf4870c0f02cbf1b77daa9",
    GRL_CODE / "base_train_tile.py": "41386d3bf2bf5ebdb3ecbc628427e3635f579af759c32204243770bdda792b5d",
    GRL_CODE / "epoch0_diagnostics.py": "38f2cca1b976cc0a9b8bb29f23c6e21f1ccd1e6b75f5b87f610226b771b0a97a",
}


def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_gan_module():
    """Load the protected canonical Agent-Guide GAN runtime directly (never experiment 09)."""
    sys.path.insert(0, str(GRL_CODE))
    sys.path.insert(0, "@WORKSPACE@/code")
    sys.path.insert(0, "@WORKSPACE@/code/exp_convconnect")
    spec = importlib.util.spec_from_file_location("canonical_agent_gan_runtime", GAN_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    if spec.loader is None:
        raise ImportError(GAN_FILE)
    spec.loader.exec_module(module)
    common = sys.modules.get("common")
    if common is None:
        raise RuntimeError("canonical Agent-Guide GAN did not load its common runtime")
    return module, common


def source_label_resolver(label_root: Path):
    def resolve(name: str) -> Path:
        stem = Path(name).stem
        if stem.startswith("AlphaEarth_"):
            parts = stem.split("_", 2)
            if len(parts) == 3 and parts[1].isdigit():
                stem = parts[2]
        return label_root / f"label_processed_{stem}.tif"
    return resolve


def source_label_path(label_root: Path, name: str) -> Path:
    return source_label_resolver(label_root)(name)


def diagnostic_target_names(names: list[str], minimum: int = 64) -> list[str]:
    """Give legacy paired diagnostics a stable sample count on partial tiles.

    The training stream remains the real target list and keeps its exact
    coverage/count.  The protected historical domain-head diagnostic assumes
    64 source and 64 target examples and hard-codes that label length; repeat
    names only for this read-only diagnostic when a valid partial tile has
    fewer than 64 patches.  No repeated sample enters optimization, CORAL, or
    prediction.
    """
    if not names:
        raise ValueError("diagnostics require at least one target patch")
    if len(names) >= minimum:
        return list(names)
    repeats = (minimum + len(names) - 1) // len(names)
    return (list(names) * repeats)[:minimum]


def full_target_prediction_metrics(gan, common, model, target_root: Path,
                                   names: list[str], device: torch.device,
                                   batch_size: int = BATCH_SIZE,
                                   return_argmax: bool = False) -> dict[str, Any]:
    """Measure one fixed target scope with one invariant pixel-level definition.

    This runs after an epoch in eval mode, so it neither updates BatchNorm state
    nor mixes changing within-epoch model snapshots. ``class_distribution`` is
    always the hard four-class argmax fraction on valid target pixels. The mean
    softmax distribution remains available under an explicit, different name.
    """
    if not names:
        raise ValueError("target prediction metrics require at least one patch")
    if batch_size < 1:
        raise ValueError("target prediction metric batch size must be positive")
    counts = np.zeros(4, dtype=np.int64)
    probability_sums = np.zeros(4, dtype=np.float64)
    entropy_sum = 0.0
    valid_pixels = 0
    argmax_chunks = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(names), batch_size):
                part = names[start:start + batch_size]
                images = np.stack([
                    common.read_raster(target_root / name) for name in part
                ]).astype(np.float32)
                logits, _ = gan.forward_fuse(
                    model, torch.from_numpy(images).to(device).float()
                )
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                predictions = probabilities.argmax(axis=1)
                for image, probability, prediction in zip(images, probabilities, predictions):
                    support = ~np.all(image == 0, axis=0)
                    count = int(support.sum())
                    if count == 0:
                        continue
                    np.add.at(counts, prediction[support], 1)
                    probability_sums += probability[:, support].sum(axis=1, dtype=np.float64)
                    entropy = -(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum(axis=0)
                    entropy_sum += float(entropy[support].sum(dtype=np.float64))
                    valid_pixels += count
                    if return_argmax:
                        # The fixed name order and support definition make this
                        # a deterministic whole-region sequence.  It is used
                        # only for the next review's spatial-stability check.
                        argmax_chunks.append(prediction[support].astype(np.uint8, copy=True))
    finally:
        model.train(was_training)
    if valid_pixels == 0:
        raise RuntimeError("target prediction metric found no valid support pixels")
    hard_distribution = counts.astype(np.float64) / float(valid_pixels)
    mean_softmax = probability_sums / float(valid_pixels)
    result = {
        "class_distribution": hard_distribution.tolist(),
        "class_distribution_kind": "hard_argmax_valid_pixel_fraction",
        "predicted_class_counts": counts.tolist(),
        "pixel_count": int(valid_pixels),
        "patch_count": len(names),
        "inference_batch_size": int(batch_size),
        "scope": "target_full_list",
        "model_mode": "eval",
        "support_definition": "pixels where not all input bands are zero",
        "mean_softmax_distribution": mean_softmax.tolist(),
        "entropy": entropy_sum / float(valid_pixels),
        "normalized_entropy": entropy_sum / float(valid_pixels) / float(np.log(4.0)),
    }
    if return_argmax:
        result["_argmax_codes"] = np.concatenate(argmax_chunks) if argmax_chunks else np.empty(0, dtype=np.uint8)
    return result


def apply_full_target_prediction_metrics(review: dict[str, Any],
                                         metrics: dict[str, Any]) -> None:
    """Replace legacy ambiguous target distribution fields in-place."""
    target = review.setdefault("target_unsupervised", {})
    target.update(copy.deepcopy(metrics))
    review.setdefault("metric_provenance", {}).update({
        "target_distribution": "target_full_list_eval_mode_hard_argmax_valid_pixels",
        "target_mean_softmax": "target_full_list_eval_mode_valid_pixel_mean",
        "target_entropy": "target_full_list_eval_mode_valid_pixel_mean",
        "target_distribution_comparable_across_adversarial_states": True,
    })


def balanced_binary_source_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Successful source2025 objective: class 2 versus log-sum-exp of 0/1/3."""
    valid = labels != 10
    if not bool(valid.any()):
        return logits.sum() * 0
    log_probability = F.log_softmax(logits, dim=1)
    binary_log_probability = torch.stack(
        [torch.logsumexp(log_probability[:, (0, 1, 3)], dim=1), log_probability[:, 2]],
        dim=1,
    )
    target = (labels == 2).long()
    negative_log_likelihood = -binary_log_probability.gather(1, target.unsqueeze(1)).squeeze(1)
    negative_weight, positive_weight = 0.6311714517, 2.4059025179
    weights = torch.where(
        target == 1,
        torch.as_tensor(positive_weight, device=logits.device),
        torch.as_tensor(negative_weight, device=logits.device),
    )
    return (negative_log_likelihood[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-8)


def four_class_source_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Original source2025 four-class supervision used for epoch-173 pretraining.

    The target output is a four-way argmax map, so source supervision must also
    distinguish crop/other, mixed vegetation, cashew, and built-up.  Class 10
    remains the ignore value.  The weights are the recorded epoch-173
    pretraining weights.
    """
    valid = labels != 10
    if not bool(valid.any()):
        return logits.sum() * 0.0
    weights = torch.as_tensor([3.0, 3.0, 10.0, 3.0], dtype=logits.dtype,
                              device=logits.device)
    return F.cross_entropy(logits, labels.long(), weight=weights,
                           ignore_index=10)


def configure_runtime(job: dict, device: torch.device):
    source = job["source"]
    source_root = Path(source["source_root"])
    label_root = Path(source["source_label_root"])
    source_train = Path(source["source_train"])
    source_valid = Path(source["source_valid"])
    pretrained = Path(source["pretrained"])
    target_root = Path(job["target_root"])
    target_train = Path(job["target_train_list"])
    target_full = Path(job["target_full_list"])
    gan, common = load_gan_module()
    resolver = source_label_resolver(label_root)
    values = {
        "SOURCE_TRAIN": source_train,
        "SOURCE_VALID": source_valid,
        "SOURCE_ROOT": source_root,
        "SOURCE_IMAGE_ROOT": source_root,
        "SOURCE_LABEL_ROOT": label_root,
        "PRETRAINED": pretrained,
        "TARGET_ROOT": target_root,
        "TARGET_TRAIN": target_train,
        "TARGET_FULL": target_full,
    }
    for name, value in values.items():
        setattr(gan, name, value)
        setattr(common, name, value)
    gan.source_label_path = resolver
    common.source_label_path = resolver
    common.base.SOURCE_IMAGE_PREFIX = f"AlphaEarth_{int(source['source_year'])}_"
    common.base.SOURCE_IMAGE_ROOT = str(source_root)
    common.base.SOURCE_LABEL_ROOT = str(label_root)
    common.base.TARGET_IMAGE_ROOT = str(target_root)
    common.old_diag.SOURCE_ROOT = source_root
    common.old_diag.LABEL_ROOT = label_root
    common.old_diag.SOURCE_VALID = source_valid
    common.old_diag.source_label_path = resolver
    common.old_diag.TARGET_ROOT = target_root
    common.old_diag.DEVICE = device
    return gan, common


def train_epoch(gan, model, discriminator, model_optimizer, d_optimizer, dataset,
                source_count: int, target_count: int, config: dict,
                device: torch.device, rng: np.random.RandomState,
                review_accumulator: ReviewAccumulator | None = None) -> dict[str, Any]:
    """One full-coverage epoch with the FADA joint domain/class objective."""
    model.train()
    discriminator.train()
    totals = {key: [] for key in (
        "D_loss", "D_acc", "cls", "adv", "D_grad", "cls_grad", "adv_grad",
        "D_source_loss", "D_target_loss",
    )}
    observed_shapes = set()
    batch_size = int(config["batch_size"])
    steps = int(config["steps_per_epoch"])
    knowledge_config = config["class_knowledge"]
    prior_shift = (np.asarray(config["prior_clipped_log_ratio"], dtype=np.float64) *
                   float(config["prior_strength"])).tolist()
    adversarial_active = config.get("adversarial_state", "active") == "active"
    review_probe_step_numbers = gradient_probe_steps(steps) if review_accumulator is not None else set()
    source_batches = epoch_batches(source_count, batch_size, steps, rng)
    target_batches = epoch_batches(target_count, batch_size, steps, rng)
    coverage = {
        "source_unique": len({item for batch in source_batches for item in batch}),
        "target_unique": len({item for batch in target_batches for item in batch}),
    }
    if coverage != {"source_unique": source_count, "target_unique": target_count}:
        raise RuntimeError(f"coverage failed: {coverage}")
    for step in range(steps):
        step_d_losses = []
        step_d_accuracies = []
        target_features_for_review = None
        target_logits_for_review = None
        source_batch, target_batch = gan.batch_from(
            dataset, source_batches[step], target_batches[step], device
        )
        source_aug, source_label, source_raw, _ = source_batch
        _, _, _, target_raw = target_batch
        source_label = source_label.long()
        gan.set_requires_grad(model, False)
        gan.set_requires_grad(discriminator, adversarial_active)
        for _ in range(int(config["D_steps_per_G_step"]) if adversarial_active else 0):
            with torch.no_grad():
                source_features = gan.stage4_map(model, source_raw)
                target_features = gan.stage4_map(model, target_raw)
                source_segmentation, _ = gan.forward_fuse(model, source_raw.float())
                target_segmentation, _ = gan.forward_fuse(model, target_raw.float())
                source_lowres = resize_segmentation_logits(source_segmentation, source_features)
                target_lowres = resize_segmentation_logits(target_segmentation, target_features)
                source_knowledge = class_knowledge(
                    source_lowres,
                    temperature=knowledge_config["temperature"],
                    confidence_cap=knowledge_config["confidence_cap"],
                )
                target_knowledge = class_knowledge(
                    target_lowres,
                    temperature=knowledge_config["temperature"],
                    confidence_cap=knowledge_config["confidence_cap"],
                    prior_shift=prior_shift,
                )
            d_optimizer.zero_grad(set_to_none=True)
            discriminator_loss, d_stats = fada_discriminator_loss(
                discriminator, source_features.detach(), target_features.detach(),
                source_knowledge, target_knowledge,
            )
            discriminator_loss.backward()
            totals["D_grad"].append(gan.gradient_norm([p.grad for p in discriminator.parameters()]))
            d_optimizer.step()
            totals["D_loss"].append(float(discriminator_loss.item()))
            totals["D_acc"].append(float(d_stats["marginal_domain_accuracy"]))
            totals["D_source_loss"].append(float(d_stats["source_loss"]))
            totals["D_target_loss"].append(float(d_stats["target_loss"]))
            step_d_losses.append(float(discriminator_loss.item()))
            step_d_accuracies.append(float(d_stats["marginal_domain_accuracy"]))
            observed_shapes.add(tuple(d_stats["output_shape_source"]))
        gan.set_requires_grad(model, True)
        gan.set_requires_grad(discriminator, False)
        model_optimizer.zero_grad(set_to_none=True)
        source_logits, _ = gan.forward_fuse(model, source_aug.float())
        classification_loss = gan.weighted_source_loss(source_logits, source_label)
        encoder_parameters = feature_learning_parameters(model)
        classification_gradients = torch.autograd.grad(
            classification_loss, encoder_parameters, retain_graph=True, allow_unused=True
        )
        if adversarial_active:
            target_features = gan.stage4_map(model, target_raw)
            target_segmentation, _ = gan.forward_fuse(model, target_raw.float())
            target_lowres = resize_segmentation_logits(target_segmentation, target_features)
            target_knowledge = class_knowledge(
                target_lowres,
                temperature=knowledge_config["temperature"],
                confidence_cap=knowledge_config["confidence_cap"],
                prior_shift=prior_shift,
            )
            adversarial_loss = fada_target_adversarial_loss(discriminator, target_features, target_knowledge)
            target_features_for_review = target_features
            target_logits_for_review = target_segmentation
            adversarial_gradients = torch.autograd.grad(
                adversarial_loss, encoder_parameters, retain_graph=True, allow_unused=True
            )
            (classification_loss + config["lambda_adv"] * adversarial_loss).backward()
        else:
            adversarial_loss = classification_loss.detach() * 0.0
            adversarial_gradients = tuple(None for _ in encoder_parameters)
            classification_loss.backward()
        model_optimizer.step()
        totals["cls"].append(float(classification_loss.item()))
        totals["adv"].append(float(adversarial_loss.item()))
        totals["cls_grad"].append(gan.gradient_norm(classification_gradients))
        totals["adv_grad"].append(gan.gradient_norm(adversarial_gradients))
        if review_accumulator is not None:
            review_accumulator.update(
                source_loss=float(classification_loss.detach().item()),
                adv_loss=float(adversarial_loss.detach().item()),
                d_loss=float(np.mean(step_d_losses)) if step_d_losses else 0.0,
                source_logits=source_logits,
                source_labels=source_label,
                target_logits=target_logits_for_review,
                d_accuracy=float(np.mean(step_d_accuracies)) if step_d_accuracies else None,
                d_updates=len(step_d_accuracies),
                cls_grad_norm=(gan.gradient_norm(classification_gradients)
                               if step + 1 in review_probe_step_numbers else None),
                adv_grad_norm=(gan.gradient_norm(adversarial_gradients)
                               if step + 1 in review_probe_step_numbers else None),
                lambda_adv=float(config["lambda_adv"]),
                target_features=target_features_for_review,
            )
        gan.set_requires_grad(discriminator, adversarial_active)
    result = {key: float(np.mean(values)) if values else 0.0 for key, values in totals.items()}
    result["D_accuracy_per_branch"] = {"fada_2k_marginal": result["D_acc"]}
    result["effective_adv_to_cls_grad_ratio"] = (
        config["lambda_adv"] * result["adv_grad"] / max(1e-12, result["cls_grad"])
        if adversarial_active else 0.0
    )
    result.update({
        "batch_size": batch_size, "steps_per_epoch": steps,
        "source_count": source_count, "target_count": target_count,
        "full_epoch_sampling": True,
        "source_unique_coverage": coverage["source_unique"],
        "target_unique_coverage": coverage["target_unique"],
        "fada_discriminator_count": 1,
        "fada_output_channels": 8,
        "fada_output_shapes": [list(shape) for shape in sorted(observed_shapes)],
        "fada_feature_input": "Swin Stage4 low-resolution spatial map",
        "target_prior_logit_shift": list(prior_shift),
        "prior_applied_to_target_encoding_only": True,
        "adversarial_state": config.get("adversarial_state", "active"),
        "formal_adv_update_applied": adversarial_active,
        "potential_resume_gradient_ratio": None,
    })
    return result


def fada_challengers(gan, model, discriminator, device: torch.device) -> dict[str, Any]:
    """Use the protected label-free probes with a binary marginal view of FADA."""
    view = nn.ModuleDict({"fada_domain_marginal": FadaDomainMarginalView(discriminator)})
    return gan.heldout_challengers(
        model, view, "v2b", {"fada_domain_marginal": 1.0}, device,
    )


def source_validation_review(common, model, source_valid_path: Path) -> dict[str, Any]:
    """Keep the fixed source validation metric separate from training metrics."""
    source = common.old_diag.evaluate_source(model, read_lines(source_valid_path))
    return {"source": source}


def last_persistent_agent_decision_hash(output: Path, decisions: list[dict[str, Any]]) -> str | None:
    """Use the durable session as the chain authority when resuming old runs."""
    session_path = output / "agent/session.json"
    if session_path.is_file():
        session = json.loads(session_path.read_text())
        value = session.get("last_decision_sha256")
        if isinstance(value, str) and value:
            return value
    for item in reversed(decisions):
        value = item.get("decision_sha256")
        if isinstance(value, str) and value:
            return value
    return None


def restore_persisted_agent_history(output: Path, decisions: list[dict[str, Any]],
                                    decisions_path: Path) -> list[dict[str, Any]]:
    """Repair a pre-existing checkpoint history missing its agent bookkeeping.

    Older checkpoints can contain an empty/stale decision_history even though
    the durable agent_decisions.json and session chain are valid.  Preserve
    safety records from the checkpoint and merge in missing Agent records;
    this does not create a turn or alter any decision.
    """
    if not decisions_path.is_file():
        return decisions
    persisted = json.loads(decisions_path.read_text())
    if not isinstance(persisted, list):
        return decisions
    merged = list(decisions)
    existing = {
        (item.get("decision_kind"), int(item.get("epoch", 0)))
        for item in merged if isinstance(item, dict)
    }
    for item in persisted:
        if not isinstance(item, dict):
            continue
        key = (item.get("decision_kind"), int(item.get("epoch", 0)))
        if key not in existing:
            merged.append(item)
            existing.add(key)
    return merged


def finalize_review_epoch(common, model, source_valid_path: Path,
                          accumulator: ReviewAccumulator, epoch: int,
                          config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a review from one completed review epoch without a new batch."""
    accumulated = accumulator.finalize(
        epoch=epoch,
        config=config,
        snd_fn=common.old_diag.soft_neighborhood_density,
    )
    review = source_validation_review(common, model, source_valid_path)
    review.update({
        "source_training": accumulated["source"],
        "target_unsupervised": accumulated["target_unsupervised"],
        "discriminator_training": accumulated["discriminator"],
        "gradient_diagnostics": accumulated["gradients"],
        "training": accumulated["training"],
        "metric_provenance": accumulated["metric_provenance"],
        "safety_events": accumulated["safety_events"],
    })
    return review, accumulated


def model_state_digest(model) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _fixed_probe_batches(count: int, steps: int, rng: np.random.RandomState) -> list[list[int]]:
    if count < BATCH_SIZE or steps < 1:
        raise ValueError(f"pre-run requires at least one complete batch: count={count}")
    permutation = rng.permutation(count).tolist()
    stream = permutation[:steps * BATCH_SIZE]
    return [stream[index:index + BATCH_SIZE] for index in range(0, len(stream), BATCH_SIZE)]


def _probe_batches(source_count: int, target_count: int, rng: np.random.RandomState) -> tuple[list[list[int]], list[list[int]]]:
    # The pre-run asks for eight fixed probe batches, not eight full epochs.
    # When a partial tile has fewer complete batches, use the same number of
    # complete batches from both streams and never cycle a sample here.
    steps = min(8, source_count // BATCH_SIZE, target_count // BATCH_SIZE)
    if steps < 1:
        raise ValueError(f"pre-run requires one complete source and target batch: {source_count}/{target_count}")
    source = _fixed_probe_batches(source_count, steps, rng)
    target = _fixed_probe_batches(target_count, steps, rng)
    return source, target


def _probe_knowledge(logits: torch.Tensor, temperature: float, confidence_cap: float, prior_shift=None) -> torch.Tensor:
    return class_knowledge(logits, temperature=temperature, confidence_cap=confidence_cap, prior_shift=prior_shift)


def review_only_gradient_probe(gan, model, discriminator, dataset, source_count: int,
                               target_count: int, config: dict[str, Any],
                               device: torch.device, epoch: int) -> dict[str, Any]:
    """Measure a possible adversarial resume without touching formal state."""
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model_training, discriminator_training = model.training, discriminator.training
    model_digest, discriminator_digest = model_state_digest(model), model_state_digest(discriminator)
    model_requires_grad = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    discriminator_requires_grad = {id(parameter): parameter.requires_grad for parameter in discriminator.parameters()}
    try:
        # This is retained only for the paused-state no-update resume
        # diagnostic.  It is deliberately a fixed diagnostic batch, not the
        # strict full-coverage formal epoch sampler.  Active reviews use the
        # ReviewAccumulator fed by normal training steps instead.
        probe_rng = np.random.RandomState(SEED + 3000 + int(epoch))
        source_indices = _fixed_probe_batches(source_count, 1, probe_rng)[0]
        target_indices = _fixed_probe_batches(target_count, 1, probe_rng)[0]
        source_batch, target_batch = gan.batch_from(dataset, source_indices, target_indices, device)
        source_aug, source_label, _, target_raw = source_batch[0], source_batch[1], source_batch[2], target_batch[3]
        model.eval()
        discriminator.eval()
        gan.set_requires_grad(discriminator, False)
        source_logits, _ = gan.forward_fuse(model, source_aug.float())
        cls_loss = gan.weighted_source_loss(source_logits, source_label.long())
        target_feature = gan.stage4_map(model, target_raw)
        target_logits, _ = gan.forward_fuse(model, target_raw.float())
        target_lowres = resize_segmentation_logits(target_logits, target_feature)
        prior_shift = (np.asarray(config["prior_clipped_log_ratio"], dtype=np.float64) * float(config["prior_strength"])).tolist()
        target_knowledge = class_knowledge(
            target_lowres,
            temperature=config["class_knowledge"]["temperature"],
            confidence_cap=config["class_knowledge"]["confidence_cap"],
            prior_shift=prior_shift,
        )
        adv_loss = fada_target_adversarial_loss(discriminator, target_feature, target_knowledge)
        parameters = feature_learning_parameters(model)
        cls_grad = gan.gradient_norm(torch.autograd.grad(cls_loss, parameters, retain_graph=True, allow_unused=True))
        adv_grad = gan.gradient_norm(torch.autograd.grad(adv_loss, parameters, allow_unused=True))
        return {
            "potential_resume_cls_grad": float(cls_grad),
            "potential_resume_adv_grad": float(adv_grad),
            "potential_resume_gradient_ratio": float(config["lambda_adv"] * adv_grad / max(1e-12, cls_grad)),
            "potential_resume_probe_batch_size": BATCH_SIZE,
            "potential_resume_probe_seed": SEED + 3000 + int(epoch),
        }
    finally:
        model.train(model_training)
        discriminator.train(discriminator_training)
        for parameter in model.parameters():
            parameter.requires_grad_(model_requires_grad[id(parameter)])
        for parameter in discriminator.parameters():
            parameter.requires_grad_(discriminator_requires_grad[id(parameter)])
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if model_state_digest(model) != model_digest or model_state_digest(discriminator) != discriminator_digest:
            raise RuntimeError("review-only gradient probe contaminated formal model state")


def run_prerun(gan, model, discriminator, dataset, source_names: list[str], target_names: list[str],
               review_target_names: list[str], sampler_rng: np.random.RandomState,
               prior_clipped_log_ratio: list[float], device: torch.device) -> dict[str, Any]:
    """Run only discardable diagnostics, leaving formal model/RNG state untouched."""
    before_model = model_state_digest(model)
    before_sampler = sampler_rng.get_state()
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    probe_rng = np.random.RandomState(SEED + 1001)
    source_batches, target_batches = _probe_batches(len(source_names), len(target_names), probe_rng)
    cache = []
    model.eval()
    with torch.no_grad():
        for source_indices, target_indices in zip(source_batches, target_batches):
            source_batch, target_batch = gan.batch_from(dataset, source_indices, target_indices, device)
            source_aug, source_label, source_raw, _ = source_batch
            _, _, _, target_raw = target_batch
            source_feature = gan.stage4_map(model, source_raw)
            target_feature = gan.stage4_map(model, target_raw)
            source_segmentation, _ = gan.forward_fuse(model, source_raw.float())
            target_segmentation, _ = gan.forward_fuse(model, target_raw.float())
            cache.append({"source_raw": source_raw.detach().clone(), "source_aug": source_aug.detach().clone(), "source_label": source_label.detach().clone(), "target_raw": target_raw.detach().clone(), "source_feature": source_feature.detach().clone(), "target_feature": target_feature.detach().clone(), "source_lowres": resize_segmentation_logits(source_segmentation, source_feature).detach().clone(), "target_lowres": resize_segmentation_logits(target_segmentation, target_feature).detach().clone()})
    baseline = gan.review_diagnostics(model, review_target_names)
    probe_d = copy.deepcopy(discriminator).to(device)
    anchor_shift = (np.asarray(prior_clipped_log_ratio, dtype=np.float64) * 0.25).tolist()
    temperature_probe = {}
    for temperature in (1.2, 1.5, 1.8, 2.2, 2.6, 3.0):
        source_values = []
        target_values = []
        for item in cache:
            for values, logits, shift in (
                (source_values, item["source_lowres"], None),
                (target_values, item["target_lowres"], anchor_shift),
            ):
                probs = _probe_knowledge(logits, temperature, 0.9, shift)
                max_prob = probs.max(dim=1).values
                values.append({
                    "mean_max_probability": float(max_prob.mean()),
                    "median_max_probability": float(max_prob.median()),
                    "normalized_entropy": float((-(probs * probs.clamp_min(1e-12).log()).sum(dim=1) / np.log(4)).mean()),
                    "fraction_max_gt_0.8": float((max_prob > 0.8).float().mean()),
                    "fraction_max_gt_0.9": float((max_prob > 0.9).float().mean()),
                    "predicted_soft_class_distribution": probs.mean(dim=(0, 2, 3)).tolist(),
                })

        def summarize(values):
            return {
                key: float(np.mean([value[key] for value in values]))
                if key != "predicted_soft_class_distribution"
                else np.mean([value[key] for value in values], axis=0).tolist()
                for key in values[0]
            }

        temperature_probe[str(temperature)] = {
            "source": summarize(source_values),
            "target": summarize(target_values),
        }
    main_lr_probe = {}
    for lr in (2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5):
        clone = copy.deepcopy(model).to(device)
        optimizer = torch.optim.AdamW(clone.parameters(), lr=lr, weight_decay=0.01)
        initial, losses, gradients = None, [], []
        for item in cache[:8]:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = gan.forward_fuse(clone, item["source_aug"].float())
            loss = gan.weighted_source_loss(logits, item["source_label"].long())
            if initial is None:
                initial = float(loss.item())
            loss.backward()
            grad = gan.gradient_norm([parameter.grad for parameter in clone.parameters()])
            gradients.append(grad)
            optimizer.step()
            losses.append(float(loss.item()))
        relative_change = losses[-1] / max(abs(initial or 0.0), 1e-12) - 1.0
        nonfinite = not bool(np.isfinite(losses + gradients).all())
        gradient_explosion = bool(gradients and np.max(gradients) > max(1e6, 100.0 * max(np.median(gradients), 1e-12)))
        main_lr_probe[str(lr)] = {"initial_source_loss": initial, "final_source_loss": losses[-1], "relative_loss_change": relative_change, "median_gradient_norm": float(np.median(gradients)), "max_gradient_norm": float(np.max(gradients)), "nonfinite": nonfinite, "loss_worsened": bool(relative_change > 0.5), "gradient_explosion": gradient_explosion, "unsafe": bool(nonfinite or relative_change > 0.5 or gradient_explosion)}
        del clone, optimizer
    d_lr_probe = {}
    for lr in (5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5):
        clone_d = copy.deepcopy(probe_d).to(device)
        optimizer = torch.optim.AdamW(clone_d.parameters(), lr=lr, weight_decay=0.01)
        losses, accuracies, gradients = [], [], []
        for item in cache[:12]:
            source_k = _probe_knowledge(item["source_lowres"], 1.8, 0.9)
            target_k = _probe_knowledge(item["target_lowres"], 1.8, 0.9, anchor_shift)
            optimizer.zero_grad(set_to_none=True)
            loss, stats = fada_discriminator_loss(clone_d, item["source_feature"], item["target_feature"], source_k, target_k)
            loss.backward()
            gradients.append(gan.gradient_norm([parameter.grad for parameter in clone_d.parameters()]))
            optimizer.step()
            losses.append(float(loss.item()))
            accuracies.append(float(stats["marginal_domain_accuracy"]))
        d_nonfinite = not bool(np.isfinite(losses + accuracies + gradients).all())
        rapid_saturation = bool(accuracies[-1] > 0.99 and (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-12) > 0.5)
        d_lr_probe[str(lr)] = {"initial_D_loss": losses[0], "final_D_loss": losses[-1], "relative_D_loss_decrease": (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-12), "final_marginal_domain_accuracy": accuracies[-1], "median_D_grad": float(np.median(gradients)), "max_D_grad": float(np.max(gradients)), "nonfinite": d_nonfinite, "rapid_saturation": rapid_saturation, "unsafe": bool(d_nonfinite or rapid_saturation)}
        del clone_d, optimizer
    warm_d = copy.deepcopy(probe_d).to(device)
    warm_optimizer = torch.optim.AdamW(warm_d.parameters(), lr=2e-6, weight_decay=0.01)
    for step in range(50):
        item = cache[step % len(cache)]
        source_k = _probe_knowledge(item["source_lowres"], 1.8, 0.9)
        target_k = _probe_knowledge(item["target_lowres"], 1.8, 0.9, anchor_shift)
        warm_optimizer.zero_grad(set_to_none=True)
        loss, _ = fada_discriminator_loss(warm_d, item["source_feature"], item["target_feature"], source_k, target_k)
        loss.backward()
        warm_optimizer.step()
    gradient_cls, gradient_adv = [], []
    for item in cache:
        probe_model = copy.deepcopy(model).to(device)
        probe_model.train()
        source_logits, _ = gan.forward_fuse(probe_model, item["source_aug"].float())
        cls_loss = gan.weighted_source_loss(source_logits, item["source_label"].long())
        target_feature = gan.stage4_map(probe_model, item["target_raw"])
        target_logits, _ = gan.forward_fuse(probe_model, item["target_raw"].float())
        target_lowres = resize_segmentation_logits(target_logits, target_feature)
        target_k = _probe_knowledge(target_lowres, 1.8, 0.9, anchor_shift)
        adv_loss = fada_target_adversarial_loss(warm_d, target_feature, target_k)
        params = feature_learning_parameters(probe_model)
        gradient_cls.append(gan.gradient_norm(torch.autograd.grad(cls_loss, params, allow_unused=True)))
        gradient_adv.append(gan.gradient_norm(torch.autograd.grad(adv_loss, params, allow_unused=True)))
        del probe_model
    cls_grad, adv_grad = float(np.median(gradient_cls)), float(np.median(gradient_adv))
    raw_ratio = adv_grad / max(cls_grad, 1e-12)
    lambda_candidates = {"R_%.2f" % ratio: float(np.clip(ratio * cls_grad / max(adv_grad, 1e-12), 1e-4, 5e-2)) for ratio in (0.08, 0.10, 0.12, 0.15, 0.20, 0.25)}
    heldout_probe = fada_challengers(gan, model, discriminator, device)
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    after_model = model_state_digest(model)
    if before_model != after_model or not np.array_equal(before_sampler[1], sampler_rng.get_state()[1]) or before_sampler[0] != sampler_rng.get_state()[0]:
        raise RuntimeError("pre-run contaminated formal model or sampler RNG")
    _, initial_domain_stats = fada_discriminator_loss(
        probe_d,
        cache[0]["source_feature"],
        cache[0]["target_feature"],
        _probe_knowledge(cache[0]["source_lowres"], 1.8, 0.9),
        _probe_knowledge(cache[0]["target_lowres"], 1.8, 0.9, anchor_shift),
    )
    return {
        "schema_version": 1,
        "independent_seed": SEED + 1001,
        "source_probe_batches": len(cache),
        "target_probe_batches": len(cache),
        "source_baseline": {
            "class2_f1": float(baseline["source"]["f1_per_class"][2]),
            "macro_f1": float(baseline["source"]["macro_f1"]),
        },
        "target": {key: baseline["target_unsupervised"][key] for key in ("entropy", "snd", "class_distribution")},
        "domain": {
            "heldout_probe": heldout_probe,
            "initial_fada_marginal_accuracy": float(initial_domain_stats["marginal_domain_accuracy"]),
        },
        "feature": {
            "source_stage4_norm": float(np.mean([item["source_feature"].norm().item() for item in cache])),
            "target_stage4_norm": float(np.mean([item["target_feature"].norm().item() for item in cache])),
        },
        "temperature_probe": temperature_probe,
        "main_lr_probe": main_lr_probe,
        "D_lr_probe": d_lr_probe,
        "gradient_probe": {
            "cls_grad_median": cls_grad,
            "adv_grad_median": adv_grad,
            "raw_adv_to_cls_ratio": raw_ratio,
            "lambda_candidates": lambda_candidates,
        },
        "formal_model_hash_before": before_model,
        "formal_model_hash_after": after_model,
        "formal_sampler_rng_unchanged": True,
        "formal_optimizer_created": False,
    }


def preflight(job_path: Path, require_cuda: bool = False) -> dict[str, Any]:
    job = json.loads(job_path.read_text())
    errors = []
    if job.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("protocol_version mismatch")
    if job.get("state") != "queued":
        errors.append(f"job state is {job.get('state')}, not queued")
    training = job.get("training", {})
    expected = canonical_training_config()
    if training != expected:
        mismatches = {
            key: {"actual": training.get(key), "expected": expected.get(key)}
            for key in sorted(set(training) | set(expected))
            if training.get(key) != expected.get(key)
        }
        errors.append(f"canonical training config mismatch: {mismatches}")
    agent_contract = job.get("agent_guide", {})
    expected_agent_contract = {
        "controller": "persistent_codex_luna_per_tile",
        "model": AGENT_MODEL,
        "reasoning_effort": AGENT_REASONING_EFFORT,
        "sandbox": "read-only",
        "persistent_thread": True,
        "fresh_ephemeral_session": False,
        "review_interval": REVIEW_INTERVAL,
        "skill": AGENT_SKILL_NAME,
        "skill_path": ".codex/skills/cashew-fada-agent-controller/SKILL.md",
        "planet_inputs_exposed": False,
        "target_labels_exposed": False,
    }
    if agent_contract != expected_agent_contract:
        errors.append(f"formal Luna Agent contract mismatch: {agent_contract}")
    if training.get("adaptive_controller") is not True or training.get("review_interval") != REVIEW_INTERVAL:
        errors.append("adaptive persistent controller or review interval mismatch")
    if training.get("bounds") != PARAMETER_BOUNDS or training.get("anchors") != ANCHOR_CONFIG:
        errors.append("parameter bounds/anchors mismatch")
    if not AGENT_CONTROLLER_SKILL.is_file():
        errors.append(f"missing controller Skill: {AGENT_CONTROLLER_SKILL}")
    registry = json.loads(MODEL_REGISTRY.read_text())
    registered_source = registry.get("profiles", {}).get(job.get("source_profile"))
    if registered_source is None:
        errors.append(f"source profile is not registered: {job.get('source_profile')}")
    elif job.get("source") != registered_source:
        errors.append("job source profile differs from the canonical model registry")
    try:
        prior_config = canonical_prior_config(job.get("prior", {}).get("config"))
    except Exception as exc:
        prior_config = None
        errors.append(f"prior config invalid: {type(exc).__name__}: {exc}")
    planet_profile = None
    try:
        _, planet_profile = resolve_planet_profile(job)
        if job.get("prior", {}).get("planet_path") != str(Path(planet_profile["path"]).resolve()):
            errors.append("job PlanetMap path differs from the registered country profile")
        if job.get("prior", {}).get("planet_country_key") != planet_profile["country_key"]:
            errors.append("job PlanetMap country key differs from the registry")
    except Exception as exc:
        errors.append(f"PlanetMap registry validation failed: {type(exc).__name__}: {exc}")
    runtime_hash_report = {}
    for runtime_path, expected_hash in RUNTIME_HASHES.items():
        actual_hash = sha256(runtime_path) if runtime_path.is_file() else None
        runtime_hash_report[str(runtime_path)] = actual_hash
        if actual_hash != expected_hash:
            errors.append(f"canonical runtime hash mismatch: {runtime_path}: {actual_hash}")
    paths = {
        "target_root": Path(job["target_root"]),
        "target_train_list": Path(job["target_train_list"]),
        "target_full_list": Path(job["target_full_list"]),
        "source_root": Path(job["source"]["source_root"]),
        "source_label_root": Path(job["source"]["source_label_root"]),
        "source_train": Path(job["source"]["source_train"]),
        "source_valid": Path(job["source"]["source_valid"]),
        "pretrained": Path(job["source"]["pretrained"]),
    }
    for label, path in paths.items():
        if not path.exists():
            errors.append(f"missing {label}: {path}")
    source_names = read_lines(paths["source_train"]) if paths["source_train"].is_file() else []
    target_names = read_lines(paths["target_train_list"]) if paths["target_train_list"].is_file() else []
    full_names = read_lines(paths["target_full_list"]) if paths["target_full_list"].is_file() else []
    if len(source_names) != int(job["source"]["source_train_expected_count"]):
        errors.append(f"source count mismatch: {len(source_names)}")
    valid_names = read_lines(paths["source_valid"]) if paths["source_valid"].is_file() else []
    if len(valid_names) != int(job["source"]["source_valid_expected_count"]):
        errors.append(f"source valid count mismatch: {len(valid_names)}")
    expected_target = int(job["expected_target_patch_count"])
    declared_target = int(job.get("target_patch_count", len(full_names)))
    declared_train = int(job.get("target_train_count", len(target_names)))
    if len(target_names) < 1 or len(target_names) > TARGET_TRAIN_COUNT:
        errors.append(f"target training count must be in 1..{TARGET_TRAIN_COUNT}, got {len(target_names)}")
    if len(target_names) != declared_train:
        errors.append(f"target training count differs from job: {len(target_names)}/{declared_train}")
    if len(full_names) < 1 or len(full_names) != declared_target:
        errors.append(f"full target count mismatch: {len(full_names)}/{declared_target}")
    if len(source_names) != len(set(source_names)) or len(target_names) != len(set(target_names)) or len(full_names) != len(set(full_names)):
        errors.append("duplicate entries in a data list")
    if not set(target_names).issubset(full_names):
        errors.append("target training list is not a subset of target full list")
    if paths["pretrained"].is_file() and sha256(paths["pretrained"]) != job["source"]["pretrained_sha256"]:
        errors.append("pretrained SHA-256 mismatch")
    if paths["source_train"].is_file() and sha256(paths["source_train"]) != job.get("source_qa", {}).get("source_train_sha256"):
        errors.append("source train list SHA-256 mismatch")
    if paths["source_valid"].is_file() and sha256(paths["source_valid"]) != job.get("source_qa", {}).get("source_valid_sha256"):
        errors.append("source valid list SHA-256 mismatch")
    if paths["target_train_list"].is_file() and sha256(paths["target_train_list"]) != job["target_train_list_sha256"]:
        errors.append("target train list SHA-256 mismatch")
    if paths["target_full_list"].is_file() and sha256(paths["target_full_list"]) != job["target_full_list_sha256"]:
        errors.append("target full list SHA-256 mismatch")
    missing_target = [name for name in full_names if not (paths["target_root"] / name).is_file()]
    if missing_target:
        errors.append(f"missing target files: {len(missing_target)}; examples={missing_target[:3]}")
    missing_source = [
        name for name in source_names
        if not (paths["source_root"] / name).is_file()
        or not source_label_path(paths["source_label_root"], name).is_file()
    ]
    missing_valid = [
        name for name in valid_names
        if not (paths["source_root"] / name).is_file()
        or not source_label_path(paths["source_label_root"], name).is_file()
    ]
    if missing_source:
        errors.append(f"missing source train image/label files: {len(missing_source)}; examples={missing_source[:3]}")
    if missing_valid:
        errors.append(f"missing source valid image/label files: {len(missing_valid)}; examples={missing_valid[:3]}")
    geometry_summary = None
    if not missing_target and full_names:
        try:
            geometry = inspect_geometry(
                paths["target_root"], full_names, expected_target,
                allow_partial=(len(full_names) != expected_target),
            )
            geometry_summary = {
                "width": geometry["width"], "height": geometry["height"],
                "patch_width": geometry["patch_width"], "patch_height": geometry["patch_height"],
                "columns": geometry["columns"], "rows": geometry["rows"],
                "crs": str(geometry["crs"]), "transform": list(geometry["transform"]),
                "bounds": list(geometry["bounds"]),
            }
        except Exception as exc:
            errors.append(f"target affine geometry QA failed: {type(exc).__name__}: {exc}")
    try:
        derived = steps_per_epoch(len(source_names), len(target_names), BATCH_SIZE)
    except Exception as exc:
        errors.append(f"steps derivation failed: {exc}")
        derived = None
    if derived != job.get("derived_steps_per_epoch"):
        errors.append(f"derived steps mismatch: {derived}/{job.get('derived_steps_per_epoch')}")
    if require_cuda and not torch.cuda.is_available():
        errors.append("CUDA is required for formal training")
    prior_lock_path = Path(job["output_root"]) / PRIOR_LOCK_RELATIVE
    prior_lock_status = "missing_create_before_training"
    prior_lock_sha256 = None
    if prior_lock_path.is_file():
        try:
            validate_prior_lock(job_path, prior_lock_path)
            prior_lock_status = "valid"
            prior_lock_sha256 = sha256(prior_lock_path)
        except Exception as exc:
            errors.append(f"Planet prior lock invalid: {type(exc).__name__}: {exc}")
            prior_lock_status = "invalid"
    report = {
        "qa_pass": not errors,
        "job": str(job_path.resolve()),
        "job_id": job.get("job_id"),
        "protocol_version": job.get("protocol_version"),
        "source_count": len(source_names),
        "source_valid_count": len(valid_names),
        "target_train_count": len(target_names),
        "target_full_count": len(full_names),
        "target_expected_count": expected_target,
        "target_completeness": job.get("target_completeness"),
        "batch_size": BATCH_SIZE,
        "steps_per_epoch": derived,
        "max_epoch": MAX_EPOCH,
        "early_stop_enabled": True,
        "agent_guide": agent_contract,
        "job_manifest_sha256": sha256(job_path),
        "verified_paths": {key: str(path.resolve()) for key, path in paths.items()},
        "verified_hashes": {
            "source_train": sha256(paths["source_train"]) if paths["source_train"].is_file() else None,
            "source_valid": sha256(paths["source_valid"]) if paths["source_valid"].is_file() else None,
            "pretrained": sha256(paths["pretrained"]) if paths["pretrained"].is_file() else None,
            "target_train": sha256(paths["target_train_list"]) if paths["target_train_list"].is_file() else None,
            "target_full": sha256(paths["target_full_list"]) if paths["target_full_list"].is_file() else None,
        },
        "canonical_runtime_hashes": runtime_hash_report,
        "geometry": geometry_summary,
        "class_mapping": job.get("source", {}).get("class_mapping"),
        "prior_config": prior_config,
        "planet_profile": planet_profile,
        "prior_lock": str(prior_lock_path.resolve()),
        "prior_lock_status": prior_lock_status,
        "prior_lock_sha256": prior_lock_sha256,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(report)
    return report


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_rng(state: dict | None, saved_device: str | None, current_device: torch.device) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = [item.cpu() for item in state.get("torch_cuda", [])]
    if torch.cuda.is_available() and cuda_states:
        saved_index = 0
        if saved_device and saved_device.startswith("cuda:") and saved_device.split(":", 1)[1].isdigit():
            saved_index = int(saved_device.split(":", 1)[1])
        if saved_index >= len(cuda_states):
            raise RuntimeError(f"saved CUDA RNG device {saved_device} is unavailable in checkpoint state")
        torch.cuda.set_rng_state(cuda_states[saved_index], device=current_device)


def checkpoint_payload(epoch: int, model, discriminators, model_optimizer, d_optimizer,
                       config: dict, metrics: dict | None, decisions: list, sampler_rng,
                       early_stop: dict, job_path: Path, device_name: str,
                       pending_agent: dict | None = None) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "discriminators_state_dict": discriminators.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "discriminator_optimizer_state_dict": d_optimizer.state_dict(),
        "scheduler_state_dict": {"epoch": int(epoch), "type": "constant"},
        "rng_state": None,
        "sampler_rng_state": sampler_rng.get_state(),
        "config": copy.deepcopy(config),
        "metrics": metrics,
        "decision_history": copy.deepcopy(decisions),
        "early_stop_state": copy.deepcopy(early_stop),
        "pending_agent": copy.deepcopy(pending_agent),
        "job_manifest": str(job_path.resolve()),
        "job_manifest_sha256": sha256(job_path),
        "training_device": device_name,
        "fada_discriminator_kind": "single_fully_convolutional_2k",
        "fada_discriminator_output_channels": 8,
        "planet_used_in_training": "target_fada_class_encoding_only",
        "planet_used_for_agent_or_selection": False,
        "planet_prior_lock": config["planet_prior_lock"],
        "planet_prior_lock_sha256": config["planet_prior_lock_sha256"],
    }


def enrich_rng(payload: dict[str, Any], common) -> dict[str, Any]:
    payload["rng_state"] = common.rng_state()
    return payload


def deployment_checkpoint_payload(selected_payload: dict[str, Any], job_path: Path,
                                  training_checkpoint_sha256: str) -> dict[str, Any]:
    """Strip every training-only FADA/optimizer/prior component for inference."""
    if "state_dict" not in selected_payload or "epoch" not in selected_payload:
        raise KeyError("selected training checkpoint lacks state_dict or epoch")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "epoch": int(selected_payload["epoch"]),
        "state_dict": selected_payload["state_dict"],
        "job_manifest_sha256": sha256(job_path),
        "training_checkpoint_sha256": str(training_checkpoint_sha256),
        "fada_discriminator_included": False,
        "optimizer_state_included": False,
        "inference_prior_logit_adjustment": False,
    }
    forbidden = {
        "discriminators_state_dict", "discriminator_optimizer_state_dict",
        "optimizer_state_dict", "target_prior_logit_shift",
    }
    overlap = sorted(forbidden.intersection(payload))
    if overlap:
        raise RuntimeError(f"deployment payload retained training-only keys: {overlap}")
    return payload


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def claim_worker(output: Path, job_id: str, device_name: str,
                 slot: int | None = None) -> Path:
    lock = output / "worker.lock"
    if lock.exists():
        try:
            old = json.loads(lock.read_text())
        except Exception:
            old = {}
        if old.get("hostname") == socket.gethostname() and isinstance(old.get("pid"), int) and not process_alive(old["pid"]):
            lock.unlink()
        elif slot is not None and old.get("slot") != slot:
            raise RuntimeError({"job_slot_owned_by_another_agent": job_id,
                                "requested_slot": slot, "owner": old})
        else:
            raise RuntimeError({"job_already_claimed": job_id, "lock": str(lock), "owner": old})
    payload = {
        "job_id": job_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "device": device_name,
        "slot": slot,
    }
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    return lock


def gpu_index(device_name: str) -> int:
    if not device_name.startswith("cuda:") or not device_name.split(":", 1)[1].isdigit():
        raise ValueError(f"formal work requires an explicit cuda:N device, got {device_name}")
    return int(device_name.split(":", 1)[1])


def validate_cuda_device(device_name: str) -> int:
    index = gpu_index(device_name)
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError(f"requested CUDA device is unavailable: {device_name}")
    return index


def claim_pipeline_gpu(output: Path, job_id: str, device_name: str,
                       requested_slot: int | None = None,
                       max_slots: int = DEFAULT_MAX_WORKERS_PER_GPU) -> Path:
    """Reserve one explicit slot on a GPU across the full pipeline.

    ``max_slots`` is a shared-GPU concurrency cap, not a fixed hardware
    assumption. New requests should set it explicitly to any positive value;
    the conservative fallback is one slot when an old manifest omits it.
    """
    index = gpu_index(device_name)
    max_slots = int(max_slots)
    if max_slots < 1:
        raise ValueError(f"max_slots must be positive, got {max_slots}")
    if requested_slot is not None and not 0 <= int(requested_slot) < max_slots:
        raise ValueError(f"GPU slot must be in 0..{max_slots - 1}, got {requested_slot}")
    requested_slot = None if requested_slot is None else int(requested_slot)
    GPU_LEASE_ROOT.mkdir(parents=True, exist_ok=True)
    for other in GPU_LEASE_ROOT.glob("cuda_*.lock"):
        try:
            owner = json.loads(other.read_text())
        except Exception:
            continue
        if owner.get("job_id") == job_id and owner.get("device") != device_name:
            # A job may be retried on another GPU after a host/VPN failure.
            # Do not let a dead worker's lease permanently block that retry.
            pid = int(owner.get("pid", -1))
            try:
                os.kill(pid, 0)
                owner_alive = True
            except (OSError, ValueError):
                owner_alive = False
            if not owner_alive:
                other.unlink()
                continue
            raise RuntimeError({
                "job_already_reserved_on_another_gpu": job_id,
                "requested": device_name, "existing_lock": str(other), "owner": owner,
            })
    locks = sorted(GPU_LEASE_ROOT.glob(f"cuda_{index}*.lock"))
    for other in locks:
        try:
            owner = json.loads(other.read_text())
        except Exception:
            continue
        if owner.get("job_id") == job_id:
            if requested_slot is not None and int(owner.get("slot", 0)) != requested_slot:
                raise RuntimeError({"job_slot_mismatch": job_id, "requested_slot": requested_slot,
                                    "existing_slot": owner.get("slot")})
            atomic_json(other, {
                "job_id": job_id, "pid": os.getpid(), "hostname": socket.gethostname(),
                "device": device_name, "job_output": str(output.resolve()),
                "slot": owner.get("slot", 0), "max_workers_per_gpu": max_slots,
            })
            return other
    # A legacy unsuffixed lease occupies slot 0; new jobs use slot files.
    occupied = set()
    for lock_path in locks:
        try:
            old = json.loads(lock_path.read_text())
        except Exception:
            old = {}
        if old.get("job_id") == job_id:
            occupied.add(int(old.get("slot", 0)))
            continue
        slot = int(old.get("slot", 0))
        owner_status = Path(str(old.get("job_output", ""))) / "status.json"
        owner_state = None
        if owner_status.is_file():
            try:
                owner_state = json.loads(owner_status.read_text()).get("state")
            except Exception:
                owner_state = None
        if owner_state in {"failed", "training_verified", "verified_complete",
                           "training_complete", "prediction_complete"}:
            lock_path.unlink()
            continue
        occupied.add(slot)
    available = [slot for slot in range(max_slots)
                 if slot not in occupied and (requested_slot is None or slot == requested_slot)]
    if not available:
        raise RuntimeError({
            "gpu_worker_slots_exhausted": device_name,
            "max_workers_per_gpu": max_slots,
            "active_leases": len(locks),
        })
    slot = available[0]
    lock = GPU_LEASE_ROOT / f"cuda_{index}_slot_{slot}.lock"
    payload = {
        "job_id": job_id, "pid": os.getpid(), "hostname": socket.gethostname(),
        "device": device_name, "job_output": str(output.resolve()),
        "slot": slot, "max_workers_per_gpu": max_slots,
    }
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        # Another worker won the slot race; caller can retry safely.
        raise RuntimeError({"gpu_slot_race": device_name, "slot": slot})
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    return lock


def release_pipeline_gpu(lock: Path, job_id: str) -> None:
    if not lock.exists():
        return
    owner = json.loads(lock.read_text())
    if owner.get("job_id") != job_id:
        raise RuntimeError(f"refusing to release another job's GPU lease: {owner}")
    lock.unlink()


def release_job_gpu_leases(job_id: str) -> list[str]:
    released = []
    if not GPU_LEASE_ROOT.is_dir():
        return released
    for lock in sorted(GPU_LEASE_ROOT.glob("cuda_*.lock")):
        try:
            owner = json.loads(lock.read_text())
        except Exception:
            continue
        if owner.get("job_id") == job_id:
            lock.unlink()
            released.append(str(lock))
    return released


def validate_checkpoint_lock(job_path: Path, lock_path: Path | None = None) -> dict[str, Any]:
    job = json.loads(job_path.read_text())
    output = Path(job["output_root"])
    path = lock_path or output / "final/checkpoint_lock.json"
    lock = json.loads(path.read_text())
    checkpoint = Path(lock["checkpoint"])
    prior_path = Path(lock["planet_prior_lock"])
    payload = torch_load(checkpoint, "cpu") if checkpoint.is_file() else {}
    checks = {
        "protocol": lock.get("protocol_version") == PROTOCOL_VERSION,
        "job_id": lock.get("job_id") == job.get("job_id"),
        "job_manifest": lock.get("job_manifest") == str(job_path.resolve()),
        "job_manifest_sha256": lock.get("job_manifest_sha256") == sha256(job_path),
        "checkpoint": checkpoint.is_file() and sha256(checkpoint) == lock.get("checkpoint_sha256"),
        "batch": lock.get("batch_size") == BATCH_SIZE,
        "max_epoch": lock.get("max_epoch") == MAX_EPOCH,
        "prior_lock": (
            prior_path.is_file()
            and sha256(prior_path) == lock.get("planet_prior_lock_sha256")
        ),
        "prior_training_only": (
            lock.get("planet_used_for_training") == "target_fada_class_encoding_only"
            and lock.get("planet_used_for_agent_selection_or_early_stop") is False
            and lock.get("planet_used_at_inference") is False
        ),
        "deployment_model_only": (
            lock.get("deployment_checkpoint_model_only") is True
            and lock.get("fada_discriminator_included_in_deployment") is False
            and payload.get("fada_discriminator_included") is False
            and "discriminators_state_dict" not in payload
            and "optimizer_state_dict" not in payload
        ),
        "luna_high_selection": (
            lock.get("selection_agent_model") == AGENT_MODEL
            and lock.get("selection_agent_reasoning_effort") == AGENT_REASONING_EFFORT
            and lock.get("selection_agent_persistent_thread") is True
            and lock.get("selection_agent_fresh_ephemeral_session") is False
            and lock.get("deterministic_score_was_advisory_not_automatic_argmax") is True
        ),
    }
    if not all(checks.values()):
        raise RuntimeError({"invalid_checkpoint_lock": str(path), "checks": checks})
    return lock


def safety_action(config: dict, metrics: dict, epoch: int) -> dict | None:
    ratio = float(metrics["effective_adv_to_cls_grad_ratio"])
    if not np.isfinite(ratio):
        raise FloatingPointError(f"non-finite gradient ratio at epoch {epoch}: {ratio}")
    if ratio <= 0.5 or config.get("adversarial_state", "active") != "active":
        return None
    old = float(config["lambda_adv"])
    new = max(1e-4, old * min(0.5, 0.45 / ratio))
    config["lambda_adv"] = new
    paused = bool(new <= 1e-4 + 1e-12 and ratio > 0.5)
    if paused:
        config["adversarial_state"] = "paused"
    return {
        "epoch": epoch,
        "action": ["PER_EPOCH_GRADIENT_SAFETY_FUSE"],
        "reason": "effective adversarial/classification gradient ratio exceeded 0.5",
        "parameter_changes": {"lambda_adv": {"old": old, "new": new}},
        "changed_parameters": ["lambda_adv"],
        "emergency": True,
        "adversarial_state_after": config.get("adversarial_state", "active"),
        "safety_event": "PER_EPOCH_GRADIENT_SAFETY_PAUSE" if paused else "PER_EPOCH_GRADIENT_SAFETY_FUSE",
        "planet_used": False,
    }


def agent_artifact_paths(output: Path, kind: str, epoch: int) -> dict[str, Path]:
    if kind == "initialization":
        root = output / "agent/initialization"
    elif kind == "training_review":
        root = output / "agent/reviews" / f"epoch_{epoch:03d}"
    elif kind == "checkpoint_selection":
        root = output / "agent/selection"
    else:
        raise ValueError(f"unknown Agent request kind: {kind}")
    return {
        "root": root,
        "request": root / "request.json",
        "decision": root / "decision.json",
        "provenance": root / "provenance.json",
    }


def pending_agent_record(output: Path, request: dict[str, Any], epoch: int) -> dict[str, Any]:
    kind = request["decision_kind"]
    paths = agent_artifact_paths(output, kind, epoch)
    if paths["request"].exists():
        existing = json.loads(paths["request"].read_text())
        if existing != request:
            raise RuntimeError(f"immutable Agent request already differs: {paths['request']}")
    else:
        atomic_json(paths["request"], request)
    session_path = output / "agent/session.json"
    session = json.loads(session_path.read_text()) if session_path.is_file() else {}
    return {
        "decision_kind": kind,
        "epoch": int(epoch),
        "request": str(paths["request"].resolve()),
        "request_sha256": sha256(paths["request"]),
        "decision": str(paths["decision"].resolve()),
        "provenance": str(paths["provenance"].resolve()),
        "model": AGENT_MODEL,
        "reasoning_effort": AGENT_REASONING_EFFORT,
        "thread_id": session.get("thread_id"),
    }


def agent_status_payload(job: dict, pending: dict[str, Any], device_name: str,
                         slot: int, heartbeat_epoch: int) -> dict[str, Any]:
    state = {"initialization": "awaiting_luna_initialization", "training_review": "awaiting_luna_review", "checkpoint_selection": "awaiting_luna_selection"}[pending["decision_kind"]]
    return {
        "job_id": job["job_id"],
        "state": state,
        "worker": None,
        "hostname": socket.gethostname(),
        "gpu": device_name,
        "gpu_slot": slot,
        "heartbeat_epoch": int(heartbeat_epoch),
        "agent_model": AGENT_MODEL,
        "agent_reasoning_effort": AGENT_REASONING_EFFORT,
        "agent_thread_id": pending.get("thread_id"),
        "agent_request_kind": pending["decision_kind"],
        "agent_request": pending["request"],
        "agent_request_sha256": pending["request_sha256"],
        "agent_decision": pending["decision"],
        "agent_provenance": pending["provenance"],
    }


def validated_agent_artifacts(pending: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request_path = Path(pending["request"])
    decision_path = Path(pending["decision"])
    provenance_path = Path(pending["provenance"])
    if not request_path.is_file() or sha256(request_path) != pending["request_sha256"]:
        raise RuntimeError("pending Agent request is missing or hash-mismatched")
    if not decision_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("pending Luna Agent decision has not been dispatched")
    request = json.loads(request_path.read_text())
    decision = json.loads(decision_path.read_text())
    if request["decision_kind"] == "initialization":
        validate_initialization_decision(request, decision, pending["request_sha256"])
    elif request["decision_kind"] == "training_review":
        validate_review_decision(request, decision, pending["request_sha256"])
    elif request["decision_kind"] == "checkpoint_selection":
        validate_selection_decision(request, decision, pending["request_sha256"])
    else:
        raise ValueError(f"unknown Agent request kind: {request.get('decision_kind')}")
    provenance = validate_agent_provenance(
        request_path=request_path,
        decision_path=decision_path,
        provenance_path=provenance_path,
        schema_path=AGENT_DECISION_SCHEMA,
        sha256_file=sha256,
        expected_previous_decision_sha256=request.get("previous_decision_sha256"),
    )
    return request, decision, provenance


def agent_stop_reason(request: dict[str, Any]) -> str:
    envelope = request["safety_envelope"]
    if envelope.get("terminal_epoch"):
        return "terminal epoch 50 reviewed by luna-high Agent"
    if envelope.get("plateau_stop_eligible"):
        return "luna-high Agent selected stable label-free plateau"
    return "luna-high Agent safety stop to hard-safe checkpoint"


def remaining_epochs(start_epoch: int, early_stop: dict) -> list[int]:
    if bool(early_stop.get("stop")):
        return []
    return list(range(int(start_epoch) + 1, MAX_EPOCH + 1))


def train(job_path: Path, device_name: str, slot: int | None = None,
          max_slots: int | None = None) -> dict[str, Any]:
    if slot is None:
        raise ValueError("formal training requires an explicit owned --slot")
    preflight(job_path, require_cuda=True)
    job = json.loads(job_path.read_text())
    output = Path(job["output_root"])
    final_lock = output / "final/checkpoint_lock.json"
    if final_lock.is_file():
        validate_checkpoint_lock(job_path, final_lock)
        return {"state": "training_complete", "checkpoint_lock": str(final_lock)}
    validate_cuda_device(device_name)
    worker_lock = claim_worker(output, job["job_id"], device_name, slot)
    gpu_lease = None
    handoff_or_stage_succeeded = False
    status_path = output / "status.json"
    device = torch.device(device_name)
    try:
        gpu_lease = claim_pipeline_gpu(
            output, job["job_id"], device_name, requested_slot=slot,
            max_slots=int(max_slots if max_slots is not None else
                          job.get("max_workers_per_gpu", DEFAULT_MAX_WORKERS_PER_GPU)),
        )
        write_status(status_path, {
            "job_id": job["job_id"], "state": "training", "worker": os.getpid(),
            "hostname": socket.gethostname(), "gpu": device_name, "gpu_slot": slot,
            "heartbeat_epoch": 0,
        })
        gan, common = configure_runtime(job, device)
        gan.weighted_source_loss = balanced_binary_source_loss
        gan.seed_all(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        source_names = read_lines(Path(job["source"]["source_train"]))
        target_names = read_lines(Path(job["target_train_list"]))
        full_names = read_lines(Path(job["target_full_list"]))
        review_target_names = diagnostic_target_names(full_names)
        steps = steps_per_epoch(len(source_names), len(target_names), BATCH_SIZE)
        base = canonical_training_config()
        prior_lock_path = output / PRIOR_LOCK_RELATIVE
        prior_lock = validate_prior_lock(job_path, prior_lock_path)
        config = {
            "protocol_version": PROTOCOL_VERSION,
            "variant": "fada_2k_stage4",
            "tile": job["job_id"],
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "source_count": len(source_names),
            "target_count": len(target_names),
            "steps_per_epoch": steps,
            "full_epoch_sampling": True,
            "max_epoch": MAX_EPOCH,
            "review_interval": REVIEW_INTERVAL,
            "minimum_early_stop_epoch": MIN_EARLY_STOP_EPOCH,
            "source_objective": base["source_objective"],
            "source_binary_loss_weights": base["source_binary_loss_weights"],
            "lambda_adv": base["anchors"]["lambda_adv"],
            "D_lr": base["anchors"]["D_lr"],
            "main_lr": base["anchors"]["main_lr"],
            "D_steps_per_G_step": base["anchors"]["D_steps_per_G_step"],
            "class_knowledge": {"temperature": base["anchors"]["temperature"], "confidence_cap": base["anchors"]["confidence_cap"]},
            "main_weight_decay": base["anchors"]["main_weight_decay"],
            "D_weight_decay": base["anchors"]["D_weight_decay"],
            "prior_strength": base["anchors"]["prior_strength"],
            "model_adam_beta1": base["anchors"]["model_adam_beta1"],
            "model_adam_beta2": base["anchors"]["model_adam_beta2"],
            "model_adam_eps": base["anchors"]["model_adam_eps"],
            "D_adam_beta1": base["anchors"]["D_adam_beta1"],
            "D_adam_beta2": base["anchors"]["D_adam_beta2"],
            "D_adam_eps": base["anchors"]["D_adam_eps"],
            "adversarial_state": "active",
            "prior_clipped_log_ratio": prior_lock["clipped_log_prior_ratio"],
            "planet_prior_lock": str(prior_lock_path.resolve()),
            "planet_prior_lock_sha256": sha256(prior_lock_path),
            "planet_used_in_training": "target_fada_class_encoding_only",
            "planet_used_for_agent_or_selection": False,
            "planet_used_at_inference": False,
            "agent_controller": base["agent_guide"],
            "target_sampling": "full-permutation-then-cycle",
            "target_sampling_seed": SEED,
            "target_train_list": job["target_train_list"],
            "target_full_list": job["target_full_list"],
        }
        model, load_stats = make_model(gan, job, device)
        torch.manual_seed(SEED)
        discriminator = FineGrainedDomainDiscriminator(
            in_channels=768, num_classes=4, hidden_channels=(256, 128),
        ).to(device)
        model_optimizer = None
        d_optimizer = None

        def create_formal_optimizers():
            """Create the only optimizers used by formal training/resume."""
            return (
                torch.optim.AdamW(
                    model_parameter_groups(
                        model, config["main_lr"], config["main_weight_decay"],
                        (config["model_adam_beta1"], config["model_adam_beta2"]), config["model_adam_eps"],
                    ),
                ),
                torch.optim.AdamW(
                    discriminator.parameters(), lr=config["D_lr"], weight_decay=config["D_weight_decay"],
                    betas=(config["D_adam_beta1"], config["D_adam_beta2"]), eps=config["D_adam_eps"],
                ),
            )
        dataset = gan.StrategyDataset(source_names, target_names, "none")
        sampler_rng = np.random.RandomState(SEED)
        checkpoints = output / "checkpoints"
        latest = checkpoints / "latest.pt"
        candidates_path = output / "candidates.json"
        decisions_path = output / "agent_decisions.json"
        selection_decision = None
        selection_pending = None
        if latest.is_file():
            model_optimizer, d_optimizer = create_formal_optimizers()
            resume = torch_load(latest, "cpu")
            if resume.get("protocol_version") != PROTOCOL_VERSION:
                raise RuntimeError("resume checkpoint protocol mismatch")
            if resume.get("job_manifest_sha256") != sha256(job_path):
                raise RuntimeError("resume checkpoint belongs to a different job manifest")
            model.load_state_dict(resume["state_dict"], strict=True)
            discriminator.load_state_dict(resume["discriminators_state_dict"], strict=True)
            model_optimizer.load_state_dict(resume["optimizer_state_dict"])
            d_optimizer.load_state_dict(resume["discriminator_optimizer_state_dict"])
            move_optimizer_state(model_optimizer, device)
            move_optimizer_state(d_optimizer, device)
            config = resume["config"]
            decisions = resume["decision_history"]
            decisions = restore_persisted_agent_history(output, decisions, decisions_path)
            restore_rng(resume.get("rng_state"), resume.get("training_device"), device)
            sampler_rng.set_state(resume["sampler_rng_state"])
            early_stop = resume["early_stop_state"]
            candidates = json.loads(candidates_path.read_text())
            start_epoch = int(resume["epoch"])
            last_metrics = resume.get("metrics")
            review0 = json.loads((output / "review_epoch_000.json").read_text())["review"]
            pending = resume.get("pending_agent")
            if pending:
                if not Path(pending["decision"]).is_file() or not Path(pending["provenance"]).is_file():
                    write_status(
                        status_path,
                        agent_status_payload(job, pending, device_name, slot, start_epoch),
                    )
                    handoff_or_stage_succeeded = True
                    return {
                        "state": {"initialization": "awaiting_luna_initialization", "training_review": "awaiting_luna_review", "checkpoint_selection": "awaiting_luna_selection"}[pending["decision_kind"]],
                        "agent_request": pending["request"],
                        "agent_request_sha256": pending["request_sha256"],
                    }
                request, agent_decision, _agent_provenance = validated_agent_artifacts(pending)
                if pending["decision_kind"] == "initialization":
                    config_update = apply_agent_training_decision(config, model_optimizer, d_optimizer, agent_decision, 0)
                    config["pre_run_selected_config"] = parameter_payload(config)
                    decision_record = {
                        **agent_decision, "decision_kind": "initialization", "agent_model": AGENT_MODEL,
                        "agent_reasoning_effort": AGENT_REASONING_EFFORT, "request": pending["request"],
                        "decision": pending["decision"], "decision_sha256": sha256(Path(pending["decision"])),
                        "provenance": pending["provenance"], "provenance_sha256": sha256(Path(pending["provenance"])),
                        "changed_parameters": config_update["changed_parameters"], "parameters_after": config_update["after"],
                    }
                    decisions.append(decision_record)
                    atomic_json(output / "pre_run_selected_config.json", config["pre_run_selected_config"])
                    atomic_json(decisions_path, decisions)
                    resolved_payload = checkpoint_payload(
                        0, model, discriminator, model_optimizer, d_optimizer, config, None,
                        decisions, sampler_rng, early_stop, job_path, device_name, pending_agent=None,
                    )
                    atomic_torch_save(latest, enrich_rng(resolved_payload, common))
                    pending = None
                elif pending["decision_kind"] == "training_review":
                    config_update = apply_agent_training_decision(config, model_optimizer, d_optimizer, agent_decision, start_epoch)
                    early_stop = copy.deepcopy(pending["proposed_early_stop_state"])
                    early_stop["agent_model"] = AGENT_MODEL
                    early_stop["agent_reasoning_effort"] = AGENT_REASONING_EFFORT
                    early_stop["last_agent_request_sha256"] = pending["request_sha256"]
                    early_stop["last_agent_decision_sha256"] = sha256(Path(pending["decision"]))
                    early_stop["stop"] = not bool(agent_decision["continue_training"])
                    early_stop["stop_reason"] = agent_stop_reason(request) if early_stop["stop"] else None
                    decision_record = {
                        **agent_decision,
                        "agent_model": AGENT_MODEL,
                        "agent_reasoning_effort": AGENT_REASONING_EFFORT,
                        "request": pending["request"],
                        "request_sha256": pending["request_sha256"],
                        "decision": pending["decision"],
                        "decision_sha256": sha256(Path(pending["decision"])),
                        "provenance": pending["provenance"],
                        "provenance_sha256": sha256(Path(pending["provenance"])),
                        "parameter_changes": config_update["changed_parameters"],
                        "changed_parameters": config_update["changed_parameters"],
                        "parameters_after": config_update["after"],
                        "config_after_decision": copy.deepcopy(config),
                    }
                    decisions.append(decision_record)
                    candidate = copy.deepcopy(pending["candidate"])
                    candidate.update({
                        "agent_decision": decision_record,
                        "agent_decision_status": "resolved",
                    })
                    if any(int(item["epoch"]) == int(candidate["epoch"]) for item in candidates):
                        raise RuntimeError(f"candidate epoch already resolved: {candidate['epoch']}")
                    candidates.append(candidate)
                    atomic_json(output / f"review_epoch_{start_epoch:03d}.json", candidate)
                    atomic_json(output / f"decision_epoch_{start_epoch:03d}.json", decision_record)
                    atomic_json(output / "early_stop_state.json", early_stop)
                    atomic_json(candidates_path, candidates)
                    atomic_json(decisions_path, decisions)
                    resolved_payload = checkpoint_payload(
                        start_epoch, model, discriminator, model_optimizer, d_optimizer,
                        config, resume.get("metrics"), decisions, sampler_rng, early_stop,
                        job_path, device_name, pending_agent=None,
                    )
                    atomic_torch_save(latest, enrich_rng(resolved_payload, common))
                else:
                    selection_decision = agent_decision
                    selection_pending = pending
        else:
            # The pre-run is deliberately completed before formal optimizers are
            # created, and its temporary probes cannot alter the formal model or
            # sampler RNG. Luna initialization is a real handoff: epoch 1 does
            # not start until the durable thread returns its decision.
            prerun = run_prerun(
                gan, model, discriminator, dataset, source_names, target_names,
                review_target_names, sampler_rng, config["prior_clipped_log_ratio"], device,
            )
            initial_target_metrics = full_target_prediction_metrics(
                gan, common, model, Path(job["target_root"]), full_names, device,
            )
            prerun["target"].update(copy.deepcopy(initial_target_metrics))
            prerun["target"]["metric_provenance"] = (
                "target_full_list_eval_mode_hard_argmax_valid_pixels"
            )
            atomic_json(output / "agent/prerun/prerun_report.json", prerun)
            torch.manual_seed(SEED)
            discriminator = FineGrainedDomainDiscriminator(
                in_channels=768, num_classes=4, hidden_channels=(256, 128),
            ).to(device)
            model_optimizer, d_optimizer = create_formal_optimizers()
            review0 = gan.review_diagnostics(model, review_target_names)
            apply_full_target_prediction_metrics(review0, initial_target_metrics)
            challenger0 = fada_challengers(gan, model, discriminator, device)
            decisions = []
            early_stop = new_early_stop_state()
            baseline_checkpoint = checkpoints / "accepted_epoch_000.pt"
            baseline_payload = checkpoint_payload(
                0, model, discriminator, model_optimizer, d_optimizer,
                config, None, decisions, sampler_rng, early_stop, job_path, device_name,
            )
            atomic_torch_save(baseline_checkpoint, enrich_rng(baseline_payload, common))
            candidates = [{
                "epoch": 0, "checkpoint": str(baseline_checkpoint.resolve()),
                "checkpoint_sha256": sha256(baseline_checkpoint),
                "review": review0, "challenger": challenger0, "metrics": None,
                "eligible_for_selection": True,
                "selection_role": "safe pretrained fallback if every adapted checkpoint fails a hard gate",
            }]
            atomic_json(output / "review_epoch_000.json", candidates[0])
            atomic_json(candidates_path, candidates)
            init_request = build_initialization_request(
                job_id=job["job_id"],
                job_manifest_sha256=sha256(job_path),
                prior_lock_sha256=config["planet_prior_lock_sha256"],
                pre_run_report=prerun,
                anchor_config=base["anchors"],
            )
            pending = pending_agent_record(output, init_request, 0)
            pending["pre_run_report"] = str((output / "agent/prerun/prerun_report.json").resolve())
            pending["candidate"] = candidates[0]
            latest_payload = checkpoint_payload(
                0, model, discriminator, model_optimizer, d_optimizer,
                config, None, decisions, sampler_rng, early_stop, job_path, device_name,
                pending_agent=pending,
            )
            atomic_torch_save(latest, enrich_rng(latest_payload, common))
            write_status(status_path, agent_status_payload(job, pending, device_name, slot, 0))
            print(json.dumps({
                "job": job["job_id"], "state": "awaiting_luna_initialization", "epoch": 0,
                "agent_model": AGENT_MODEL, "agent_reasoning_effort": AGENT_REASONING_EFFORT,
                "agent_request": pending["request"], "agent_request_sha256": pending["request_sha256"],
            }), flush=True)
            handoff_or_stage_succeeded = True
            return {
                "state": "awaiting_luna_initialization", "epoch": 0,
                "agent_request": pending["request"], "agent_request_sha256": pending["request_sha256"],
            }
        baseline_c2 = float(review0["source"]["f1_per_class"][2])
        completed_epoch = start_epoch
        for epoch in remaining_epochs(start_epoch, early_stop):
            should_review = epoch % REVIEW_INTERVAL == 0 or epoch == MAX_EPOCH
            review_accumulator = ReviewAccumulator() if should_review else None
            metrics = train_epoch(
                gan, model, discriminator, model_optimizer, d_optimizer, dataset,
                len(source_names), len(target_names), config, device, sampler_rng,
                review_accumulator=review_accumulator,
            )
            metrics.update({
                "epoch": epoch,
                "protocol_version": PROTOCOL_VERSION,
                "planet_used": "target_fada_class_encoding_only",
                "planet_prior_lock_sha256": config["planet_prior_lock_sha256"],
            })
            safety = safety_action(config, metrics, epoch)
            if safety:
                decisions.append(safety)
                atomic_json(output / f"decision_epoch_{epoch:03d}_safety.json", safety)
                if review_accumulator is not None:
                    review_accumulator.safety_events.append(dict(safety))
            review = None
            accumulated_review = None
            if should_review and config.get("adversarial_state", "active") == "paused":
                # PAUSED is the deliberate exception: it has no formal
                # adversarial update, so retain the no-update resume probe.
                metrics.update(review_only_gradient_probe(
                    gan, model, discriminator, dataset, len(source_names), len(target_names),
                    config, device, epoch,
                ))
            if should_review and review_accumulator is not None:
                if config.get("adversarial_state", "active") == "active":
                    review, accumulated_review = finalize_review_epoch(
                        common, model, Path(job["source"]["source_valid"]),
                        review_accumulator, epoch, config,
                    )
                else:
                    # Keep the established independent diagnostic path for
                    # paused state, while exposing the in-epoch source/D
                    # statistics and the special resume diagnostic separately.
                    review = gan.review_diagnostics(model, review_target_names)
                    accumulated_review = review_accumulator.finalize(
                        epoch=epoch, config=config,
                        snd_fn=common.old_diag.soft_neighborhood_density,
                    )
                    review.update({
                        "source_training": accumulated_review["source"],
                        "discriminator_training": accumulated_review["discriminator"],
                        "gradient_diagnostics": accumulated_review["gradients"],
                        "training": accumulated_review["training"],
                        "metric_provenance": {
                            **accumulated_review["metric_provenance"],
                            "target_training": "existing_paused_diagnostic",
                            "extra_active_review_batch": False,
                        },
                        "safety_events": accumulated_review["safety_events"],
                    })
                # Keep the target quantity identical before and after PAUSE_ADV.
                # Legacy active reviews used in-epoch hard argmax, while paused
                # reviews used mean softmax on the first 128 patches.
                target_prediction_metrics = full_target_prediction_metrics(
                    gan, common, model, Path(job["target_root"]), full_names, device,
                )
                apply_full_target_prediction_metrics(review, target_prediction_metrics)
                accumulated_review["metric_provenance"].update(
                    review["metric_provenance"]
                )
                gradient = accumulated_review["gradients"]
                metrics["review_metrics_finalized"] = True
                metrics["review_metric_provenance"] = accumulated_review["metric_provenance"]
                metrics["gradient_diagnostics"] = gradient
                if gradient["cls_grad_median"] is not None:
                    metrics["cls_grad"] = gradient["cls_grad_median"]
                    metrics["adv_grad"] = gradient["adv_grad_median"]
                    metrics["effective_adv_to_cls_grad_ratio"] = gradient["effective_adv_to_cls_median"]
                    metrics["raw_adv_to_cls_grad_ratio"] = gradient["raw_adv_to_cls_median"]
            last_metrics = metrics
            with (output / "training_log.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            completed_epoch = epoch
            if should_review:
                if review is None:
                    raise RuntimeError("review epoch completed without finalized review metrics")
                # Keep the compact, detached review snapshot as a standalone
                # artifact.  This is written before the checkpoint and before
                # the Luna request, so a timeout can be reconciled without
                # recomputing the epoch or sampling another batch.
                atomic_json(output / f"review_metrics_epoch_{epoch:03d}.json", {
                    "epoch": epoch,
                    "review": review,
                    "metrics": metrics,
                })
                challenger = fada_challengers(gan, model, discriminator, device)
                accepted = checkpoints / f"accepted_epoch_{epoch:03d}.pt"
                candidate = {
                    "epoch": epoch,
                    "checkpoint": str(accepted.resolve()),
                    "checkpoint_sha256": None,
                    "review": review,
                    "challenger": challenger,
                    "metrics": metrics,
                }
                candidate["eligible_for_selection"] = bool(
                    float(review["source"]["f1_per_class"][2]) >= baseline_c2 - 0.05
                    and target_distribution_valid(review)
                    and np.isfinite(float(metrics["effective_adv_to_cls_grad_ratio"]))
                    and float(metrics["effective_adv_to_cls_grad_ratio"]) <= 0.5
                )
                candidate["selection_role"] = "adapted hard-safe candidate"
                recent_safety = any(
                    item.get("epoch", -1000) > epoch - 3 * REVIEW_INTERVAL
                    and (
                        item.get("action") == "PER_EPOCH_GRADIENT_SAFETY_FUSE"
                        or "PER_EPOCH_GRADIENT_SAFETY_FUSE" in item.get("action", [])
                    )
                    for item in decisions
                )
                proposed_early_stop, deterministic_stop_eligible = update_early_stop(
                    early_stop, candidate, baseline_c2, recent_safety_fuse=recent_safety,
                )
                proposed_early_stop["deterministic_plateau_stop_eligible"] = bool(deterministic_stop_eligible)
                proposed_early_stop["stop"] = False
                proposed_early_stop["stop_reason"] = None
                accepted_payload = checkpoint_payload(
                    epoch, model, discriminator, model_optimizer, d_optimizer,
                    config, metrics, decisions, sampler_rng, proposed_early_stop,
                    job_path, device_name,
                )
                atomic_torch_save(accepted, enrich_rng(accepted_payload, common))
                candidate["checkpoint_sha256"] = sha256(accepted)
                request = build_review_request(
                    job_id=job["job_id"],
                    job_manifest_sha256=sha256(job_path),
                    prior_lock_sha256=config["planet_prior_lock_sha256"],
                    checkpoint_sha256=candidate["checkpoint_sha256"],
                    candidate=candidate,
                    baseline_class2_f1=baseline_c2,
                    config=config,
                    proposed_early_stop=proposed_early_stop,
                    prior_reviews=candidates,
                    prior_decisions=decisions,
                    safety_fuse_applied_this_epoch=safety is not None,
                    previous_decision_sha256=last_persistent_agent_decision_hash(output, decisions),
                    previous_expectation=(decisions[-1].get("next_stage_expectation") if decisions else None),
                )
                pending = pending_agent_record(output, request, epoch)
                pending.update({
                    "candidate": candidate,
                    "proposed_early_stop_state": proposed_early_stop,
                })
                latest_payload = checkpoint_payload(
                    epoch, model, discriminator, model_optimizer, d_optimizer,
                    config, metrics, decisions, sampler_rng, early_stop,
                    job_path, device_name, pending_agent=pending,
                )
                atomic_torch_save(latest, enrich_rng(latest_payload, common))
                atomic_json(output / f"review_epoch_{epoch:03d}_pending.json", {
                    **candidate,
                    "agent_decision_status": "pending",
                    "agent_request": pending["request"],
                    "agent_request_sha256": pending["request_sha256"],
                })
                write_status(status_path, agent_status_payload(job, pending, device_name, slot, epoch))
                print(json.dumps({
                    "job": job["job_id"],
                    "state": "awaiting_luna_review",
                    "epoch": epoch,
                    "agent_model": AGENT_MODEL,
                    "agent_reasoning_effort": AGENT_REASONING_EFFORT,
                    "agent_request": pending["request"],
                    "agent_request_sha256": pending["request_sha256"],
                }), flush=True)
                handoff_or_stage_succeeded = True
                return {
                    "state": "awaiting_luna_review",
                    "epoch": epoch,
                    "agent_request": pending["request"],
                    "agent_request_sha256": pending["request_sha256"],
                }
            latest_payload = checkpoint_payload(
                epoch, model, discriminator, model_optimizer, d_optimizer,
                config, metrics, decisions, sampler_rng, early_stop, job_path, device_name,
            )
            atomic_torch_save(latest, enrich_rng(latest_payload, common))
            write_status(status_path, {
                "job_id": job["job_id"], "state": "training", "worker": os.getpid(),
                "hostname": socket.gethostname(), "gpu": device_name,
                "gpu_slot": slot,
                "heartbeat_epoch": epoch, "max_epoch": MAX_EPOCH,
                "early_stop": False, "last_metrics": metrics,
            })
            print(json.dumps({
                "job": job["job_id"], "epoch": epoch, "max_epoch": MAX_EPOCH,
                "steps": steps, "source_coverage": metrics["source_unique_coverage"],
                "target_coverage": metrics["target_unique_coverage"],
                "gradient_ratio": metrics["effective_adv_to_cls_grad_ratio"],
            }), flush=True)
        if not bool(early_stop.get("stop")):
            raise RuntimeError("training loop ended without a Luna review stop decision")
        if selection_decision is None:
            selection_request = build_selection_request(
                job_id=job["job_id"],
                job_manifest_sha256=sha256(job_path),
                prior_lock_sha256=config["planet_prior_lock_sha256"],
                stopped_after_epoch=completed_epoch,
                stop_reason=early_stop["stop_reason"],
                candidates=candidates,
                baseline_class2_f1=baseline_c2,
                decision_history=decisions,
                current_config=config,
                previous_decision_sha256=last_persistent_agent_decision_hash(output, decisions),
            )
            selection_pending = pending_agent_record(
                output, selection_request, completed_epoch,
            )
            latest_payload = checkpoint_payload(
                completed_epoch, model, discriminator, model_optimizer, d_optimizer,
                config, last_metrics,
                decisions, sampler_rng, early_stop, job_path, device_name,
                pending_agent=selection_pending,
            )
            atomic_torch_save(latest, enrich_rng(latest_payload, common))
            write_status(
                status_path,
                agent_status_payload(job, selection_pending, device_name, slot, completed_epoch),
            )
            handoff_or_stage_succeeded = True
            return {
                "state": "awaiting_luna_selection",
                "epoch": completed_epoch,
                "agent_request": selection_pending["request"],
                "agent_request_sha256": selection_pending["request_sha256"],
            }
        eligible = [item for item in candidates if item.get("eligible_for_selection")]
        if not eligible:
            raise RuntimeError("no eligible reviewed checkpoint")
        scores = {str(item["epoch"]): candidate_score(item, baseline_c2) for item in eligible}
        selected_epoch = int(selection_decision["selected_epoch"])
        selected = next(item for item in eligible if int(item["epoch"]) == selected_epoch)
        selected_source = Path(selected["checkpoint"])
        if sha256(selected_source) != selected["checkpoint_sha256"]:
            raise RuntimeError("selected candidate hash mismatch")
        selected_training_hash = selected["checkpoint_sha256"]
        final = output / "final"
        final.mkdir(parents=True, exist_ok=True)
        selected_checkpoint = final / "selected_checkpoint.pt"
        selected_payload = torch_load(selected_source, "cpu")
        deployment_payload = deployment_checkpoint_payload(
            selected_payload, job_path, selected_training_hash,
        )
        atomic_torch_save(selected_checkpoint, deployment_payload)
        selected_hash = sha256(selected_checkpoint)
        decisions.append({
            **selection_decision,
            "agent_model": AGENT_MODEL,
            "agent_reasoning_effort": AGENT_REASONING_EFFORT,
            "request": selection_pending["request"],
            "request_sha256": selection_pending["request_sha256"],
            "decision": selection_pending["decision"],
            "decision_sha256": sha256(Path(selection_pending["decision"])),
            "provenance": selection_pending["provenance"],
            "provenance_sha256": sha256(Path(selection_pending["provenance"])),
        })
        atomic_json(decisions_path, decisions)
        lock = {
            "protocol_version": PROTOCOL_VERSION,
            "job_id": job["job_id"],
            "job_manifest": str(job_path.resolve()),
            "job_manifest_sha256": sha256(job_path),
            "selected_epoch": selected["epoch"],
            "selected_score": scores[str(selected["epoch"])],
            "candidate_scores": scores,
            "candidate_eligibility": {str(item["epoch"]): bool(item.get("eligible_for_selection")) for item in candidates},
            "selected_is_pretrained_fallback": selected["epoch"] == 0,
            "checkpoint": str(selected_checkpoint.resolve()),
            "checkpoint_sha256": selected_hash,
            "selected_training_checkpoint_sha256": selected_training_hash,
            "deployment_checkpoint_model_only": True,
            "fada_discriminator_included_in_deployment": False,
            "stopped_after_epoch": completed_epoch,
            "stop_reason": early_stop["stop_reason"],
            "early_stop_state": early_stop,
            "selection_stage": "same persistent Codex Luna-high thread over hard-safe source-plus-unlabeled-target candidates",
            "selection_agent_model": AGENT_MODEL,
            "selection_agent_reasoning_effort": AGENT_REASONING_EFFORT,
            "selection_agent_persistent_thread": True,
            "selection_agent_fresh_ephemeral_session": False,
            "selection_agent_request": selection_pending["request"],
            "selection_agent_request_sha256": selection_pending["request_sha256"],
            "selection_agent_decision": selection_pending["decision"],
            "selection_agent_decision_sha256": sha256(Path(selection_pending["decision"])),
            "selection_agent_provenance": selection_pending["provenance"],
            "selection_agent_provenance_sha256": sha256(Path(selection_pending["provenance"])),
            "deterministic_score_was_advisory_not_automatic_argmax": True,
            "planet_used_for_training": "target_fada_class_encoding_only",
            "planet_used_for_agent_selection_or_early_stop": False,
            "planet_used_at_inference": False,
            "planet_prior_lock": str(prior_lock_path.resolve()),
            "planet_prior_lock_sha256": sha256(prior_lock_path),
            "batch_size": BATCH_SIZE,
            "max_epoch": MAX_EPOCH,
            "source_count": len(source_names),
            "target_count": len(target_names),
            "source_train_list_sha256": job["source_qa"]["source_train_sha256"],
            "source_valid_list_sha256": job["source_qa"]["source_valid_sha256"],
            "target_train_list_sha256": job["target_train_list_sha256"],
            "target_full_list_sha256": job["target_full_list_sha256"],
            "steps_per_epoch": steps,
            "full_epoch_sampling": True,
            "load_stats": load_stats,
            "next_required_step": "none; CORAL-P90 is an optional standalone experiment and is disabled by default",
            "optional_coral_p90": "scripts/fit_coral_p90.py and scripts/predict_four_class.py are retained for explicit manual experiments only",
        }
        atomic_json(final_lock, lock)
        pruned = []
        for path in sorted(checkpoints.glob("accepted_epoch_*.pt")):
            pruned.append({"path": str(path), "sha256": sha256(path), "reason": "redundant after final checkpoint hash lock"})
            path.unlink()
        if latest.exists():
            pruned.append({"path": str(latest), "sha256": sha256(latest), "reason": "resume state no longer needed after completion"})
            latest.unlink()
        atomic_json(output / "retention_manifest.json", {
            "policy": "compact formal run",
            "retained": [{"path": str(selected_checkpoint), "sha256": selected_hash}],
            "pruned_generated_checkpoints": pruned,
            "agent_requests_decisions_and_provenance_retained": True,
            "historical_reports_and logs_retained": True,
        })
        write_status(status_path, {
            "job_id": job["job_id"], "state": "training_complete", "worker": os.getpid(),
            "hostname": socket.gethostname(), "gpu": device_name, "gpu_slot": slot,
            "heartbeat_epoch": completed_epoch, "selected_epoch": selected["epoch"],
            "agent_model": AGENT_MODEL,
            "agent_reasoning_effort": AGENT_REASONING_EFFORT,
            "checkpoint_lock": str(final_lock.resolve()),
        })
        handoff_or_stage_succeeded = True
        return {"state": "training_complete", "checkpoint_lock": str(final_lock)}
    except Exception as exc:
        write_status(status_path, {
            "job_id": job.get("job_id"), "state": "failed", "worker": os.getpid(),
            "hostname": socket.gethostname(), "gpu": device_name, "gpu_slot": slot,
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise
    finally:
        if worker_lock.exists():
            worker_lock.unlink()
        if gpu_lease is not None and not handoff_or_stage_succeeded:
            release_pipeline_gpu(gpu_lease, job["job_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--slot", type=int, help="owned GPU slot (0..max-slots-1); assigned by the master")
    parser.add_argument("--max-slots", type=int, help="shared-GPU process cap for this run")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        result = preflight(args.job, require_cuda=False)
        print(json.dumps(result, indent=2))
    else:
        result = train(args.job, args.device, args.slot, args.max_slots)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
