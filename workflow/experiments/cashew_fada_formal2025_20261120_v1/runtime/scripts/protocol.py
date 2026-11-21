#!/usr/bin/env python3
"""Pure protocol helpers for the single-tile FADA + prior workflow."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOL_VERSION = "agent-guide-fada-prior-v4-persistent-luna-b16-e50-fulltile-argmax"
BATCH_SIZE = 16
MAX_EPOCH = 50
REVIEW_INTERVAL = 5
MIN_EARLY_STOP_EPOCH = 10
EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_DELTA = 0.002
STABLE_SCORE_SPAN = 0.005
TARGET_TRAIN_COUNT = 1000
SEED = 16
NUM_CLASSES = 4
IGNORE_INDEX = 10
CORAL_SOURCE_PRECISION = 0.90
CLASS_VALUES = {0, 1, 2, 3}
RASTER_VALUES = {0, 1, 2, 3, 255}

PARAMETER_BOUNDS = {
    "main_lr": [2e-7, 1e-5],
    "D_lr": [5e-7, 2e-5],
    "lambda_adv": [1e-4, 5e-2],
    "D_steps_per_G_step": [1, 4],
    "temperature": [1.2, 3.0],
    "confidence_cap": [0.80, 0.95],
    "main_weight_decay": [1e-4, 3e-2],
    "D_weight_decay": [0.0, 3e-2],
    "prior_strength": [0.0, 0.50],
    "model_adam_beta1": [0.85, 0.95],
    "model_adam_beta2": [0.98, 0.9999],
    "D_adam_beta1": [0.50, 0.95],
    "D_adam_beta2": [0.95, 0.9999],
    "model_adam_eps": [1e-8, 1e-6],
    "D_adam_eps": [1e-8, 1e-6],
}

ANCHOR_CONFIG = {
    "main_lr": 2e-6,
    "D_lr": 2e-6,
    "lambda_adv": 0.008,
    "D_steps_per_G_step": 3,
    "temperature": 1.8,
    "confidence_cap": 0.9,
    "main_weight_decay": 0.01,
    "D_weight_decay": 0.01,
    "prior_strength": 0.25,
    "model_adam_beta1": 0.9,
    "model_adam_beta2": 0.999,
    "D_adam_beta1": 0.9,
    "D_adam_beta2": 0.999,
    "model_adam_eps": 1e-8,
    "D_adam_eps": 1e-8,
}


def canonical_training_config() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "variant": "fada_2k_stage4",
        "batch_size": BATCH_SIZE,
        "max_epoch": MAX_EPOCH,
        "review_interval": REVIEW_INTERVAL,
        "minimum_early_stop_epoch": MIN_EARLY_STOP_EPOCH,
        "early_stop_patience_reviews": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "early_stop_stable_score_span": STABLE_SCORE_SPAN,
        "seed": SEED,
        "adaptive_controller": True,
        "anchors": copy_json(ANCHOR_CONFIG),
        "bounds": copy_json(PARAMETER_BOUNDS),
        "pre_run": {
            "independent_seed": SEED + 1001,
            "source_batches": 8,
            "target_batches": 8,
            "temperature_candidates": [1.2, 1.5, 1.8, 2.2, 2.6, 3.0],
            "main_lr_candidates": [2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5],
            "D_lr_candidates": [5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5],
            "main_lr_probe_steps": 8,
            "D_lr_probe_steps": 12,
            "gradient_warmup_steps": 50,
            "gradient_target_ratios": [0.08, 0.10, 0.12, 0.15, 0.20, 0.25],
        },
        "discriminator": {
            "kind": "single_fully_convolutional_2k",
            "feature_input": "Swin Stage4 Bx768xHfxWf",
            "num_classes": NUM_CLASSES,
            "output_channels": 2 * NUM_CLASSES,
            "channel_order": "source_class_0..3,target_class_0..3",
            "hidden_channels": [256, 128],
        },
        "class_knowledge": {
            "kind": "softmax_detached",
            "temperature": 1.8,
            "confidence_cap": 0.9,
            "source_prior_adjustment": False,
            "target_prior_adjustment": True,
        },
        "source_objective": "balanced_binary_nll_class2_vs_rest",
        "source_binary_loss_weights": {"negative": 0.6311714517, "positive": 2.4059025179},
        "full_epoch_sampling": True,
        "target_train_count": TARGET_TRAIN_COUNT,
        "target_review_metric": {
            "scope": "target_full_list",
            "model_mode": "eval",
            "inference_batch_size": BATCH_SIZE,
            "class_distribution": "hard_argmax_valid_pixel_fraction",
            "mean_softmax_field": "mean_softmax_distribution",
            "support_definition": "pixels where not all input bands are zero",
            "same_definition_when_adversarial_active_or_paused": True,
        },
        "primary_output": "selected_model_checkpoint",
        "optional_postprocessing": {
            "coral_p90_default_enabled": False,
            "coral_p90_mode": "standalone_manual_only",
            "coral_source_precision_floor": CORAL_SOURCE_PRECISION,
            "coral_target_samples_per_patch": 512,
        },
        "planet_training": "target_fada_class_encoding_only",
        "planet_agent_decisions": False,
        "planet_checkpoint_selection": False,
        "planet_threshold_selection": False,
        "planet_inference_logit_adjustment": False,
        "agent_guide": {
            "controller": "persistent_codex_luna_per_tile",
            "model": "gpt-5.6-luna",
            "reasoning_effort": "high",
            "sandbox": "read-only",
            "persistent_thread": True,
            "fresh_ephemeral_session": False,
            "review_interval": REVIEW_INTERVAL,
            "skill": "cashew-fada-agent-controller",
            "skill_path": ".codex/skills/cashew-fada-agent-controller/SKILL.md",
            "output_schema": "references/agent_decision.schema.json",
            "review_handoff": True,
            "checkpoint_selection_by_agent": True,
            "hard_safety_envelope_overridable": False,
        },
    }


def copy_json(value: Any) -> Any:
    """Copy protocol constants without introducing a mutable shared object."""
    return json.loads(json.dumps(value))


def canonical_prior_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return and validate the tile-prior policy.

    Tau is deliberately conservative because the available Planet validation is
    binary and cannot establish four-class reliability.  A request may override
    only the documented numeric controls; the one-time application policy is
    fixed by the protocol.
    """
    config: dict[str, Any] = {
        "tau": [0.25, 0.25, 0.25, 0.25],
        "epsilon": 1e-6,
        "dirichlet_pseudocount": 1.0,
        "log_ratio_clip": [-2.0, 2.0],
        # Retained solely so pre-existing immutable jobs and prior locks stay
        # readable.  planet_prior.py no longer uses this value as a gate.
        "minimum_planet_valid_fraction": 0.0,
        "resampling": "nearest_to_target_patch_grid",
        "zero_class_policy": "finite_smoothing_and_clip_never_delete",
        "apply_to": "training_target_fada_class_encoding_only",
        "apply_at_inference": False,
    }
    numeric_overrides = {
        "tau", "epsilon", "dirichlet_pseudocount", "log_ratio_clip",
        "minimum_planet_valid_fraction",
    }
    values = dict(overrides or {})
    unknown = sorted(set(values) - set(config))
    if unknown:
        raise ValueError(f"unknown prior config overrides: {unknown}")
    for key in set(values) - numeric_overrides:
        if values[key] != config[key]:
            raise ValueError(f"fixed prior policy field cannot be overridden: {key}")
    config.update({key: value for key, value in values.items() if key in numeric_overrides})
    tau = np.asarray(config["tau"], dtype=float)
    clip = np.asarray(config["log_ratio_clip"], dtype=float)
    if tau.shape != (NUM_CLASSES,) or not np.isfinite(tau).all() or (tau < 0).any() or (tau > 1).any():
        raise ValueError("prior tau must contain four finite values in [0,1]")
    if clip.shape != (2,) or not np.isfinite(clip).all() or not clip[0] < clip[1]:
        raise ValueError("log_ratio_clip must be two increasing finite values")
    if not (0 < float(config["epsilon"]) < 1):
        raise ValueError("epsilon must be in (0,1)")
    if float(config["dirichlet_pseudocount"]) <= 0:
        raise ValueError("dirichlet_pseudocount must be positive")
    if not (0 <= float(config["minimum_planet_valid_fraction"]) <= 1):
        raise ValueError("minimum_planet_valid_fraction must be in [0,1]")
    config["tau"] = tau.tolist()
    config["log_ratio_clip"] = clip.tolist()
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def write_status(path: Path, payload: dict[str, Any]) -> None:
    value = dict(payload)
    value["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, value)


def read_lines(path: Path) -> list[str]:
    return [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]


def steps_per_epoch(source_count: int, target_count: int, batch_size: int = BATCH_SIZE,
                    stable_pair_count: int | None = None) -> int:
    if batch_size != BATCH_SIZE:
        raise ValueError(f"formal protocol requires batch_size={BATCH_SIZE}, got {batch_size}")
    counts = [source_count, target_count]
    if stable_pair_count is not None:
        if stable_pair_count <= 0:
            raise ValueError("stable_pair_count must be positive when provided")
        counts.append(stable_pair_count)
    if min(counts) <= 0:
        raise ValueError(f"all stream counts must be positive: {counts}")
    return max(math.ceil(count / batch_size) for count in counts)


def epoch_batches(count: int, batch_size: int, steps: int,
                  rng: np.random.RandomState) -> list[list[int]]:
    expected_minimum = math.ceil(count / batch_size)
    if batch_size != BATCH_SIZE:
        raise ValueError(f"formal protocol requires batch_size={BATCH_SIZE}")
    if steps < expected_minimum:
        raise ValueError(f"steps={steps} cannot cover count={count} at batch={batch_size}")
    permutation = rng.permutation(count).tolist()
    required = steps * batch_size
    stream = (permutation * math.ceil(required / count))[:required]
    batches = [stream[index:index + batch_size] for index in range(0, required, batch_size)]
    if len(batches) != steps or len(set(item for batch in batches for item in batch)) != count:
        raise RuntimeError("full-epoch coverage invariant failed")
    return batches


def candidate_score(candidate: dict[str, Any], baseline_class2_f1: float) -> float:
    review = candidate["review"]
    challenger = candidate["challenger"]
    source_class2 = float(review["source"]["f1_per_class"][2])
    source_macro = float(review["source"]["macro_f1"])
    snd = float(review["target_unsupervised"]["snd"])
    fresh_domain_accuracy = float(challenger["fresh_global_mlp"]["test"]["accuracy"])
    collapse_penalty = max(0.0, baseline_class2_f1 - source_class2 - 0.05) * 5.0
    return source_class2 + 0.25 * source_macro + 0.01 * snd - 0.02 * fresh_domain_accuracy - collapse_penalty


def target_distribution_valid(review: dict[str, Any]) -> bool:
    values = np.asarray(review["target_unsupervised"]["class_distribution"], dtype=float)
    return bool(
        values.shape == (NUM_CLASSES,)
        and np.isfinite(values).all()
        and abs(float(values.sum()) - 1.0) <= 1e-3
        and float(values.max()) < 0.98
        and int((values > 0.01).sum()) >= 2
    )


def new_early_stop_state() -> dict[str, Any]:
    return {
        "enabled": True,
        "minimum_epoch": MIN_EARLY_STOP_EPOCH,
        "patience_reviews": EARLY_STOP_PATIENCE,
        "min_delta": EARLY_STOP_MIN_DELTA,
        "stable_score_span": STABLE_SCORE_SPAN,
        "review_scores": [],
        "bad_reviews": 0,
        "best_score": -1e30,
        "best_epoch": None,
        "stop": False,
        "stop_reason": None,
    }


def update_early_stop(state: dict[str, Any], candidate: dict[str, Any],
                      baseline_class2_f1: float, recent_safety_fuse: bool = False) -> tuple[dict[str, Any], bool]:
    state = json.loads(json.dumps(state))
    epoch = int(candidate["epoch"])
    score = candidate_score(candidate, baseline_class2_f1)
    state["review_scores"].append({"epoch": epoch, "score": score})
    if score > float(state["best_score"]) + EARLY_STOP_MIN_DELTA:
        state["best_score"] = score
        state["best_epoch"] = epoch
        state["bad_reviews"] = 0
    else:
        state["bad_reviews"] = int(state["bad_reviews"]) + 1
    recent = state["review_scores"][-3:]
    stable = len(recent) == 3 and max(x["score"] for x in recent) - min(x["score"] for x in recent) <= STABLE_SCORE_SPAN
    source_class2 = float(candidate["review"]["source"]["f1_per_class"][2])
    gradient_ratio = float(candidate["metrics"]["effective_adv_to_cls_grad_ratio"])
    gates = {
        "minimum_epoch": epoch >= MIN_EARLY_STOP_EPOCH,
        "plateau_patience": int(state["bad_reviews"]) >= EARLY_STOP_PATIENCE,
        "three_review_score_span": stable,
        "source_class2_protected": source_class2 >= baseline_class2_f1 - 0.05,
        "gradient_ratio_finite_and_safe": bool(np.isfinite(gradient_ratio) and gradient_ratio <= 0.5),
        "target_distribution_not_collapsed": target_distribution_valid(candidate["review"]),
        "no_recent_safety_fuse": not recent_safety_fuse,
    }
    stop = all(gates.values())
    state.update({
        "last_epoch": epoch,
        "last_score": score,
        "last_gates": gates,
        "stop": stop,
        "stop_reason": "deterministic plateau gates satisfied; awaiting Luna decision" if stop else None,
    })
    return state, stop


def relative_logits(logits: np.ndarray) -> np.ndarray:
    if logits.ndim < 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"expected logits shaped [N,4,...], got {logits.shape}")
    return np.stack(
        [logits[:, 1] - logits[:, 0], logits[:, 2] - logits[:, 0], logits[:, 3] - logits[:, 0]],
        axis=-1,
    )


def matrix_power_symmetric(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-6, None)
    return (vectors * np.power(values, power)) @ vectors.T


def coral_parameters(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError(f"CORAL expects Nx3 source/target features, got {source.shape}/{target.shape}")
    if len(source) < 2 or len(target) < 2 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("CORAL requires at least two finite source and target feature rows")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_cov = np.cov(source, rowvar=False) + np.eye(3) * 1e-5
    target_cov = np.cov(target, rowvar=False) + np.eye(3) * 1e-5
    transform = matrix_power_symmetric(target_cov, -0.5) @ matrix_power_symmetric(source_cov, 0.5)
    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "source_cov": source_cov,
        "target_cov": target_cov,
        "transform": transform,
    }


def coral_apply(features: np.ndarray, parameters: dict[str, np.ndarray]) -> np.ndarray:
    return (features - parameters["target_mean"]) @ parameters["transform"] + parameters["source_mean"]


def precision_constrained_threshold(scores: np.ndarray, binary_truth: np.ndarray,
                                    precision_floor: float = CORAL_SOURCE_PRECISION) -> tuple[float, dict[str, Any]]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    truth = np.asarray(binary_truth, dtype=np.uint8).reshape(-1)
    if len(scores) != len(truth) or len(scores) == 0 or set(np.unique(truth)) - {0, 1}:
        raise ValueError("scores/truth must be non-empty aligned binary arrays")
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    positives = truth[order].astype(np.int64)
    tp = np.cumsum(positives)
    fp = np.cumsum(1 - positives)
    group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    precision = tp[group_ends] / np.maximum(1, tp[group_ends] + fp[group_ends])
    eligible = group_ends[precision >= precision_floor]
    if len(eligible) == 0:
        threshold = float(np.nextafter(scores.max(), np.inf))
    else:
        threshold = float(sorted_scores[int(eligible[-1])])
    prediction = scores >= threshold
    tp_value = int(np.count_nonzero(prediction & (truth == 1)))
    fp_value = int(np.count_nonzero(prediction & (truth == 0)))
    fn_value = int(np.count_nonzero(~prediction & (truth == 1)))
    metrics = {
        "threshold": threshold,
        "tp": tp_value,
        "fp": fp_value,
        "fn": fn_value,
        "precision": tp_value / max(1, tp_value + fp_value),
        "recall": tp_value / max(1, tp_value + fn_value),
        "f1": 2 * tp_value / max(1, 2 * tp_value + fp_value + fn_value),
        "precision_floor": precision_floor,
    }
    return threshold, metrics


def compose_four_class(logits: np.ndarray, class2_score: np.ndarray, threshold: float,
                       support: np.ndarray | None = None) -> np.ndarray:
    """Apply CORAL-P90 only to class 2 and preserve the 0/1/3 decision."""
    if logits.shape[0] != NUM_CLASSES:
        raise ValueError(f"expected one 4-class logit map, got {logits.shape}")
    if class2_score.shape != logits.shape[1:]:
        raise ValueError(f"score/logit shape mismatch: {class2_score.shape}/{logits.shape}")
    non2_classes = np.asarray((0, 1, 3), dtype=np.uint8)
    non2 = non2_classes[np.argmax(logits[[0, 1, 3]], axis=0)]
    prediction = np.where(class2_score >= threshold, 2, non2).astype(np.uint8)
    if support is not None:
        if support.shape != prediction.shape:
            raise ValueError("support/prediction shape mismatch")
        prediction[~support] = 255
    values = set(np.unique(prediction).tolist())
    if values - RASTER_VALUES:
        raise RuntimeError(f"invalid four-class output values: {values}")
    return prediction


def assert_unique(items: Iterable[str], label: str) -> None:
    values = list(items)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate entries")
