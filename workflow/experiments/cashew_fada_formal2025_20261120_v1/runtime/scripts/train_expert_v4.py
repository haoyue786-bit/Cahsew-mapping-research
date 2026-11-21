#!/usr/bin/env python3
"""Training Expert v4 worker; formal jobs are validated before runtime loading."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from fada import FineGrainedDomainDiscriminator, resize_segmentation_logits
from fada_expert_v4 import GlobalDomainDiscriminator, class_knowledge_v4, global_loss, global_target_adversarial_loss, local_loss, local_target_adversarial_loss
from expert_policy_v4 import PARAM_BOUNDS, apply_agent_early_stopping_policy, flat, normalize, request_base
from planet_risk_v4 import summarize as planet_summary
from protocol import BATCH_SIZE, epoch_batches, read_lines, sha256
from train_tile import (balanced_binary_source_loss, four_class_source_loss, configure_runtime, diagnostic_target_names as _diagnostic_target_names,
                        apply_full_target_prediction_metrics, enrich_rng,
                        full_target_prediction_metrics as _full_target_prediction_metrics, model_state_digest, torch_load, atomic_torch_save)
from training_experience_v4 import recent_for_job, record, record_once
from formal_v4 import max_epoch_for_job, validate_formal_job
from agentguide_policy_v1 import enabled as distilled_policy_enabled, enrich as enrich_distilled_packet
from architecture import feature_learning_parameters, make_model as make_architecture_model, model_parameter_groups
from schedule_v4 import apply_to_optimizers, default_state, effective, update_state
from expert_policy_v4 import parameter_bounds
import supplementary_control_v4 as negative
from contextlib import nullcontext
from human_negative_label_paths import resolve_label_path
from evaluation_sample_v1 import EvaluationNames, names_for_job, measure as measure_evaluation
from initialization_metric_evidence_v001 import initialization_baseline_evidence, initialization_probe_evidence
import human_polygon_v1 as human
import gradient_control_v1 as gradient_control
import cosine_finish_v1 as finish


ROOT = Path(__file__).resolve().parents[1]
DISPATCH = Path(__file__).resolve().parent / "memory_dispatch_v1.py"
SCHEMA = ROOT / "references/agent_decision_v4.schema.json"
SOURCE_COLLAPSE_ABSOLUTE_F1 = 0.35
SOURCE_COLLAPSE_DROP_FROM_BASELINE = 0.30
DEFAULT_MAX_EPOCH = 200
DEFAULT_MINIMUM_STOP_EPOCH = 0
from minimum_training_v1 import floor as minimum_training_floor, guard as guard_minimum_training
REGION_STABILITY_WINDOW_REVIEWS = 4
REGION_STABILITY_MAX_FRACTION_RANGE = 0.015
REGION_STABILITY_MIN_PIXEL_AGREEMENT = 0.97
REGION_STABILITY_MIN_CLASS2_IOU = 0.90


class AccountUsageLimitExceeded(RuntimeError):
    """Agent account quota exhausted; training state must stay resumable."""


def agent_dispatch_failure(request_path: Path) -> dict:
    """Read only the failure record for this exact request, not stale retries."""
    path = request_path.with_name("dispatch_failure.json")
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return payload if payload.get("request_sha256") == sha256(request_path) else {}


def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if isinstance(value, str):
        tmp.write_text(value)
    else:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def load_research_candidate_window(path: Path) -> list[int] | None:
    """Read a persisted label-free window without making resume fragile.

    A malformed or empty artifact is treated as absent; the selection policy
    then records its normal runtime fallback instead of widening or crashing.
    """
    try:
        payload = json.loads(path.read_text())
        values = payload.get("candidate_checkpoint_window")
        if not isinstance(values, list) or not values:
            return None
        result = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            epoch = value
            if epoch < 0 or epoch in result:
                continue
            result.append(epoch)
        return result or None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def config_anchor(variant: dict) -> dict:
    return {"lambda_adv": 0.004, "main_lr": 2e-6, "D_lr": 2e-6, "D_steps_per_G_step": 2,
            "temperature": 1.8, "confidence_cap": float(variant.get("confidence_cap", 0.9)),
            "main_weight_decay": 0.01, "D_weight_decay": 0.01, "adversarial_state": "active",
            "schedule_state": default_state()}


def apply_config(config, decision, model_optimizer, local_optimizer, global_optimizer):
    from fixed_supplementary_default_v1 import record_agent_intervention
    record_agent_intervention(config, decision)
    before = copy.deepcopy(config)
    after = decision["intervention"]["parameters_after"]
    for key in parameter_bounds(config):
        config[key] = after[key]
    config["adversarial_state"] = after["adversarial_state"]
    config["class_knowledge"]["temperature"] = config["temperature"]
    config["class_knowledge"]["confidence_cap"] = config["confidence_cap"]
    finish.rebase(config, before)
    update_state(config, decision["intervention"], int(decision.get("epoch", 0)))
    apply_to_optimizers(config, int(decision.get("epoch", 0)), model_optimizer, local_optimizer, global_optimizer)
    return {"before": before, "after": copy.deepcopy(config), "changed_parameters": decision["intervention"]["changed_parameters"]}


def fixed_ablation_decision(config, epoch, candidates, fixed_schedule):
    """Return deterministic decisions for an explicitly isolated non-formal ablation."""
    stop_epoch = int(fixed_schedule["stop_epoch"])
    values = {key: config[key] for key in parameter_bounds(config)}
    values["adversarial_state"] = config["adversarial_state"]
    if epoch == "selection":
        eligible = {int(item["epoch"]) for item in candidates}
        if stop_epoch not in eligible:
            raise RuntimeError({"fixed_ablation_stop_checkpoint_missing": stop_epoch, "eligible": sorted(eligible)})
        action, keep_training, selected_epoch = "SELECT_CHECKPOINT", False, stop_epoch
        state = f"fixed_ablation_terminal_epoch_{stop_epoch}"
    elif int(epoch) >= stop_epoch:
        action, keep_training, selected_epoch = "STOP", False, None
        state = f"fixed_ablation_stop_epoch_{stop_epoch}"
    else:
        action, keep_training, selected_epoch = "CONTINUE", True, None
        state = "fixed_ablation_controls_unchanged"
    return {
        "action": action, "continue_training": keep_training, "selected_epoch": selected_epoch,
        "reason_codes": ["ISOLATED_FIXED_SCHEDULE_ABLATION", "NO_AGENT_TUNING"],
        "analysis_summary": "Declared isolated ablation: all Agent-adjustable controls remain fixed; only the manifest variable differs between arms.",
        "previous_hypothesis_review": {"status": "not_applicable", "summary": "No Agent intervention occurs in a fixed-control ablation.", "evidence": []},
        "current_state_assessment": {"state": state, "summary": "Controls are held fixed by the ablation contract.", "evidence": []},
        "intervention": {"adversarial_state_after": config["adversarial_state"], "parameter_updates": [],
                         "parameters_before": dict(values), "parameters_after": dict(values),
                         "changed_parameters": [], "emergency": False,
                         "rationale": "Hold fixed configuration for a single-variable source-profile comparison."},
        "experiment": None,
        "next_stage_expectation": {"hypothesis": "Only the declared source profile differs across arms.",
                                   "expected_source": "Source metrics are descriptive, not a ranking objective.",
                                   "expected_target": "Planet F1 is evaluated post hoc only, not used to steer training.",
                                   "expected_discriminator": "Discriminator metrics are descriptive.",
                                   "expected_gradient_ratio": "The measured ratio is descriptive under unchanged controls.",
                                   "failure_triggers": ["Missing terminal checkpoint", "nonfinite training metric", "provenance mismatch between arms"]},
    }


def decide(job, config, variant, planet, epoch, current_review, history, candidates, **kwargs):
    from legacy_agent_guard_v1 import block_legacy
    block_legacy(job, 'decision')
    if job.get("experimental_fixed_schedule"):
        decision=fixed_ablation_decision(config, epoch, candidates, job["experimental_fixed_schedule"])
        if job['experimental_fixed_schedule'].get('version')=='user_group_fixed100_v1':
            active=(epoch=='selection' or int(epoch)>=50) and config['supplementary_training']['count']>0
            after=decision['intervention']['parameters_after']
            after.update(supplementary_loss_weight=.5 if active else 0.,supplementary_positive_scale=.25,
                         supplementary_occurrence_cap=4,supplementary_interval_steps=1,supplementary_horizon_epochs=100)
            decision['intervention']['changed_parameters']=[k for k in parameter_bounds(config) if after[k]!=config[k]]
            decision['epoch']=100 if epoch=='selection' else int(epoch)
            decision['analysis_summary']='User-authorized fixed100 baseline; common constant scalars; no Agent calls; both signs disabled through E50, then w_neg0.5/w_pos0.125, cap4; select E100 only.'
            decision['intervention']['rationale']='Predetermined cohort-wide baseline, not a scientific Agent judgment.'
        return decision, None
    if job.get("planet_decision_reference", True) is False or distilled_policy_enabled(job):
        # Hide whole-tile Planet prevalence and risk metadata from every Agent
        # stage. Supplementary Planet labels remain in their segmentation loss.
        planet = None
    return call_agent(job, config, variant, planet, epoch, current_review, history, candidates, **kwargs)


def selection_metadata_packet(packet, deterministic_fixed_schedule=False):
    """Normalize optional selector metadata without weakening Agent decisions.

    Fixed-schedule ablations deliberately return no metadata packet; their
    selected epoch is already in the deterministic decision.  Agent-backed
    selection must still provide a dictionary, so unexpected missing/malformed
    packets remain a hard error.
    """
    if packet is None and deterministic_fixed_schedule:
        return {}
    if not isinstance(packet, dict):
        raise TypeError("checkpoint selection metadata must be an object")
    return packet


def optimizer_to(optimizer, device: torch.device):
    """Move restored optimizer tensors onto the active training device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def checkpoint_selection_policy(candidates: list[dict], minimum_epoch: int = 0,
                                flexible_trend=False, named_window=None, formal_distilled=False) -> dict:
    """Use source F1 only as a catastrophic-collapse gate, never as ranking."""
    if not candidates:
        raise ValueError("checkpoint selection requires candidates")
    baseline = next((item for item in candidates if int(item["epoch"]) == 0), candidates[0])
    baseline_f1 = float(baseline["source_class2_f1"])
    collapse_floor = max(
        SOURCE_COLLAPSE_ABSOLUTE_F1,
        baseline_f1 - SOURCE_COLLAPSE_DROP_FROM_BASELINE,
    )
    summaries = []
    eligible = []
    stable_active = []
    for item in candidates:
        source_f1 = float(item["source_class2_f1"])
        collapsed = not np.isfinite(source_f1) or source_f1 < collapse_floor
        if not collapsed and int(item["epoch"]) >= int(minimum_epoch):
            eligible.append(int(item["epoch"]))
            review = item.get("review", {})
            if (int(item["epoch"]) > 0 and review.get("adversarial_state") == "active"
                    and (review.get("target_region_stability") or {}).get("full_region_stable")
                    and float((review.get("metrics") or {}).get("adv", 0.0)) > 0
                    and np.isfinite(float((review.get("metrics") or {}).get("effective_gradient_ratio", 0.0)))
                    and float((review.get("metrics") or {}).get("effective_gradient_ratio", 0.0)) > 0):
                stable_active.append(int(item["epoch"]))
        summaries.append({
            "epoch": int(item["epoch"]),
            "target_class2_fraction": float(item["target_class2_fraction"]),
            "target_class2_delta_from_epoch0": (
                float(item["target_class2_fraction"])
                - float(baseline["target_class2_fraction"])
            ),
            "source_class2_f1": source_f1,
            "source_class2_f1_delta_from_epoch0": source_f1 - baseline_f1,
            "source_collapsed": bool(collapsed),
            "diagnostics": {
                key: copy.deepcopy(item.get("review", {}).get(key))
                for key in ("metrics", "source", "target", "fresh_probe", "target_region_stability", "human_supervision_evaluation")
            },
        })
    if flexible_trend or formal_distilled:
        policy_name = (flexible_trend if isinstance(flexible_trend, str)
                       else "trend_judgment_v2")
        policy_tag = "trend_v2" if policy_name == "trend_judgment_v2" else policy_name
        trained = [int(item["epoch"]) for item in candidates
                   if int(item["epoch"]) > 0
                   and int(item["epoch"]) >= int(minimum_epoch)
                   and np.isfinite(float(item["source_class2_f1"]))
                   and float(item["source_class2_f1"]) >= collapse_floor]
        eligible = trained or [int(baseline["epoch"])]
        selection_mode = (("formal_role_phase_coverage_v2_all_noncollapsed_trained_candidates" if trained else "formal_role_phase_coverage_v2_baseline_safety_fallback") if formal_distilled else (f"research_{policy_tag}_all_noncollapsed_trained_candidates" if trained else f"research_{policy_tag}_baseline_safety_fallback"))
    elif stable_active:
        eligible = stable_active
        selection_mode = "verified_stable_active_adversarial_candidates"
    elif eligible:
        trained = [value for value in eligible if value > 0]
        eligible = trained or eligible
        selection_mode = "no_verified_stable_candidate_safety_fallback"
    else:
        # Recovery fallback when every observed checkpoint crossed the
        # catastrophic-collapse gate.  Normal selection accepts all
        # non-collapsed checkpoints, including the epoch-0 baseline.
        eligible = [int(baseline["epoch"])]
        selection_mode = "all_trained_candidates_collapsed_baseline_fallback"
    window_source = "runtime_eligible_trained_checkpoint_set"
    if flexible_trend and named_window is not None:
        requested = []
        for value in named_window:
            try:
                epoch = int(value)
            except (TypeError, ValueError):
                continue
            if epoch not in requested:
                requested.append(epoch)
        narrowed = [epoch for epoch in eligible if epoch in requested]
        if narrowed:
            eligible = narrowed
            selection_mode = selection_mode.replace(
                "all_noncollapsed_trained_candidates", "named_candidate_window"
            )
            window_source = "preceding_agent_named_candidate_window"
        else:
            window_source = "invalid_or_empty_named_window_runtime_fallback"
    # Keep the exact eligible window and its immutable checkpoint provenance in
    # the request. A later selection review must not silently widen this set.
    candidate_records = []
    for item in candidates:
        epoch = int(item["epoch"])
        if epoch not in eligible:
            continue
        record = {"epoch": epoch, "checkpoint": str(item.get("checkpoint", "")),
                  "source_collapsed": False}
        checkpoint_hash = item.get("checkpoint_sha256")
        if checkpoint_hash:
            record["checkpoint_sha256"] = str(checkpoint_hash)
        candidate_records.append(record)
    return {
        "eligible_candidates": eligible,
        "candidate_checkpoint_window": [item["epoch"] for item in candidate_records],
        "candidate_checkpoint_records": candidate_records,
        "candidate_window_provenance": {
            "source": window_source,
            "minimum_epoch": int(minimum_epoch),
            "selection_mode": selection_mode,
            "label_free": True,
        },
        "candidate_summaries": summaries,
        "selection_policy": {
            "source_metric_role": "catastrophic_collapse_gate_only_not_a_ranking_objective",
            "source_baseline_epoch": int(baseline["epoch"]),
            "source_baseline_class2_f1": baseline_f1,
            "source_collapse_absolute_floor": SOURCE_COLLAPSE_ABSOLUTE_F1,
            "source_collapse_drop_from_baseline": SOURCE_COLLAPSE_DROP_FROM_BASELINE,
            "source_collapse_effective_floor": collapse_floor,
            "minimum_selection_epoch": int(minimum_epoch),
            "selection_mode": selection_mode,
            "ranking_instruction": (
                ("Compare role/phase coverage and intervention response across all eligible candidates. Stability is advisory, not an eligibility filter or quality reward. " if formal_distilled else "Prefer verified stable active-adversarial candidates when available. ") +
                "Otherwise explain why stability was not reached and why the safety fallback is chosen. "
                "Do not prefer higher source F1 or Planet agreement as a ranking objective."
            ),
        },
    }


def assess_full_region_stability(current_codes: np.ndarray,
                                 previous_codes: np.ndarray | None,
                                 observations: list[dict],
                                 action_history: list[dict],
                                 epoch: int,
                                 current_fraction: float,
                                 scope: str = "target_full_list_valid_pixels") -> dict:
    """Describe stability over the complete fixed target region.

    ``current_codes`` and ``previous_codes`` contain one argmax class per
    valid pixel in the same full-target-list order. The result is a necessary,
    not sufficient, normal STOP gate.
    """
    recent = list(observations)[-(REGION_STABILITY_WINDOW_REVIEWS - 1):] + [{
        "epoch": int(epoch), "class2_fraction": float(current_fraction),
        "region_comparison": None,
    }]
    result = {
        "scope": scope,
        "measurement_is_sample": scope == "fixed_tile_stratified_sample_valid_pixels",
        "window_epochs": [item["epoch"] for item in recent],
        "required_review_points": REGION_STABILITY_WINDOW_REVIEWS,
        "review_points": len(recent),
        "required_transitions": REGION_STABILITY_WINDOW_REVIEWS - 1,
        "full_region_stable": False,
        "stable_reason": "insufficient_review_window",
        "class2_fraction_range": None,
        "adjacent_comparisons": [],
        "parameter_changes_in_window": [],
    }
    if previous_codes is not None and current_codes.shape == previous_codes.shape:
        current_class2 = current_codes == 2
        previous_class2 = previous_codes == 2
        union = bool(np.logical_or(current_class2, previous_class2).any())
        intersection = int(np.logical_and(current_class2, previous_class2).sum())
        union_count = int(np.logical_or(current_class2, previous_class2).sum())
        result["adjacent_comparisons"].append({
            "from_epoch": int(observations[-1]["epoch"]) if observations else None,
            "to_epoch": int(epoch),
            "all_class_pixel_agreement": float(np.mean(current_codes == previous_codes)),
            "class2_iou": float(intersection / union_count) if union else None,
            "class2_pixels_previous": int(previous_class2.sum()),
            "class2_pixels_current": int(current_class2.sum()),
        })
    elif previous_codes is not None:
        result["stable_reason"] = "incompatible_region_pixel_sequence"

    fractions = [item["class2_fraction"] for item in recent]
    if len(fractions) == REGION_STABILITY_WINDOW_REVIEWS:
        fraction_range = float(max(fractions) - min(fractions))
        result["class2_fraction_range"] = fraction_range
        # Each observation stores the transition from the previous review to
        # itself.  The current comparison is appended below.
        comparison_history = [
            item.get("region_comparison")
            for item in observations[-(REGION_STABILITY_WINDOW_REVIEWS - 1):]
        ] + result["adjacent_comparisons"]
        comparison_history = [item for item in comparison_history if item]
        result["adjacent_comparisons"] = comparison_history
        all_agreement = all(
            float(item["all_class_pixel_agreement"]) >= REGION_STABILITY_MIN_PIXEL_AGREEMENT
            for item in comparison_history
        )
        all_iou = all(
            item["class2_iou"] is not None and float(item["class2_iou"]) >= REGION_STABILITY_MIN_CLASS2_IOU
            for item in comparison_history
        )
        window_start = int(recent[0]["epoch"])
        changes = [
            {
                "decision_epoch": int(item["decision_epoch"]),
                "action": item.get("action"),
                "changed_parameters": list(item.get("changed_parameters", [])),
                "schedule_updates": list(item.get("schedule_updates", [])),
            }
            for item in action_history
            if window_start <= int(item.get("decision_epoch", -1)) < int(epoch)
            and (item.get("changed_parameters") or item.get("schedule_updates") or item.get("action") in {"PAUSE_ADV", "RESUME_ADV"})
        ]
        result["parameter_changes_in_window"] = changes
        result["full_region_stable"] = bool(
            len(comparison_history) == REGION_STABILITY_WINDOW_REVIEWS - 1
            and all_agreement
            and all_iou
            and fraction_range <= REGION_STABILITY_MAX_FRACTION_RANGE
            and not changes
        )
        if result["full_region_stable"]:
            result["stable_reason"] = "whole_region_spatial_and_area_metrics_stable"
        elif changes:
            result["stable_reason"] = "parameter_or_adversarial_state_change_in_window"
        elif not all_agreement or not all_iou:
            result["stable_reason"] = "whole_region_predictions_still_moving"
        else:
            result["stable_reason"] = "whole_region_class2_fraction_still_moving"
    return result


def supplementary_label_path(label_root: Path, image_name: str) -> Path:
    """Resolve a supported year's image to its year-independent human label."""
    return resolve_label_path(label_root, image_name)


def supplementary_batch(common, names: list[str], indices: list[int], image_root: Path,
                        label_root: Path, device: torch.device, *, augment: bool = False,
                        rng=None):
    """Load a supervised-only supplementary batch.

    These tensors are intentionally separate from ``gan.batch_from``: they
    are used only by the segmentation classifier and never by FADA.
    """
    image_arrays = []
    label_arrays = []
    if augment and rng is None:
        raise ValueError("supplementary augmentation requires the epoch RNG")
    for index in indices:
        name = names[int(index)]
        image = common.read_raster(image_root / name, dtype=np.float32)
        label = common.read_raster(
            supplementary_label_path(label_root, name), dtype=np.int64
        )
        label = label[0] if label.ndim == 3 else label
        # Apply the identical spatial transform to imagery and Planet labels.
        # Only supplementary supervision is augmented; source/target/FADA
        # batches retain their established augmentation and sampling behavior.
        if augment:
            turns = int(rng.randint(0, 4))
            image = np.rot90(image, turns, axes=(-2, -1))
            label = np.rot90(label, turns, axes=(-2, -1))
            if bool(rng.randint(0, 2)):
                image = np.flip(image, axis=-1)
                label = np.flip(label, axis=-1)
            if bool(rng.randint(0, 2)):
                image = np.flip(image, axis=-2)
                label = np.flip(label, axis=-2)
        image_arrays.append(np.ascontiguousarray(image))
        label_arrays.append(np.ascontiguousarray(label))
    images = torch.from_numpy(np.stack(image_arrays).astype(np.float32, copy=False)).to(device)
    labels = torch.from_numpy(np.stack(label_arrays).astype(np.int64, copy=False)).to(device)
    return images, labels


def sparse_supplementary_schedule(count: int, steps: int, rng: np.random.RandomState,
                                  interval_steps: int = 4,
                                  max_occurrences_per_epoch: int | None = None) -> dict[int, list[int]]:
    """Spread bounded supplementary updates across an epoch.

    A small tile-specific pool is never expanded into every training step.
    A finite occurrence cap is optional.  ``None`` means unrestricted random
    sampling, including repeats within an epoch or a mini-batch.
    """
    if count < 1:
        return {}
    if interval_steps < 1 or (max_occurrences_per_epoch is not None and max_occurrences_per_epoch < 1):
        raise ValueError("supplementary interval and finite occurrence cap must be positive")
    batch_size = min(BATCH_SIZE, count)
    slots = range(0, steps, interval_steps)
    if max_occurrences_per_epoch is None:
        return {int(slot): rng.randint(0, count, size=batch_size).tolist() for slot in slots}
    pool = np.repeat(np.arange(count), max_occurrences_per_epoch)
    rng.shuffle(pool)
    schedule, offset = {}, 0
    for slot in slots:
        if offset >= len(pool):
            break
        indices = pool[offset:offset + batch_size].tolist()
        offset += len(indices)
        schedule[int(slot)] = indices
    return schedule


def supplementary_hard_negative_loss(logits, planet_labels, threshold=0.5):
    """Softly suppress high-confidence class-2 false positives only.

    Planet class-2 pixels are explicitly exempt.  The selected mask is
    detached from the model probabilities, so the model is trained by the
    penalty but cannot change the selection through the same backward pass.
    """
    probabilities = torch.softmax(logits, dim=1)[:, 2]
    eligible = ((planet_labels == 0) | (planet_labels == 1) | (planet_labels == 3))
    selected = eligible & (probabilities.detach() > float(threshold))
    if not bool(selected.any()):
        zero = logits.sum() * 0.0
        return zero, {"selected_pixels": 0, "eligible_pixels": int(eligible.sum().item()),
                      "pixels": int(planet_labels.numel()), "p2_mean": float(probabilities.detach().mean().item())}
    clipped = probabilities.clamp(min=1e-6, max=1.0 - 1e-6)
    loss = (-torch.log1p(-clipped[selected])).mean()
    return loss, {"selected_pixels": int(selected.sum().item()),
                  "eligible_pixels": int(eligible.sum().item()),
                  "pixels": int(planet_labels.numel()),
                  "p2_mean": float(probabilities.detach().mean().item())}


def supplementary_weighted_supervision_loss(logits, labels, negative_scale=0.5):
    """Binary class-2 supervision with weaker supplementary negatives only."""
    valid = (labels == 0) | (labels == 1) | (labels == 2) | (labels == 3)
    if not bool(valid.any()):
        return logits.sum() * 0.0, {"negative_pixels": 0, "positive_pixels": 0,
                                   "valid_pixels": 0}
    log_probability = torch.log_softmax(logits, dim=1)
    binary_log_probability = torch.stack(
        [torch.logsumexp(log_probability[:, (0, 1, 3)], dim=1), log_probability[:, 2]],
        dim=1,
    )
    target = (labels == 2).long()
    nll = -binary_log_probability.gather(1, target.unsqueeze(1)).squeeze(1)
    negative_weight, positive_weight = 0.6311714517, 2.4059025179
    weights = torch.where(
        target == 1,
        torch.as_tensor(positive_weight, device=logits.device),
        torch.as_tensor(negative_weight * float(negative_scale), device=logits.device),
    )
    loss = (nll[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-8)
    return loss, {
        "negative_pixels": int(((target == 0) & valid).sum().item()),
        "positive_pixels": int(((target == 1) & valid).sum().item()),
        "valid_pixels": int(valid.sum().item()),
    }


def supplementary_class2_suppression_loss(logits, planet_labels, images=None):
    """Continuously suppress class 2 on every known Planet non-class-2 pixel."""
    probabilities = torch.softmax(logits, dim=1)[:, 2]
    # Only the documented PlanetMap classes are meaningful.  Unknown labels,
    # ignore pixels, and empty AlphaEarth support must never become negatives.
    known = (planet_labels == 0) | (planet_labels == 1) | (planet_labels == 2) | (planet_labels == 3)
    support = torch.ones_like(known, dtype=torch.bool)
    if images is not None:
        support = torch.any(images != 0, dim=1)
    eligible = known & support & (planet_labels != 2)
    if not bool(eligible.any()):
        return logits.sum() * 0.0, {"selected_pixels": 0, "eligible_pixels": 0,
                                   "pixels": int(planet_labels.numel()),
                                   "p2_mean": float(probabilities.detach().mean().item())}
    clipped = probabilities.clamp(min=1e-6, max=1.0 - 1e-6)
    loss = (-torch.log1p(-clipped[eligible])).mean()
    return loss, {"selected_pixels": int(eligible.sum().item()),
                  "eligible_pixels": int(eligible.sum().item()),
                  "pixels": int(planet_labels.numel()),
                  "p2_mean": float(probabilities.detach().mean().item())}


def diagnostic_target_names(names, minimum=64):
    if not isinstance(names, EvaluationNames):
        return _diagnostic_target_names(names, minimum)
    groups = {}
    for row in names.locations:
        groups.setdefault(row["tile_id"], []).append(row["relative_tif"])
    rng = np.random.RandomState(917)
    pools = [list(rng.permutation(values)) for _, values in sorted(groups.items())]
    selected = []
    while len(selected) < min(minimum, len(names)):
        for pool in pools:
            if pool and len(selected) < minimum:
                selected.append(pool.pop())
    return _diagnostic_target_names(selected, minimum)


def evaluation_scope(names):
    return "fixed_tile_stratified_sample_valid_pixels" if isinstance(names, EvaluationNames) else "target_full_list_valid_pixels"


def full_target_prediction_metrics(gan, common, model, target_root, names, device,
                                   batch_size=16, return_argmax=False):
    return measure_evaluation(_full_target_prediction_metrics, gan, common, model,
                              target_root, names, device, batch_size, return_argmax)


def light_review(gan, common, model, source_valid_names, target_names,
                 target_root, full_target_names, device,
                 return_target_codes: bool = False, supplementary_job=None):
    """Review without the legacy domain-head diagnostic (which rereads labels)."""
    source_valid_names = list(source_valid_names)
    source_probe_names = source_valid_names[:64]
    target_names = list(target_names)[:64]
    source = common.old_diag.evaluate_source(model, source_valid_names)
    source_features = common.old_diag.collect_source_features(model, source_probe_names, max_items=64)
    target_features, probs = common.old_diag.collect_target_features(model, target_names, max_items=64)
    snd = common.old_diag.soft_neighborhood_density(target_features, temperature=0.05)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1).mean())
    review = {
        "source": source,
        "target_unsupervised": {
            "snd": float(snd), "entropy": entropy,
            "class_distribution": probs.mean(axis=0).tolist(),
        },
        "independent_frozen_feature_probe": common.old_diag.independent_probe(
            source_features, target_features
        ),
        "metric_provenance": {
            "source_validation": "full_source_valid_list",
            "source_validation_patch_count": len(source_valid_names),
            "domain_probe_source_patch_count": len(source_probe_names),
        },
    }
    # The old v4 review used the first 64 target patches and mean softmax,
    # which was not comparable with active-training hard argmax fractions.
    # Keep the legacy SND/probe diagnostics, but make the Agent-facing target
    # area metric one fixed full-tile eval-mode hard-argmax measurement.
    full_metric = full_target_prediction_metrics(
        gan, common, model, target_root, full_target_names, device,
        batch_size=16, return_argmax=return_target_codes,
    )
    target_codes = full_metric.pop("_argmax_codes", None)
    apply_full_target_prediction_metrics(review, full_metric)
    if isinstance(full_target_names, EvaluationNames):
        review["metric_provenance"].update({
            "target_distribution": "fixed_tile_sample_eval_mode_weighted_hard_argmax_population_estimate",
            "target_mean_softmax": "fixed_tile_sample_eval_mode_weighted_mean",
            "target_entropy": "fixed_tile_sample_eval_mode_weighted_mean",
            "evaluation_sample_manifest": full_target_names.policy,
            "whole_target_census": False,
        })
    if return_target_codes:
        review["_target_argmax_codes"] = target_codes
    if supplementary_job is not None:
        evidence=human.evaluation(gan,common,model,supplementary_job,device,supplementary_batch)
        if evidence is not None:review['human_supervision_evaluation']=evidence
    return review


def _gradient_norm(values) -> float:
    tensors = [value.detach() for value in values if value is not None]
    if not tensors:
        return 0.0
    return float(torch.sqrt(sum((value ** 2).sum() for value in tensors)).item())


def _fixed_initialization_batches(count: int, steps: int,
                                  rng: np.random.RandomState,
                                  allow_replacement: bool = False) -> list[list[int]]:
    """Select complete fixed mini-batches without invoking full-epoch coverage."""
    required = int(steps) * BATCH_SIZE
    if count < required and not allow_replacement:
        raise ValueError(
            f"initialization probe requires {required} samples, received {count}"
        )
    if count < 1:
        raise ValueError("initialization probe requires at least one sample")
    indices = (rng.choice(count, size=required, replace=True).tolist()
               if count < required else rng.permutation(count)[:required].tolist())
    return [
        indices[offset:offset + BATCH_SIZE]
        for offset in range(0, required, BATCH_SIZE)
    ]


def _fixed_branch_batches(count: int, steps: int,
                          rng: np.random.RandomState) -> list[list[int]]:
    """Build a deterministic paired branch sequence without a size gate.

    A short diagnostic can legitimately request more examples than a domain
    contains (for example 64 x 16 versus 880 source patches).  Use every
    shuffled item once, then draw the remainder with replacement.  The caller
    resets the same seed for control and treatment, so both branches receive
    exactly the same ordered sequence; formal epoch sampling and the strict
    initialization probe remain unchanged.
    """
    if count < 1:
        raise ValueError("branch experiment requires at least one sample")
    required = int(steps) * BATCH_SIZE
    first = rng.permutation(count).tolist()
    if required > count:
        repeated = rng.randint(0, count, size=required - count).tolist()
        first.extend(repeated)
    indices = first[:required]
    return [
        indices[offset:offset + BATCH_SIZE]
        for offset in range(0, required, BATCH_SIZE)
    ]


def initialization_probe(gan, common, model, local_d, global_d, dataset,
                         source_names, target_names, supplementary_names,
                         supplementary_image_root, supplementary_label_root,
                         config, device) -> dict:
    """Measure a discardable mini-pilot before the Agent chooses epoch-1 controls.

    The formal model/discriminators and all RNG streams are restored exactly.
    No optimizer connected to a formal network is created or stepped.
    """
    model_digest = model_state_digest(model)
    local_digest = model_state_digest(local_d)
    global_digest = model_state_digest(global_d)
    modes = (model.training, local_d.training, global_d.training)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    probe_rng = np.random.RandomState(1017)
    probe_steps = 2
    source_batches = _fixed_initialization_batches(
        len(source_names), probe_steps, probe_rng
    )
    target_batches = _fixed_initialization_batches(
        len(target_names), probe_steps, probe_rng
    )
    supplementary_batches = (
        _fixed_initialization_batches(
            len(supplementary_names), probe_steps, probe_rng, allow_replacement=True
        )
        if supplementary_names and (not config.get("conditional_negative_policy") or negative.weight(config, 0) > 0) else [None] * probe_steps
    )
    cache = []
    try:
        model.eval(); local_d.eval(); global_d.eval()
        with torch.no_grad():
            for source_indices, target_indices, supplementary_indices in zip(
                    source_batches, target_batches, supplementary_batches):
                source_batch, target_batch = gan.batch_from(
                    dataset, source_indices, target_indices, device
                )
                source_aug, source_label, source_raw, _ = source_batch
                _, _, _, target_raw = target_batch
                source_feature = gan.stage4_map(model, source_raw)
                target_feature = gan.stage4_map(model, target_raw)
                source_logits, _ = gan.forward_fuse(model, source_raw.float())
                target_logits, _ = gan.forward_fuse(model, target_raw.float())
                item = {
                    "source_aug": source_aug.detach().clone(),
                    "source_label": source_label.long().detach().clone(),
                    "source_raw": source_raw.detach().clone(),
                    "target_raw": target_raw.detach().clone(),
                    "source_feature": source_feature.detach().clone(),
                    "target_feature": target_feature.detach().clone(),
                    "source_lowres": resize_segmentation_logits(
                        source_logits, source_feature
                    ).detach().clone(),
                    "target_lowres": resize_segmentation_logits(
                        target_logits, target_feature
                    ).detach().clone(),
                }
                if supplementary_indices is not None:
                    images, labels = supplementary_batch(
                        common, supplementary_names, supplementary_indices,
                        supplementary_image_root, supplementary_label_root, device,
                    )
                    item["supplementary_images"] = images.detach().clone()
                    item["supplementary_labels"] = labels.detach().clone()
                cache.append(item)

        temperature_probe = {}
        for temperature in (1.2, 1.5, 1.8, 2.2, 2.6, 3.0):
            values = []
            for item in cache:
                probabilities = torch.softmax(item["target_lowres"] / temperature, dim=1)
                maximum = probabilities.max(dim=1).values
                values.append({
                    "normalized_entropy": float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1) / np.log(4)).mean()),
                    "mean_max_probability": float(maximum.mean()),
                    "class_distribution": probabilities.mean(dim=(0, 2, 3)).tolist(),
                })
            temperature_probe[str(temperature)] = {
                "target_normalized_entropy": float(np.mean([x["normalized_entropy"] for x in values])),
                "target_mean_max_probability": float(np.mean([x["mean_max_probability"] for x in values])),
                "target_class_distribution": np.mean([x["class_distribution"] for x in values], axis=0).tolist(),
            }

        cap_probe = {}
        anchor_probabilities = [
            torch.softmax(item["target_lowres"] / float(config["temperature"]), dim=1)
            for item in cache
        ]
        for cap in (0.80, 0.85, 0.90, 0.95):
            cap_probe[str(cap)] = {
                "target_values_clipped_fraction": float(np.mean([
                    float((probabilities > cap).float().mean())
                    for probabilities in anchor_probabilities
                ])),
            }

        def knowledge(item, domain):
            return class_knowledge_v4(
                item[f"{domain}_lowres"], mode=config["local_mode"],
                temperature=config["temperature"], confidence_cap=config["confidence_cap"],
                hard_threshold=config["hard_threshold"] or 0.9,
            )[0]

        d_lr_probe = {}
        for learning_rate in (5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5):
            probe_local = copy.deepcopy(local_d).to(device)
            probe_global = copy.deepcopy(global_d).to(device)
            local_optimizer = torch.optim.AdamW(
                probe_local.parameters(), lr=learning_rate,
                weight_decay=float(config["D_weight_decay"]),
            )
            global_optimizer = torch.optim.AdamW(
                probe_global.parameters(), lr=learning_rate,
                weight_decay=float(config["D_weight_decay"]),
            )
            trajectory = []
            for step in range(8):
                item = cache[step % len(cache)]
                local_optimizer.zero_grad(set_to_none=True)
                global_optimizer.zero_grad(set_to_none=True)
                local_value, local_stats = local_loss(
                    probe_local, item["source_feature"], item["target_feature"],
                    knowledge(item, "source"), knowledge(item, "target"),
                )
                global_value, global_stats = global_loss(
                    probe_global, item["source_feature"], item["target_feature"]
                )
                (local_value + global_value).backward()
                local_optimizer.step(); global_optimizer.step()
                trajectory.append({
                    "step": step + 1,
                    "loss": float((local_value + global_value).detach()),
                    "local_accuracy": float(local_stats["accuracy"]),
                    "global_accuracy": float(global_stats["accuracy"]),
                })
            d_lr_probe[str(learning_rate)] = {
                "initial": trajectory[0], "after_2_steps": trajectory[1],
                "after_4_steps": trajectory[3], "after_8_steps": trajectory[7],
                "nonfinite": not all(np.isfinite(point["loss"]) for point in trajectory),
            }
            del probe_local, probe_global, local_optimizer, global_optimizer

        warm_local = copy.deepcopy(local_d).to(device)
        warm_global = copy.deepcopy(global_d).to(device)
        warm_local_optimizer = torch.optim.AdamW(
            warm_local.parameters(), lr=float(config["D_lr"]),
            weight_decay=float(config["D_weight_decay"]),
        )
        warm_global_optimizer = torch.optim.AdamW(
            warm_global.parameters(), lr=float(config["D_lr"]),
            weight_decay=float(config["D_weight_decay"]),
        )
        for step in range(8):
            item = cache[step % len(cache)]
            warm_local_optimizer.zero_grad(set_to_none=True)
            warm_global_optimizer.zero_grad(set_to_none=True)
            local_value, _ = local_loss(
                warm_local, item["source_feature"], item["target_feature"],
                knowledge(item, "source"), knowledge(item, "target"),
            )
            global_value, _ = global_loss(
                warm_global, item["source_feature"], item["target_feature"]
            )
            (local_value + global_value).backward()
            warm_local_optimizer.step(); warm_global_optimizer.step()

        cls_gradients, local_gradients, global_gradients, adv_gradients = [], [], [], []
        source_losses, supplementary_losses = [], []
        negative_component_probes = []
        direction_probes = []
        encoder_parameters = feature_learning_parameters(model)
        nonzero_gradient_elements = []
        parameter_norm = _gradient_norm(encoder_parameters)
        for item in cache:
            source_logits, _ = gan.forward_fuse(model, item["source_aug"].float())
            source_cls = four_class_source_loss(source_logits, item["source_label"])
            supplementary_cls = source_cls.detach() * 0.0
            if "supplementary_images" in item:
                mode = config["supplementary_training"]["mode"]
                supplementary_logits, _ = human.forward(gan,model,item['supplementary_images'],mode)
                if mode == human.MODE:
                    raw_supplementary, _ = human.supervised_loss(supplementary_logits,item['supplementary_labels'],item['supplementary_images'],config)
                elif mode == "weighted_negative_supervision":
                    raw_supplementary, _ = supplementary_weighted_supervision_loss(
                        supplementary_logits, item["supplementary_labels"],
                        negative_scale=float(config["supplementary_training"]["negative_weight_scale"]),
                    )
                elif mode == "hard_negative_suppression":
                    raw_supplementary, _ = supplementary_hard_negative_loss(
                        supplementary_logits, item["supplementary_labels"],
                        threshold=float(config["supplementary_training"]["hard_threshold"]),
                    )
                elif mode == "class2_suppression":
                    raw_supplementary, _ = supplementary_class2_suppression_loss(
                        supplementary_logits, item["supplementary_labels"], item["supplementary_images"]
                    )
                else:
                    raw_supplementary = balanced_binary_source_loss(
                        supplementary_logits, item["supplementary_labels"]
                    )
                supplementary_cls = (
                    float(config["supplementary_training"]["loss_weight"])
                    * raw_supplementary
                )
            segmentation_loss = source_cls + supplementary_cls
            target_feature = gan.stage4_map(model, item["target_raw"])
            target_logits, _ = gan.forward_fuse(model, item["target_raw"].float())
            target_lowres = resize_segmentation_logits(target_logits, target_feature)
            target_knowledge, _ = class_knowledge_v4(
                target_lowres, mode=config["local_mode"],
                temperature=config["temperature"], confidence_cap=config["confidence_cap"],
                hard_threshold=config["hard_threshold"] or 0.9,
            )
            local_adv = local_target_adversarial_loss(warm_local, target_feature, target_knowledge.detach() if negative.enabled(config) else target_knowledge)
            global_adv = global_target_adversarial_loss(warm_global, target_feature)
            if negative.enabled(config):
                negative_component_probes.append(negative.component_gradients(model, source_cls, supplementary_cls, local_adv + global_adv, config["lambda_adv"]))
            cls_values = torch.autograd.grad(
                segmentation_loss, encoder_parameters, retain_graph=True, allow_unused=True
            )
            local_values = torch.autograd.grad(
                local_adv, encoder_parameters, retain_graph=True, allow_unused=True
            )
            global_values = torch.autograd.grad(
                global_adv, encoder_parameters, retain_graph=True, allow_unused=True
            )
            if gradient_control.enabled(config):
                source_values = (torch.autograd.grad(source_cls, encoder_parameters, retain_graph=True, allow_unused=True)
                                 if 'supplementary_images' in item else cls_values)
            adv_values = torch.autograd.grad(
                local_adv + global_adv, encoder_parameters, allow_unused=True
            )
            if gradient_control.enabled(config):
                direction, _ = gradient_control.prepare(
                    encoder_parameters, source_values, adv_values, local_values, global_values,
                    float(config['lambda_adv']), 0.0)
                direction_probes.append(direction)
            cls_gradients.append(_gradient_norm(cls_values))
            local_gradients.append(_gradient_norm(local_values))
            global_gradients.append(_gradient_norm(global_values))
            adv_gradients.append(_gradient_norm(adv_values))
            nonzero_gradient_elements.append(sum(
                int(torch.count_nonzero(value).item()) for value in cls_values if value is not None
            ))
            source_losses.append(float(source_cls.detach()))
            supplementary_losses.append(float(supplementary_cls.detach()))

        cls_grad = float(np.median(cls_gradients))
        adv_grad = float(np.median(adv_gradients))
        lambda_candidates = {
            f"effective_ratio_{ratio:.4f}": float(np.clip(
                ratio * cls_grad / max(adv_grad, 1e-12),
                PARAM_BOUNDS["lambda_adv"][0], PARAM_BOUNDS["lambda_adv"][1],
            ))
            for ratio in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)
        }
        active_elements = max(1, int(np.median(nonzero_gradient_elements)))
        main_lr_probe = {
            str(learning_rate): {
                "estimated_first_adamw_update_to_parameter_norm": float(
                    learning_rate * np.sqrt(active_elements) / max(parameter_norm, 1e-12)
                ),
                "method": "gradient-supported first-step AdamW scale estimate; no formal update",
            }
            for learning_rate in (2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5)
        }
        report = {
            "schema_version": 1,
            "kind": "discardable_pre_epoch1_diagnostic",
            "probe_seed": 1017,
            "probe_batches": probe_steps,
            "formal_training_steps": 0,
            "loss_probe": {
                "source_segmentation_loss_mean": float(np.mean(source_losses)),
                "supplementary_segmentation_loss_mean": float(np.mean(supplementary_losses)),
                "combined_segmentation_loss_mean": float(np.mean(np.asarray(source_losses) + np.asarray(supplementary_losses))),
            },
            "gradient_probe": {
                "combined_segmentation_grad_median": cls_grad,
                "local_adversarial_grad_median": float(np.median(local_gradients)),
                "global_adversarial_grad_median": float(np.median(global_gradients)),
                "source_adversarial_direction_probes": direction_probes,
                "combined_adversarial_grad_median": adv_grad,
                "raw_adv_to_segmentation_ratio": float(adv_grad / max(cls_grad, 1e-12)),
                "reference_lambda_effective_ratio": float(
                    config["lambda_adv"] * adv_grad / max(cls_grad, 1e-12)
                ),
                "lambda_candidates": lambda_candidates,
            },
            "temperature_probe": temperature_probe,
            "confidence_cap_probe": cap_probe,
            "main_lr_probe": main_lr_probe,
            "D_lr_and_D_steps_probe": d_lr_probe,
            "weight_decay_note": "No reliable one-batch estimator; Agent may retain or adjust within bounds using history and measured scales.",
            "reference_only_config": {key: config[key] for key in PARAM_BOUNDS},
            "formal_state_unchanged": True,
        }
        if negative.enabled(config):
            report["negative_controls"] = {"bounds": negative.bounds(config), "pool_count": len(supplementary_names),
                "component_gradient_probes": negative_component_probes, "small_pool_is_not_a_small_loss_coefficient": True,
                "probe_coefficient": negative.weight(config, 0), "scope": "two eval-mode discardable batches, not target accuracy or a tested policy winner"}
        del warm_local, warm_global, warm_local_optimizer, warm_global_optimizer
    finally:
        model.train(modes[0]); local_d.train(modes[1]); global_d.train(modes[2])
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    if (
        model_state_digest(model) != model_digest
        or model_state_digest(local_d) != local_digest
        or model_state_digest(global_d) != global_digest
    ):
        raise RuntimeError("initialization probe contaminated formal network state")
    return report


def train_epoch(gan, common, model, local_d, global_d, model_opt, local_opt, global_opt,
                dataset, source_count, target_count, supplementary_names,
                supplementary_image_root, supplementary_label_root, config, device, rng,
                steps_override=None, automatic_negative_runtime=None, experimental_recorder=None,
                annotated_samples=None):
    from module_bn_v2 import domain as bn_domain, apply as apply_bn
    apply_bn(model, config)
    model.train(); local_d.train(); global_d.train()
    supplementary_count = len(supplementary_names)
    steps = steps_override or max(
        (source_count + BATCH_SIZE - 1) // BATCH_SIZE,
        (target_count + BATCH_SIZE - 1) // BATCH_SIZE,
    )
    batch_builder = _fixed_branch_batches if steps_override else epoch_batches
    src_batches = batch_builder(source_count, BATCH_SIZE, steps, rng) if not steps_override else batch_builder(source_count, steps, rng)
    tgt_batches = batch_builder(target_count, BATCH_SIZE, steps, rng) if not steps_override else batch_builder(target_count, steps, rng)
    totals = {"cls": [], "seg_cls": [], "supplementary_cls": [], "adv": [], "D_loss": [], "local_D_acc": [], "global_D_acc": [], "local_coverage": [], "cls_grad": [], "adv_grad": [], "local_adv_grad": [], "global_adv_grad": [], "supp_selected_pixels": 0, "supp_eligible_pixels": 0, "supp_pixels": 0, "supp_p2_mean": []}
    supplementary_policy = config.get("supplementary_training", {})
    supplementary_mode = supplementary_policy.get("mode", "supervised_segmentation")
    supplementary_weight = float(supplementary_policy.get("loss_weight", 1.0))
    supplementary_threshold = float(supplementary_policy.get("hard_threshold", 0.5))
    supplementary_augmentation = bool(supplementary_policy.get("spatial_augmentation", {}).get("enabled", False))
    adaptive = negative.enabled(config)
    full_epoch_steps = max((source_count + BATCH_SIZE - 1) // BATCH_SIZE, (target_count + BATCH_SIZE - 1) // BATCH_SIZE)
    seed = int(config.get("_negative_seed", int(rng.get_state()[1][0]))) % (2**32 - 1000000)
    epoch_start = float(config.get("_negative_epoch_start", 0))
    if adaptive:
        supplementary_schedule = {}
        for offset in range(0, steps, full_epoch_steps):
            window = min(full_epoch_steps, steps - offset)
            part = negative.exposure_schedule(supplementary_count, window, np.random.RandomState(seed + offset),
                                             int(config[negative.CAP]), int(config[negative.INTERVAL]))
            supplementary_schedule.update({offset+i:values for i,values in part.items()})
    else:
        supplementary_schedule = sparse_supplementary_schedule(
            supplementary_count, steps, rng,
            interval_steps=int(supplementary_policy.get("update_interval_steps", 1)),
            max_occurrences_per_epoch=supplementary_policy.get("max_occurrences_per_epoch"),
        )
    coefficient_values = []; component_probes = []
    from module_controls_v2 import enabled as detailed_modules
    from module_observer_v2 import gradients as module_gradients, human_terms
    from transfer_regularization_v3 import terms as regularization_terms
    if experimental_recorder is not None and hasattr(experimental_recorder,'bind_model'):
        experimental_recorder.bind_model(model)
    module_probes = []
    direction_records = []
    projection_amount = gradient_control.strength(config)
    direction_steps = gradient_control.probe_steps(steps) if gradient_control.enabled(config) else set()
    negative_probe_steps = negative.component_probe_steps(supplementary_schedule, steps, config, epoch_start, full_epoch_steps) if adaptive else set()
    scheduled_indices = []
    if experimental_recorder is not None:
        from supplementary_exposure_v1 import Exposure
        supplementary_exposure = Exposure(supplementary_schedule)
        from training_exposure_v1 import Exposure as DomainExposure
        domain_exposure = DomainExposure(source_count, target_count)
    else:
        supplementary_exposure = None
        domain_exposure = None
    if automatic_negative_runtime is not None:
        automatic_negative_runtime.begin_epoch(steps, config, seed)
    for step, (sidx, tidx) in enumerate(zip(src_batches, tgt_batches)):
        pidx = supplementary_schedule.get(step)
        if adaptive:
            supplementary_weight = negative.weight(config, epoch_start + (step+1)/full_epoch_steps)
            coefficient_values.append(supplementary_weight)
            if supplementary_weight == 0: pidx = None
        if pidx is not None:
            scheduled_indices.extend(pidx)
            if supplementary_exposure is not None:
                supplementary_exposure.start(pidx)
        active = config.get("adversarial_state", "active") == "active"
        d_steps = int(config["D_steps_per_G_step"]) if active else 0
        if domain_exposure is not None:
            domain_exposure.start(sidx, tidx, adversarial_active=active,
                                  generator_adversarial_weight=float(config['lambda_adv']),
                                  discriminator_active=d_steps > 0)
        source_batch, target_batch = gan.batch_from(dataset, sidx, tidx, device)
        source_aug, source_label, source_raw, _ = source_batch
        _, _, _, target_raw = target_batch
        if domain_exposure is not None:
            domain_exposure.loaded(int(source_aug.shape[0]), int(target_raw.shape[0]))
        source_label = source_label.long()
        for d_index in range(d_steps):
            # no_grad does not freeze BatchNorm running statistics.  D updates
            # must not mutate the segmentation model, so produce frozen-domain
            # features in eval mode and restore the training mode afterwards.
            model_was_training = model.training
            model.eval()
            with torch.no_grad():
                with bn_domain('source'):
                    sf = gan.stage4_map(model, source_raw)
                    sl, _ = gan.forward_fuse(model, source_raw.float())
                with bn_domain('target'):
                    tf = gan.stage4_map(model, target_raw)
                    tl, _ = gan.forward_fuse(model, target_raw.float())
                sl = resize_segmentation_logits(sl, sf); tl = resize_segmentation_logits(tl, tf)
                if experimental_recorder is not None and d_index == 0 and step in {0, steps // 2, steps - 1}:
                    experimental_recorder.observe_logits(sl, tl, config, step)
                sk, sm = class_knowledge_v4(sl, mode=config["local_mode"], temperature=config["temperature"], confidence_cap=config["confidence_cap"], hard_threshold=config["hard_threshold"] or 0.9)
                tk, tm = class_knowledge_v4(tl, mode=config["local_mode"], temperature=config["temperature"], confidence_cap=config["confidence_cap"], hard_threshold=config["hard_threshold"] or 0.9)
            model.train(model_was_training)
            local_opt.zero_grad(set_to_none=True); global_opt.zero_grad(set_to_none=True)
            ld, ls = local_loss(local_d, sf.detach(), tf.detach(), sk, tk)
            gd, gs = global_loss(global_d, sf.detach(), tf.detach())
            if not bool(torch.isfinite(ld + gd).all().item()):
                raise RuntimeError({"nonfinite_training_loss": "discriminator", "step": step})
            (ld + gd).backward()
            if experimental_recorder is None:
                local_opt.step(); global_opt.step()
            else:
                from experimental_gate_v1 import assert_finite_gradients
                assert_finite_gradients(local_opt, global_opt)
                sample = step in {0, steps // 2, steps - 1}
                experimental_recorder.step('D_local', local_opt, sample)
                experimental_recorder.step('D_global', global_opt, sample)
            totals["D_loss"].append(float((ld + gd).item())); totals["local_D_acc"].append(ls["accuracy"]); totals["global_D_acc"].append(gs["accuracy"]); totals["local_coverage"].append(float((sm.mean() + tm.mean()).item() / 2.0))
        model_opt.zero_grad(set_to_none=True)
        pending_projection = None
        module_losses = {}
        module_probe = detailed_modules(config) and step in ({0, steps//2, steps-1} | negative_probe_steps)
        with bn_domain('source'):
            slogits, _ = gan.forward_fuse(model, source_aug.float())
        cls = four_class_source_loss(slogits, source_label)
        if pidx is not None:
            with negative.rng_scope(rng, seed + 100000 + step, device) if adaptive else nullcontext():
                supplementary_images, supplementary_labels = supplementary_batch(
                    common, supplementary_names, pidx, supplementary_image_root,
                    supplementary_label_root, device, augment=supplementary_augmentation, rng=rng,
                )
                if supplementary_exposure is not None:
                    supplementary_exposure.loaded_batch(supplementary_images, supplementary_labels)
                supplementary_logits, _ = human.forward(gan,model,supplementary_images,supplementary_mode)
                if supplementary_exposure is not None:
                    supplementary_exposure.forward_done()
                if supplementary_mode == human.MODE:
                    raw_supplementary_cls, supplementary_stats = human.supervised_loss(supplementary_logits,supplementary_labels,supplementary_images,config)
                    supplementary_cls = supplementary_weight * raw_supplementary_cls
                    if module_probe:
                        module_losses.update(human_terms(supplementary_logits, supplementary_labels, supplementary_images, supplementary_weight, config['supplementary_positive_scale']))
                    totals['supp_selected_pixels'] += supplementary_stats['selected_pixels']
                    totals['supp_eligible_pixels'] += supplementary_stats['eligible_pixels']
                    totals['supp_pixels'] += supplementary_stats['pixels']
                    totals['supp_p2_mean'].append(supplementary_stats['p2_mean'])
                    totals.setdefault('polygon_positive_pixels',0)
                    totals['polygon_positive_pixels'] += supplementary_stats['positive_pixels']
                elif supplementary_mode == "hard_negative_suppression":
                    raw_supplementary_cls, supplementary_stats = supplementary_hard_negative_loss(
                        supplementary_logits, supplementary_labels, threshold=supplementary_threshold,
                    )
                    supplementary_cls = supplementary_weight * raw_supplementary_cls
                    totals["supp_selected_pixels"] += supplementary_stats["selected_pixels"]
                    totals["supp_eligible_pixels"] += supplementary_stats["eligible_pixels"]
                    totals["supp_pixels"] += supplementary_stats["pixels"]
                    totals["supp_p2_mean"].append(supplementary_stats["p2_mean"])
                elif supplementary_mode == "weighted_negative_supervision":
                    raw_supplementary_cls, supplementary_stats = supplementary_weighted_supervision_loss(
                        supplementary_logits, supplementary_labels,
                        negative_scale=float(supplementary_policy.get("negative_weight_scale", 0.5)),
                    )
                    supplementary_cls = supplementary_weight * raw_supplementary_cls
                    totals["supp_selected_pixels"] += supplementary_stats["negative_pixels"]
                    totals["supp_eligible_pixels"] += supplementary_stats["valid_pixels"]
                    totals["supp_pixels"] += int(supplementary_labels.numel())
                elif supplementary_mode == "class2_suppression":
                    raw_supplementary_cls, supplementary_stats = supplementary_class2_suppression_loss(
                        supplementary_logits, supplementary_labels, supplementary_images,
                    )
                    supplementary_cls = supplementary_weight * raw_supplementary_cls
                    totals["supp_selected_pixels"] += supplementary_stats["selected_pixels"]
                    totals["supp_eligible_pixels"] += supplementary_stats["eligible_pixels"]
                    totals["supp_pixels"] += supplementary_stats["pixels"]
                    totals["supp_p2_mean"].append(supplementary_stats["p2_mean"])
                else:
                    supplementary_cls = balanced_binary_source_loss(
                        supplementary_logits, supplementary_labels,
                    )
                if supplementary_exposure is not None:
                    supplementary_exposure.observed_loss(
                        supplementary_images, supplementary_labels,
                        human_mode=supplementary_mode == human.MODE,
                        weight=float(supplementary_weight),
                        positive_scale=float(config['supplementary_positive_scale']) if supplementary_mode == human.MODE else None,
                        expected_stats=supplementary_stats if supplementary_mode == human.MODE else None)
        else:
            supplementary_cls = cls.detach() * 0.0
        automatic_cls = (automatic_negative_runtime.step(step, gan, common, model, device, seed)
                         if automatic_negative_runtime is not None else None)
        automatic_active = automatic_cls is not None
        penalties = regularization_terms(model,config)
        regularization_loss = sum(penalties.values()) if penalties else cls.detach()*0.
        totals.setdefault('regularization_loss',[]).append(float(regularization_loss.detach()))
        annotation_exposure = None
        if annotated_samples is not None:
            annotation_loss, annotation_exposure = annotated_samples.forward(step, gan, model, cls)
            if annotation_loss is not None:
                supplementary_cls = supplementary_cls + annotation_loss
        seg_cls = cls + supplementary_cls + (automatic_cls if automatic_active else 0.) + regularization_loss
        if module_probe and penalties:module_losses['regularization']=regularization_loss
        if active:
            tf = gan.stage4_map(model, target_raw)
            tl, _ = gan.forward_fuse(model, target_raw.float())
            tl = resize_segmentation_logits(tl, tf)
            tk, _ = class_knowledge_v4(tl, mode=config["local_mode"], temperature=config["temperature"], confidence_cap=config["confidence_cap"], hard_threshold=config["hard_threshold"] or 0.9)
            # Class knowledge is a discriminator conditioning signal.  Do not
            # let the target adversarial objective re-label pixels merely to
            # make the local discriminator easier to fool.
            local_adv = local_target_adversarial_loss(local_d, tf, tk.detach())
            global_adv = global_target_adversarial_loss(global_d, tf)
            adv = local_adv + global_adv
            if not bool(torch.isfinite(seg_cls + float(config["lambda_adv"]) * adv).all().item()):
                raise RuntimeError({"nonfinite_training_loss": "segmentation_or_adversarial", "step": step})
            measure_gradient = (not gradient_control.enabled(config) or projection_amount > 0 or step in direction_steps)
            if measure_gradient:
                params = feature_learning_parameters(model)
                cg = torch.autograd.grad(seg_cls, params, retain_graph=True, allow_unused=True)
                lg = torch.autograd.grad(local_adv, params, retain_graph=True, allow_unused=True)
                gg = torch.autograd.grad(global_adv, params, retain_graph=True, allow_unused=True)
                ag = torch.autograd.grad(adv, params, retain_graph=True, allow_unused=True)
                if gradient_control.enabled(config) and (projection_amount > 0 or step in direction_steps):
                    # cg includes supplementary supervision whenever it was used.
                    # Projection and source-conflict diagnosis must protect pure source supervision.
                    sg = (torch.autograd.grad(cls, params, retain_graph=True, allow_unused=True)
                          if pidx is not None or automatic_active or penalties or annotation_exposure is not None else cg)
                    direction, pending_projection = gradient_control.prepare(
                        params, sg, ag, lg, gg, float(config['lambda_adv']), projection_amount)
                    direction['step'] = step
                    direction['supplementary_forward_active'] = pidx is not None
                    direction_records.append(direction)
                grad_norm = lambda values: float(torch.sqrt(sum((g.detach() ** 2).sum() for g in values if g is not None)).item()) if any(g is not None for g in values) else 0.0
                totals["cls_grad"].append(grad_norm(cg)); totals["local_adv_grad"].append(grad_norm(lg)); totals["global_adv_grad"].append(grad_norm(gg)); totals["adv_grad"].append(grad_norm(ag))
            if adaptive and step in negative_probe_steps:
                component_probes.append({"step":step,"active_negative_forward":pidx is not None,"adversarial_measured":True,
                    "measurements":negative.component_gradients(model, cls, supplementary_cls, adv, config["lambda_adv"])})
            if automatic_active and step in {min(automatic_negative_runtime.schedule), max(automatic_negative_runtime.schedule)}:
                automatic_negative_runtime.record_gradients(model, cls, automatic_cls, adv, config['lambda_adv'], step)
            if module_probe:
                module_losses.update(source=cls, local_adversarial=float(config['lambda_adv'])*local_adv, global_adversarial=float(config['lambda_adv'])*global_adv)
                module_probes.append(module_gradients(model, module_losses, step))
            (seg_cls + float(config["lambda_adv"]) * adv).backward()
        else:
            adv = cls.detach() * 0.0
            if not bool(torch.isfinite(seg_cls).all().item()):
                raise RuntimeError({"nonfinite_training_loss": "segmentation", "step": step})
            if adaptive and step in negative_probe_steps:
                component_probes.append({"step":step,"active_negative_forward":pidx is not None,"adversarial_measured":False,
                    "measurements":negative.component_gradients(model, cls, supplementary_cls, adv, 0)})
            if automatic_active and step in {min(automatic_negative_runtime.schedule), max(automatic_negative_runtime.schedule)}:
                automatic_negative_runtime.record_gradients(model, cls, automatic_cls, cls.detach()*0., 0., step)
            if module_probe:
                module_losses['source'] = cls
                module_probes.append(module_gradients(model, module_losses, step))
            seg_cls.backward()
        if pending_projection is not None:
            gradient_control.apply(pending_projection)
        if experimental_recorder is None:
            model_opt.step()
        else:
            from experimental_gate_v1 import assert_finite_gradients
            assert_finite_gradients(model_opt)
            experimental_recorder.step('G', model_opt, step in {0, steps // 2, steps - 1})
        if supplementary_exposure is not None and pidx is not None:
            supplementary_exposure.optimizer_succeeded()
        if domain_exposure is not None:
            domain_exposure.optimizer_succeeded()
        if annotated_samples is not None:
            annotated_samples.commit(annotation_exposure)
        totals["cls"].append(float(cls.item())); totals["seg_cls"].append(float(seg_cls.item())); totals["supplementary_cls"].append(float(supplementary_cls.item())); totals["adv"].append(float(adv.item()))
    result = {key: float(np.mean(values)) if isinstance(values, list) and values else 0.0 for key, values in totals.items() if isinstance(values, list)}
    result.update({key: value for key, value in totals.items() if not isinstance(value, list)})
    if supplementary_mode in {"hard_negative_suppression", "class2_suppression", human.MODE}:
        result["supp_selected_fraction"] = float(result["supp_selected_pixels"] / max(1, result["supp_eligible_pixels"]))
        result["supp_p2_mean"] = float(np.mean(totals["supp_p2_mean"])) if totals["supp_p2_mean"] else 0.0
    result["effective_gradient_ratio"] = float(config["lambda_adv"] * result["adv_grad"] / max(result["cls_grad"], 1e-12))
    result["effective_local_gradient_ratio"] = float(config["lambda_adv"] * result["local_adv_grad"] / max(result["cls_grad"], 1e-12))
    result["effective_global_gradient_ratio"] = float(config["lambda_adv"] * result["global_adv_grad"] / max(result["cls_grad"], 1e-12))
    result["supp_unique_samples"] = len(set(scheduled_indices))
    result["supp_scheduled_samples"] = len(scheduled_indices)
    result["supp_pool_coverage_fraction"] = float(len(set(scheduled_indices)) / max(1, supplementary_count))
    result['supp_legacy_counts_meaning']='scheduled indices after weight gate, not actual loaded samples, effective signs, gradient or optimizer exposure'
    if supplementary_exposure is not None:
        result['supplementary_actual_exposure'] = supplementary_exposure.snapshot()
    if domain_exposure is not None:
        result['source_target_actual_exposure'] = domain_exposure.snapshot()
    if annotated_samples is not None:
        result['annotated_training_updates'] = annotated_samples.metrics()
    if adaptive:
        from collections import Counter
        result["negative_controls"] = {"version":negative.VERSION,"base_weight":config[negative.WEIGHT],
            "effective_weight_min":min(coefficient_values),"effective_weight_max":max(coefficient_values),
            "occurrence_cap_per_epoch":config[negative.CAP],"interval_steps":config[negative.INTERVAL],
            "horizon_epochs":config[negative.HORIZON],"curve":config["schedule_state"][negative.WEIGHT]["curve"],
            "actual_max_sample_occurrences_in_call":max(Counter(scheduled_indices).values(),default=0),
            "negative_updates":sum(1 for i in supplementary_schedule if negative.weight(config,epoch_start+(i+1)/full_epoch_steps)>0),
            "count_semantics":"historical scheduled indices and planned nonzero-coefficient steps; not proven actual G updates",
            "component_gradient_probes":component_probes,"random_stream_isolated":True,
            "diagnostic_scope":("first/last model and first/last active-negative update; " if config.get('negative_response_policy',{}).get('version')=='late_negative_response_v3' else "first/last update; ")+"fusion/refiner and decoder only; norms/cosines are not optimizer ratios or target quality"}
    result["steps"] = steps
    if detailed_modules(config): result['module_gradient_probes'] = module_probes
    if experimental_recorder is not None:
        result['experimental_actual_updates'] = experimental_recorder.finish()
        if result['experimental_actual_updates']['finite'] is False:
            raise RuntimeError('nonfinite sampled optimizer update: safe pause')
    if automatic_negative_runtime is not None:
        result['automatic_negative'] = automatic_negative_runtime.summary()
    if gradient_control.enabled(config):
        result['gradient_direction'] = gradient_control.summarize(direction_records, steps, projection_amount)
        result['gradient_norm_sampling'] = dict(measured_updates=len(totals['cls_grad']),completed_updates=steps,
            scope='first/middle/last update while projection off; every update while on',
            legacy_full_epoch_norms=False,warning='Sampled norm means and ratios; not full-epoch norms or actual optimizer updates')
    if supplementary_mode==human.MODE:
        # A non-adaptive supplementary policy has one constant coefficient.
        human_coefficients=coefficient_values if adaptive else [float(supplementary_policy.get('loss_weight',1.0))]
        result['polygon_supervision']=dict(version=human.VERSION,negative_pixel_occurrences=result['supp_selected_pixels'],
            positive_pixel_occurrences=totals.get('polygon_positive_pixels',0),positive_scale=config[human.SCALE],
            effective_negative_weight_min=min(human_coefficients),effective_negative_weight_max=max(human_coefficients),
            effective_positive_weight_min=min(human_coefficients)*config[human.SCALE],effective_positive_weight_max=max(human_coefficients)*config[human.SCALE],
            auxiliary_batchnorm='frozen_running_statistics_only_during_auxiliary_forward',old_weak_label_reads=0)
    return result


from healthy_recovery_runtime_v4 import (active_saved_cuda_ordinal, restore_full_state,
                                         verify_resume_job_identity, delivery_record,
                                         resume_decisions_at_cutoff, resume_history_at_cutoff,
                                         prior_decisions_before_replay, load_epoch_zero_resume,
                                         verify_decision_checkpoint_controls)


def checkpoint_payload(epoch, model, local_d, global_d, model_opt, local_opt, global_opt, config, metrics):
    """A resumable state; branch workers never mutate the owning objects."""
    return {"epoch": int(epoch), "state_dict": model.state_dict(),
            "local_discriminator": local_d.state_dict(), "global_discriminator": global_d.state_dict(),
            "model_optimizer": model_opt.state_dict(), "local_optimizer": local_opt.state_dict(),
            "global_optimizer": global_opt.state_dict(), "config": copy.deepcopy(config), "metrics": metrics,
            "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch": torch.get_rng_state(),
                          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                          "active_cuda_ordinal": next(model.parameters()).device.index
                          if next(model.parameters()).device.type == "cuda" else None}}


def load_resumable_checkpoint(path: Path):
    """Reject truncated/incomplete checkpoints before choosing a recovery point."""
    payload = torch_load(path, "cpu")
    required = {"epoch", "state_dict", "local_discriminator", "global_discriminator",
                "model_optimizer", "local_optimizer", "global_optimizer", "config", "rng_state"}
    missing = sorted(required - set(payload)) if isinstance(payload, dict) else sorted(required)
    if missing:
        raise RuntimeError({"incomplete_checkpoint": str(path), "missing_keys": missing})
    return payload


def restore_resume_worker(job, resume, device, model, local_d, global_d,
                          model_opt, local_opt, global_opt):
    """One checked full-state entry used by both normal and rescue resumes."""
    source_run = Path(resume["source_run_root"])
    checkpoint = Path(resume["checkpoint"])
    if not source_run.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError({"source_run": str(source_run), "checkpoint": str(checkpoint)})
    if not resume.get("checkpoint_sha256") or sha256(checkpoint) != resume["checkpoint_sha256"]:
        raise RuntimeError("resume checkpoint hash missing or mismatched")
    payload = load_resumable_checkpoint(checkpoint)
    epoch = int(payload["epoch"])
    if epoch != int(resume["epoch"]):
        raise RuntimeError("resume epoch mismatch")
    source_job_path = source_run / "job.json"
    if not source_job_path.is_file():
        raise FileNotFoundError("resume source job identity missing")
    source_job = json.loads(source_job_path.read_text())
    if str(source_job.get("protocol_version", "")).startswith("agent-guide-fada-prior"):
        raise ValueError("Historical Prior checkpoints cannot enter v4 recovery")
    identity = verify_resume_job_identity(job, source_job)
    if payload['config'].get('negative_control_policy') != source_job.get('negative_control_policy'):
        raise ValueError('Resume checkpoint negative_control_policy differs from source job')
    ordinal = active_saved_cuda_ordinal(payload, resume, source_run)
    config, restored = restore_full_state(
        payload, expected_epoch=epoch, saved_cuda_ordinal=ordinal, device=device,
        model=model, local_d=local_d, global_d=global_d,
        model_opt=model_opt, local_opt=local_opt, global_opt=global_opt,
    )
    if sha256(checkpoint) != resume["checkpoint_sha256"]:
        raise RuntimeError("source checkpoint changed during restoration")
    # Restore numerical state unchanged, then enable the new repair's
    # pinned diagnostic metadata, including actual-negative probe positions.
    if job.get("negative_response_policy"):
        config["negative_response_policy"] = copy.deepcopy(job["negative_response_policy"])
        restored["diagnostic_policy_after_restore"] = copy.deepcopy(job["negative_response_policy"])
    return epoch, config, {**identity, **restored, "source_checkpoint": str(checkpoint.resolve()),
                           "source_checkpoint_sha256": resume["checkpoint_sha256"],
                           "source_original_unchanged": True}


def _branch_config(base, request):
    result = copy.deepcopy(base)
    update = request["parameter_updates"][0]
    result[update["parameter"]] = update["value"]
    result["adversarial_state"] = request["adversarial_state_after"]
    finish.rebase(result, base)
    result["class_knowledge"]["temperature"] = result["temperature"]
    result["class_knowledge"]["confidence_cap"] = result["confidence_cap"]
    return result


def train_discriminator_probe(gan, model, local_d, global_d, local_opt, global_opt,
                              dataset, source_count, target_count, config, device, rng, steps):
    """D-only paired diagnostic: model and its optimizer are never stepped."""
    model.eval(); local_d.train(); global_d.train()
    src_batches = _fixed_branch_batches(source_count, steps, rng)
    tgt_batches = _fixed_branch_batches(target_count, steps, rng)
    totals = {"D_loss": [], "local_D_acc": [], "global_D_acc": [], "local_coverage": []}
    for sidx, tidx in zip(src_batches, tgt_batches):
        source_batch, target_batch = gan.batch_from(dataset, sidx, tidx, device)
        _, _, source_raw, _ = source_batch
        _, _, _, target_raw = target_batch
        with torch.no_grad():
            sf = gan.stage4_map(model, source_raw); tf = gan.stage4_map(model, target_raw)
            sl, _ = gan.forward_fuse(model, source_raw.float()); tl, _ = gan.forward_fuse(model, target_raw.float())
            sl = resize_segmentation_logits(sl, sf); tl = resize_segmentation_logits(tl, tf)
            sk, sm = class_knowledge_v4(sl, mode=config["local_mode"], temperature=config["temperature"], confidence_cap=config["confidence_cap"], hard_threshold=config["hard_threshold"] or 0.9)
            tk, tm = class_knowledge_v4(tl, mode=config["local_mode"], temperature=config["temperature"], confidence_cap=config["confidence_cap"], hard_threshold=config["hard_threshold"] or 0.9)
        local_opt.zero_grad(set_to_none=True); global_opt.zero_grad(set_to_none=True)
        ld, ls = local_loss(local_d, sf, tf, sk, tk); gd, gs = global_loss(global_d, sf, tf)
        (ld + gd).backward(); local_opt.step(); global_opt.step()
        totals["D_loss"].append(float((ld + gd).item())); totals["local_D_acc"].append(float(ls["accuracy"])); totals["global_D_acc"].append(float(gs["accuracy"])); totals["local_coverage"].append(float((sm.mean() + tm.mean()).item() / 2.0))
    return {key: float(np.mean(value)) for key, value in totals.items()} | {"steps": int(steps), "model_updated": False}


def spatial_comparison(before, after):
    """Compare identical full-region valid-pixel sequences; empty class2 is explicit."""
    if before.shape != after.shape or not before.size:
        raise ValueError("incompatible or empty spatial evidence")
    a, b = before == 2, after == 2
    union = int(np.logical_or(a, b).sum())
    return {"pixels": int(before.size), "all_class_pixel_agreement": float(np.mean(before == after)),
            "class2_iou": float(np.logical_and(a, b).sum() / union) if union else None,
            "class2_union_pixels": union, "class2_gained_pixels": int((b & ~a).sum()),
            "class2_lost_pixels": int((a & ~b).sum()),
            "class2_fraction_delta": float(b.mean() - a.mean()),
            "meaning": "prediction consistency, not target accuracy"}


def run_branch_experiment(gan, common, job, dataset, source_names, source_valid_names,
                          target_names, full_names, supplementary_names, supplementary_image_root,
                          supplementary_label_root, checkpoint, experiment, device, current_review):
    """Run fixed-seed control/treatment branches from one frozen review checkpoint."""
    from legacy_agent_guard_v1 import block_legacy
    block_legacy(job, 'branch_experiment')
    if experiment["kind"] == "custom_code":
        from custom_branch_v4 import run_custom_branch
        return run_custom_branch(job, checkpoint, experiment, device, current_review)
    root = Path(job["output_root"]) / "experiments" / f"epoch_{int(current_review['epoch']):03d}"
    signature = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()[:12]
    root = root / signature
    comparison_path = root / "comparison.json"
    if comparison_path.is_file():
        return json.loads(comparison_path.read_text())
    payload = torch_load(checkpoint, "cpu")
    if int(payload["epoch"]) != int(current_review["epoch"]):
        raise RuntimeError("branch checkpoint is not the current review checkpoint")
    atomic(root / "manifest.json", {"schema_version": 1, "state": "running", "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint), "experiment": experiment, "main_checkpoint_replaced": False})
    saved_py, saved_np, saved_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    baseline_codes = None
    branch_codes = {}

    def one_branch(name, config):
        nonlocal baseline_codes
        branch_root = root / name
        branch_root.mkdir(parents=True, exist_ok=True)
        random.seed(314159); np.random.seed(314159); torch.manual_seed(314159)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(314159)
        model, _ = make_architecture_model(gan, job, device)
        local_d = FineGrainedDomainDiscriminator(768, 4, (256, 128)).to(device)
        global_d = GlobalDomainDiscriminator(768).to(device)
        model.load_state_dict(payload["state_dict"]); local_d.load_state_dict(payload["local_discriminator"]); global_d.load_state_dict(payload["global_discriminator"])
        if baseline_codes is None:
            saved_states = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            baseline_codes = full_target_prediction_metrics(gan, common, model, Path(job["target_root"]), full_names, device, batch_size=16, return_argmax=True)["_argmax_codes"]
            random.setstate(saved_states[0]); np.random.set_state(saved_states[1]); torch.set_rng_state(saved_states[2])
            if saved_states[3] is not None: torch.cuda.set_rng_state_all(saved_states[3])
        model_opt = torch.optim.AdamW(
            model_parameter_groups(model, config["main_lr"], config["main_weight_decay"],
                                   (0.9, 0.999), 1e-8)
        )
        local_opt = torch.optim.AdamW(local_d.parameters(), lr=config["D_lr"], weight_decay=config["D_weight_decay"])
        global_opt = torch.optim.AdamW(global_d.parameters(), lr=config["D_lr"], weight_decay=config["D_weight_decay"])
        model_opt.load_state_dict(copy.deepcopy(payload["model_optimizer"])); local_opt.load_state_dict(copy.deepcopy(payload["local_optimizer"])); global_opt.load_state_dict(copy.deepcopy(payload["global_optimizer"]))
        optimizer_to(model_opt, device); optimizer_to(local_opt, device); optimizer_to(global_opt, device)
        # A short paired branch holds the current scheduled point fixed in
        # both arms; only the proposed scalar differs.
        branch_values = apply_to_optimizers(config, int(payload["epoch"]), model_opt, local_opt, global_opt)
        branch_config = copy.deepcopy(config)
        branch_config.update(branch_values)
        branch_config["_negative_epoch_start"] = int(payload["epoch"])
        branch_config["_negative_seed"] = 314159
        observations = []
        if experiment["kind"] == "discriminator_probe":
            metrics = train_discriminator_probe(gan, model, local_d, global_d, local_opt, global_opt, dataset,
                len(source_names), len(target_names), branch_config, device, np.random.RandomState(314159), int(experiment["steps"]))
        elif job.get("conditional_negative_policy") and int(experiment["steps"]) > 128:
            from conditional_negatives_v2 import epoch_steps
            from paired_horizon_v2 import continue_arm
            rng = np.random.RandomState(314159)
            def train_one_epoch(offset):
                at = int(payload["epoch"]) + offset - 1
                values = apply_to_optimizers(config, at, model_opt, local_opt, global_opt)
                settings = copy.deepcopy(config)
                settings.update(values)
                settings["_negative_epoch_start"] = at
                settings["_negative_seed"] = 314159 + offset
                result = train_epoch(gan, common, model, local_d, global_d, model_opt, local_opt, global_opt, dataset,
                    len(source_names), len(target_names), supplementary_names, supplementary_image_root, supplementary_label_root,
                    settings, device, rng)
                result["scheduled_controls"] = values
                return result
            def review_arm(offset, measured):
                observed = light_review(gan, common, model, source_valid_names, diagnostic_target_names(full_names),
                    Path(job["target_root"]), full_names, device, return_target_codes=True, supplementary_job=job)
                codes = observed.pop("_target_argmax_codes")
                observation = {"epoch": int(payload["epoch"]) + offset, "metrics": measured, "review": observed,
                        "spatial_vs_start": spatial_comparison(baseline_codes, codes), "elapsed_epochs": offset,
                        "completed_steps": offset * epoch_steps(len(source_names), len(target_names))}
                atomic(branch_root / ("review_epoch_%03d.json" % observation["epoch"]), observation)
                return observation
            metrics, observations = continue_arm(total_steps=int(experiment["steps"]),
                steps_per_epoch=epoch_steps(len(source_names), len(target_names)), review_interval=int(job.get("review_interval", 5)),
                train_one_epoch=train_one_epoch, review=review_arm)
            metrics = copy.deepcopy(metrics)
            metrics["steps"] = int(experiment["steps"])
        else:
            metrics = train_epoch(gan, common, model, local_d, global_d, model_opt, local_opt, global_opt, dataset,
                len(source_names), len(target_names), supplementary_names, supplementary_image_root, supplementary_label_root,
                branch_config, device, np.random.RandomState(314159), steps_override=int(experiment["steps"]))
        if not observations:
            metrics["scheduled_controls"] = branch_values
        review = light_review(gan, common, model, source_valid_names, diagnostic_target_names(full_names), Path(job["target_root"]), full_names, device, return_target_codes=True, supplementary_job=job)
        branch_codes[name] = review.pop("_target_argmax_codes")
        result = {"config": flat(config), "metrics": metrics, "review": review,
                  "spatial_vs_start": spatial_comparison(baseline_codes, branch_codes[name])}
        if observations:
            result["observations"] = observations
            result["horizon_semantics"] = "actual complete epochs; schedules advance identically except the declared treatment; compare every recorded endpoint"
        json.dumps(result, allow_nan=False)
        atomic_torch_save(branch_root / "terminal.pt", {"state_dict": model.state_dict(), "local_discriminator": local_d.state_dict(), "global_discriminator": global_d.state_dict(), "config": config})
        atomic(branch_root / "result.json", result)
        del model, local_d, global_d, model_opt, local_opt, global_opt
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return result
    try:
        control = one_branch("control", copy.deepcopy(payload["config"]))
        treatment = one_branch("treatment", _branch_config(payload["config"], experiment))
        comparison = {"schema_version": 1, "state": "completed", "experiment": experiment,
            "start_epoch": int(payload["epoch"]), "control": control, "treatment": treatment,
            "spatial_control_vs_treatment": spatial_comparison(branch_codes["control"], branch_codes["treatment"]),
            "main_checkpoint_replaced": False,
            "interpretation_limit": "paired bounded mechanism evidence at every recorded endpoint, not target-label accuracy"}
        atomic(comparison_path, comparison)
        atomic(root / "manifest.json", {"state": "completed", "checkpoint": str(checkpoint.resolve()), "comparison": str(comparison_path), "main_checkpoint_replaced": False})
        return comparison
    except Exception as exc:
        comparison = {"schema_version": 1, "state": "failed", "experiment": experiment,
                      "error_type": type(exc).__name__, "error": str(exc),
                      "main_checkpoint_replaced": False}
        atomic(root / "failure.json", comparison)
        atomic(root / "manifest.json", comparison)
        return comparison
    finally:
        random.setstate(saved_py); np.random.set_state(saved_np); torch.set_rng_state(saved_torch)
        if saved_cuda is not None: torch.cuda.set_rng_state_all(saved_cuda)


def agent_job_summary(job: dict) -> dict:
    """Expose training provenance without giving the Agent writable paths."""
    source = job["source"]
    return {
        "thread_key": job.get("thread_key", job["job_id"]),
        "job_id": job["job_id"],
        "run_id": job["run_id"],
        "country": job.get("country"),
        "target_year": job.get("target_year"),
        "target_tile": job.get("target_tile", job["job_id"]),
        "target_patch_count": job.get("target_patch_count"),
        "target_train_count": job.get("target_train_count"),
        "target_evaluation": copy.deepcopy(job.get("target_evaluation")),
        "target_domain": copy.deepcopy(job.get("target_domain")),
        "source_profile": job.get("source_profile"),
        "architecture": job.get("architecture"),
        # v4's label-free research boundary overrides the legacy advisory flag
        # in the provenance summary as well as the top-level payload fields.
        "planet_decision_reference": (False if job.get("research_stopselect_policy") == "trend_judgment_v4" or distilled_policy_enabled(job)
                                       else bool(job.get("planet_decision_reference", True))),
        "agentguide_policy": copy.deepcopy(job.get("agentguide_policy")),
        "minimum_training_policy": copy.deepcopy(job.get("minimum_training_policy")),
        "continuation_authorization": copy.deepcopy(job.get("continuation_authorization")),
        "gradient_control_policy": copy.deepcopy(job.get("gradient_control_policy")),
        "cosine_finish_policy": copy.deepcopy(job.get("cosine_finish_policy")),
        "negative_control_policy": copy.deepcopy(job.get("negative_control_policy")),
        "conditional_negative_policy": copy.deepcopy(job.get("conditional_negative_policy")),
        "polygon_supervision_policy": copy.deepcopy(job.get("polygon_supervision_policy")),
        "human_samples_policy": copy.deepcopy(job.get("human_samples_policy")),
        "supplementary_default_policy": job.get("supplementary_default_policy"),
        "per_epoch_policy": job.get("per_epoch_policy"),
        "two_tier_review_policy": job.get("two_tier_review_policy"),
        "deep_review_interval": job.get("deep_review_interval"),
        "evidence_based_stop_policy": job.get("evidence_based_stop_policy"),
        "area_alert_policy": copy.deepcopy(job.get("area_alert_policy")),
        "geographic_prior_policy": copy.deepcopy(job.get("geographic_prior_policy")),
        "research_authorization": job.get("research_authorization"),
        "research_stopselect_policy": job.get("research_stopselect_policy"),
        "research_design": job.get("research_design"),
        "source_year": source.get("source_year"),
        "source_train_count": source.get("source_train_expected_count"),
        "source_valid_count": source.get("source_valid_expected_count"),
        "pretrained_checkpoint": job.get("nas_safe_io", {}).get(
            "original_pretrained", source.get("pretrained")
        ),
        "pretrained_epoch": source.get("entrypoint_pretrained_epoch", 173),
        "max_epoch": max_epoch_for_job(job),
        "minimum_stop_epoch": minimum_training_floor(job),
        "review_interval": job.get("review_interval", 5),
        "seed": job.get("seed", 16),
    }


def call_agent(job, config, variant, planet, epoch, current_review, history, candidates,
               previous_decision=None, init=False, baseline_source_f1=None,
               pre_run=None, experiment_result=None, selection_stage=None, decision_history=None):
    from legacy_agent_guard_v1 import block_legacy
    block_legacy(job, 'direct_agent_call')
    stage = "initialization" if init else ("selection" if epoch == "selection" else f"epoch_{int(epoch):03d}")
    if epoch == "selection" and selection_stage:
        stage = str(selection_stage)
    root = Path(job["output_root"]) / "agent" / stage
    if experiment_result is not None:
        root = root / "experiment_review"
    existing_decision = root / "decision.json"
    from memory_dispatch_v1 import publication_complete
    if existing_decision.is_file() and publication_complete(root / "request.json"):
        # Idempotent recovery: if this checkpoint decision was already
        # completed before a runner interruption, reuse it and do not send a
        # second turn or append a duplicate knowledge-base event.
        request_path = root / "request.json"
        packet = json.loads(request_path.read_text()) if request_path.is_file() else {}
        return json.loads(existing_decision.read_text()), packet
    if job.get("bounded_recovery"):
        from recovery_attempt_v4 import reserve_submission
        reserve_submission(job, stage)
    if init:
        if not pre_run or pre_run.get("formal_training_steps") != 0:
            raise ValueError("initialization requires a measured discardable pre-run report")
        kind = "initialization"
        experience = [] if job.get("planet_decision_reference", True) is False else recent_for_job(job["job_id"])
        packet = request_base(agent_job_summary(job), kind, 0, variant["local_mode"], variant["hard_threshold"], planet, experience)
        packet.update({
            "historical_anchor_config": config,
            "historical_anchor_role": "reference_for_discardable_probe_only_not_a_formal_initial_choice",
            "supplementary_training": job.get("supplementary", {"enabled": False}),
            "baseline_review": initialization_baseline_evidence(current_review),
            "pre_run": initialization_probe_evidence(pre_run),
            "initialization_instruction": (
                "Choose the epoch-1 optimizer and adversarial controls from the measured pre-run, "
                "baseline review, tile history, and parameter bounds. The historical anchor was "
                "used only to construct the discardable probe and is not the chosen configuration."
            ),
            "allowed_parameter_bounds": parameter_bounds(config),
            "allowed_actions": ["ADJUST", "KEEP", "CONTINUE"],
            "previous_decision_sha256": None,
        })
    else:
        kind = "checkpoint_selection" if epoch == "selection" else ("experiment_review" if experiment_result is not None else "training_review")
        if kind == "checkpoint_selection":
            job_summary = agent_job_summary(job)
            job_summary["stopped_after_epoch"] = int(job["stopped_after_epoch"])
            experience = [] if job.get("planet_decision_reference", True) is False else recent_for_job(job["job_id"])
            packet = request_base(job_summary, kind, int(job["stopped_after_epoch"]), variant["local_mode"], variant["hard_threshold"], planet, experience)
            minimum_selection_epoch = int(
                job.get("research_epoch_extension", {}).get("source_epoch", 0)
            )
            policy = checkpoint_selection_policy(
                candidates, minimum_epoch=minimum_selection_epoch,
                formal_distilled=distilled_policy_enabled(job),
                flexible_trend=(job.get("research_stopselect_policy")
                                if job.get("research_stopselect_policy") in {
                                    "trend_judgment_v2", "trend_judgment_v3", "trend_judgment_v4"
                                } else False),
                named_window=(
                    job.get("research_candidate_window")
                    or (((previous_decision or {}).get("current_state_assessment") or {}).get(
                        "candidate_checkpoint_window"
                    ) if job.get("research_stopselect_policy") == "trend_judgment_v4" else None)
                ),
            )
            packet.update({"current_config": config, "supplementary_training": job.get("supplementary", {"enabled": False}), **policy, "allowed_actions": ["SELECT_CHECKPOINT"], "previous_decision_sha256": previous_decision})
        else:
            experience = [] if job.get("planet_decision_reference", True) is False else recent_for_job(job["job_id"])
            packet = request_base(agent_job_summary(job), kind, epoch, variant["local_mode"], variant["hard_threshold"], planet, experience)
            if baseline_source_f1 is None:
                raise ValueError("training review requires baseline source F1")
            source_collapse_floor = max(
                SOURCE_COLLAPSE_ABSOLUTE_F1,
                float(baseline_source_f1) - SOURCE_COLLAPSE_DROP_FROM_BASELINE,
            )
            source_collapsed = (
                not np.isfinite(float(current_review["source_class2_f1"]))
                or float(current_review["source_class2_f1"]) < source_collapse_floor
            )
            max_epoch = max_epoch_for_job(job)
            minimum_stop_epoch = minimum_training_floor(job)
            region_stability = current_review.get("target_region_stability", {})
            full_region_stable = bool(region_stability.get("full_region_stable", False))
            stop_allowed = bool((full_region_stable and config.get("adversarial_state") == "active"
                                 and float(current_review["metrics"].get("adv", 0.0)) > 0
                                 and np.isfinite(float(current_review["metrics"].get("effective_gradient_ratio", 0.0)))
                                 and float(current_review["metrics"].get("effective_gradient_ratio", 0.0)) > 0)
                                or int(epoch) >= max_epoch or source_collapsed)
            allowed_actions = ["ADJUST", "KEEP", "CONTINUE", "PAUSE_ADV", "RESUME_ADV"]
            experiment_policy = job.get("branch_experiment", {})
            attempts_used = len(list((Path(job["output_root"]) / "experiments").glob("epoch_*/*/manifest.json")))
            if (experiment_result is None and bool(experiment_policy.get("enabled", False))
                    and int(experiment_policy.get("max_per_tile", 4)) > attempts_used):
                allowed_actions.append("REQUEST_EXPERIMENT")
            if stop_allowed:
                allowed_actions.append("STOP")
            packet.update({"current_config": config, "supplementary_training": job.get("supplementary", {"enabled": False}), "current_review": current_review, "previous_agent_decision": previous_decision, "action_score_history": history, "allowed_actions": allowed_actions, "previous_decision_sha256": previous_decision, "experiment_capability": {"enabled": experiment_result is None and bool(experiment_policy.get("enabled", False)), "kinds": ["paired_training", "discriminator_probe"], "steps": experiment_policy.get("allowed_steps", [32, 64, 128]), "max_per_tile": int(experiment_policy.get("max_per_tile", 4))}, "experiment_result": experiment_result, "source_protection_policy": {"role": "source_retention_and_catastrophic_safety", "baseline_class2_f1": float(baseline_source_f1), "collapse_absolute_floor": SOURCE_COLLAPSE_ABSOLUTE_F1, "collapse_drop_from_baseline": SOURCE_COLLAPSE_DROP_FROM_BASELINE, "effective_floor": source_collapse_floor, "source_collapsed": bool(source_collapsed), "instruction": "Source F1 is retention and safety evidence, not target quality reward. Sustained material source validation degradation together with gradient conflict may justify controlled diagnosis before collapse."}, "safety_envelope": {"gradient_ratio": current_review["metrics"].get("effective_gradient_ratio", 0.0), "source_class2_f1": current_review["source_class2_f1"], "source_collapsed": bool(source_collapsed), "target_distribution_valid": True, "stop_allowed": stop_allowed, "minimum_stop_epoch": minimum_stop_epoch, "max_epoch": max_epoch, "full_region_stable": full_region_stable, "region_stability": region_stability, "emergency": False}})
            capability = packet["experiment_capability"]
            capability.update({"enabled": "REQUEST_EXPERIMENT" in allowed_actions,
                               "kinds": ["paired_training", "discriminator_probe", "custom_code"],
                               "attempts_used": attempts_used,
                               "custom_code": {"script_path": str((Path(job["output_root"]) / "agent" / f"epoch_{int(epoch):03d}" / "custom_experiment.py").resolve()),
                                               "contract": "self-contained Python script --context context.json --output result.json; result requires state=completed, measurements, limitations",
                                               "timeout_seconds": int(experiment_policy.get("timeout_seconds", 1800))}})
            packet["runtime_artifacts"] = {"output_root": job["output_root"], "job_file": str(Path(job["output_root"]) / "job.json"),
                                           "checkpoint": candidates[-1]["checkpoint"] if candidates else None,
                                           "training_scripts": str(Path(__file__).parent.resolve())}
            packet = apply_agent_early_stopping_policy(packet)
    # Compact numeric history remains available even if thread context was
    # compacted. Reuse existing diagnostics, without rerunning training/probes.
    if not init and kind in {"training_review", "experiment_review"}:
        packet["recent_reviews"] = checkpoint_selection_policy(candidates)["candidate_summaries"][-4:]
    packet["allowed_parameter_bounds"] = parameter_bounds(config)
    if job.get("conditional_negative_policy"):
        from conditional_negatives_v2 import packet_evidence
        packet["conditional_negative_evidence"] = packet_evidence(job)
    # Keep the decision once, rather than duplicating its full text in a hash field.
    if previous_decision is not None:
        packet["previous_agent_decision"] = previous_decision
        packet["previous_decision_sha256"] = hashlib.sha256(json.dumps(previous_decision, sort_keys=True).encode()).hexdigest()
    packet["evidence_semantics"] = {
        "legacy_action_score": ("Planet prevalence gap change only; not verified target quality"
                                 if job.get("planet_decision_reference", True) else
                                 "Planet-derived action scores withheld; no target-quality score supplied"),
        "adjacent_class2_iou": "temporal prediction consistency; not ground-truth IoU",
        "branch_experiments": "paired_executor_available" if experiment_result is None else "experiment_result_under_review",
    }
    packet = enrich_distilled_packet(packet, candidates, decision_history or ([previous_decision] if previous_decision else []))
    if job.get("area_alert_policy"):
        from area_alert_v1 import packet_evidence as area_alert_evidence
        packet["area_alert_evidence"] = area_alert_evidence(
            {**job, "max_epoch": max_epoch_for_job(job)}, current_review, candidates, initialization=init)
    if job.get('user_reported_target_behavior'):
        packet['user_reported_target_behavior'] = copy.deepcopy(job['user_reported_target_behavior'])
    if job.get("geographic_prior"):
        from geographic_prior_v1 import packet_evidence as geographic_prior_evidence
        packet["geographic_prior_evidence"] = geographic_prior_evidence(job, compact=True)
        from prior_semantic_check_v1 import packet_evidence as semantic_prior_evidence
        packet["prior_prediction_semantic_check"] = semantic_prior_evidence(packet)
        from prior_negative_policy_v2 import packet_evidence as prior_negative_evidence
        guidance = prior_negative_evidence(packet)
        if guidance is not None:
            packet["prior_negative_strategy"] = guidance
    if job.get("bounded_recovery"):
        packet["bounded_recovery_runtime"] = {
            "output_root": job["output_root"], "limits": copy.deepcopy(job["bounded_recovery"])}
    if job.get("negative_response_policy"):
        from negative_response_policy_v3 import packet_evidence as negative_response_evidence
        packet["negative_response_guidance"] = negative_response_evidence(packet, job)
    packet = gradient_control.enrich_packet(packet, job)
    packet = finish.packet(packet, job)
    if job.get('human_samples_policy',{}).get('version')=='human_samples_agent_timing_v1':
        packet['information_boundary'].update(CIV_point_validation_exposed=False,
            human_polygon_development_metrics_exposed=False,
            distinction='Current human polygons are training labels; no verified independent polygon holdout is available. Country/year target points remain post-training only.')
    if job.get('automatic_negative_policy'):
        from automatic_negative_production_v4 import evidence as automatic_evidence
        packet['automatic_negative_evidence'] = automatic_evidence(job)
        packet['job']['automatic_negative_policy'] = copy.deepcopy(job['automatic_negative_policy'])
    from prior_relative_spatial_guidance_v1 import enrich as enrich_relative_prior
    packet = enrich_relative_prior(packet, job, current_review, candidates)
    if kind in {'training_review','experiment_review'} and job.get('_joint_negative_candidates',{}).get('epoch')==packet['epoch']:
        packet['joint_negative_candidates'] = copy.deepcopy(job['_joint_negative_candidates'])
    request_path = root / "request.json"; atomic(request_path, packet)
    command = [sys.executable, str(DISPATCH), "--request", str(request_path), "--timeout-seconds", "1800"]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        if agent_dispatch_failure(request_path).get("failure_kind") == "account_usage_limit":
            raise AccountUsageLimitExceeded(
                f"Agent account usage limit at {stage}; keep reviewed checkpoints and retry only after quota recovery"
            ) from exc
        raise
    decision = json.loads((root / "decision.json").read_text())
    record({"type": "agent_decision_request", "job_id": job["job_id"], "run_id": job["run_id"], "variant": variant, "epoch": epoch, "request": str(request_path.resolve()), "decision": decision})
    return decision, packet


def automatic_refresh_due(epoch, interval):
    """Schedule expensive mining only on completed multiples of interval."""
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("automatic-negative refresh interval must be a positive integer")
    return epoch > 0 and epoch % interval == 0


def main(job_path: Path, device_name: str, resume_existing: bool = False):
    if not __debug__:
        raise RuntimeError('Python optimization disables required training validation; run without -O or PYTHONOPTIMIZE')
    trial_root = Path(__file__).resolve().parents[2]
    if (trial_root / "PAUSE_REQUESTED.json").exists() or not (trial_root / "TRAINING_START_AUTHORIZED.json").exists():
        raise RuntimeError("Single-group experiment is paused/not authorized; no CUDA allocation or training has started")
    job = json.loads(job_path.read_text())
    if 'experimental_gate' in job and job['experimental_gate'] is None:
        raise ValueError('experimental gate may not be null; refusing implicit legacy routing')
    from experimental_gate_v1 import validate as validate_experimental_gate
    experimental_gate = validate_experimental_gate(job)
    if experimental_gate:
        raise RuntimeError('experimental policy is launch-blocked: complete target-point isolation of Agent inputs, history and retrieval is not yet verified')
    from per_epoch_contract_guard import validate as validate_per_epoch_contract
    validate_per_epoch_contract(job)
    from automatic_negative_production_v4 import validate_job as validate_automatic_job
    validate_automatic_job(job)
    if str(job.get("protocol_version", "")).startswith("agent-guide-fada-prior"):
        raise ValueError("Historical Prior checkpoints/jobs cannot be resumed as Training Expert v4")
    if "formal_training" in job:
        validate_formal_job(job)
        # A formal job is immutable after preparation. Refuse a mixed-code
        # recovery rather than silently changing the decision/training logic.
        for raw_path, expected in job.get("runtime_provenance", {}).get("files", {}).items():
            path = Path(raw_path)
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError({"immutable_runtime_provenance_mismatch": str(path)})
    out = Path(job["output_root"]); out.mkdir(parents=True, exist_ok=True)
    experimental_pending = out / 'experimental_gate_pending.json'
    if experimental_gate and experimental_pending.exists():
        if not resume_existing:
            raise RuntimeError('experimental pending transaction requires explicit resume and reconciliation')
        from experimental_gate_v1 import reconcile_committed_skip
        reconcile_committed_skip(out, load_resumable_checkpoint)
    research_window_path = out / "research_candidate_window.json"
    if resume_existing:
        from legacy_agent_guard_v1 import block_legacy
        block_legacy(job, 'legacy_resume_discovery')
        if job.get("resume"):
            raise RuntimeError("--resume-existing cannot be combined with a job-defined resume source")
        available = []
        for path in (out / "checkpoints").glob("epoch_*.pt"):
            try:
                epoch = int(path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if (out / f"review_epoch_{epoch:03d}.json").is_file():
                try:
                    load_resumable_checkpoint(path)
                except Exception:
                    continue
                available.append((epoch, path, "full"))
        for receipt in (out / "light_checkpoints").glob("epoch_*_committed.json"):
            try:
                committed = json.loads(receipt.read_text())
                epoch = int(committed["epoch"])
                checkpoint = Path(committed["checkpoint"])
                decision_path = Path(committed["decision"])
                if (not checkpoint.is_file() or sha256(checkpoint) != committed["checkpoint_sha256"]
                        or not decision_path.is_file() or sha256(decision_path) != committed["decision_sha256"]):
                    raise ValueError("invalid light checkpoint receipt")
                load_resumable_checkpoint(checkpoint)
                available.append((epoch, checkpoint, "light"))
            except (OSError, ValueError, KeyError, TypeError):
                raise RuntimeError({"invalid_light_recovery_receipt": str(receipt)})
        if not available:
            raise RuntimeError("--resume-existing requires an existing reviewed checkpoint")
        epoch, checkpoint, recovery_kind = max(available, key=lambda item:item[0])
        pending_light = []
        for pre_path in (out / "light_checkpoints").glob("epoch_*_pre.pt"):
            try:
                pre_epoch = int(pre_path.name.split("_")[1])
            except (ValueError, IndexError):
                raise RuntimeError({"invalid_light_pre_checkpoint": str(pre_path)})
            if pre_epoch > epoch:
                pending_light.append(pre_epoch)
        if pending_light:
            raise RuntimeError({"light_review_requires_exact_reconciliation": sorted(pending_light),
                                "reason": "Do not roll back or duplicate an unresolved decision"})
        # Keep the immutable manifest unchanged.  The normal resume pathway
        # restores model, optimizers, RNG and review candidates from this same
        # run root, then makes the pending decision before another epoch.
        job = copy.deepcopy(job)
        job["resume"] = {"source_run_root": str(out.resolve()), "checkpoint": str(checkpoint.resolve()),
                         "checkpoint_sha256": sha256(checkpoint), "epoch": epoch,
                         "recovery_mode": ("same_run_committed_light" if recovery_kind == "light" else "same_run_existing_checkpoint")}
    if job.get("research_stopselect_policy") == "trend_judgment_v4":
        # The candidate window is a durable research artifact so a controller
        # restart cannot silently widen selection to every historical epoch.
        window_source = (Path(job["resume"]["source_run_root"]) / "research_candidate_window.json"
                         if job.get("resume") else research_window_path)
        if window_source.is_file():
            persisted_window = load_research_candidate_window(window_source)
            if persisted_window:
                job["research_candidate_window"] = [int(value) for value in persisted_window]
    old_status = {}
    status_path = out / "status.json"
    if status_path.exists():
        try:
            old_status = json.loads(status_path.read_text())
        except Exception:
            old_status = {}
    if (job.get("resume", {}).get("recovery_mode") in ("same_run_existing_checkpoint", "same_run_committed_light")
            and job["resume"].get("saved_active_cuda_ordinal") is None):
        previous_device = old_status.get("gpu")
        if isinstance(previous_device, str) and previous_device.startswith("cuda:"):
            job["resume"]["saved_active_cuda_ordinal"] = int(previous_device.split(":", 1)[1])
    device = torch.device(device_name)
    recovery_epoch = int(job["resume"]["epoch"]) if job.get("resume", {}).get("recovery_mode") in ("same_run_existing_checkpoint", "same_run_committed_light") else 0
    atomic(out / "status.json", {"job_id": job["job_id"], "run_id": job["run_id"], "state": "starting", "gpu": device_name, "heartbeat_epoch": recovery_epoch, "phase": "restoring_checkpoint" if resume_existing else "initializing"})
    seed = int(job.get("seed", 16))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    gan, common = configure_runtime(job, device); gan.weighted_source_loss = balanced_binary_source_loss; gan.seed_all(seed)
    source_names = read_lines(Path(job["source"]["source_train"])); source_valid_names = read_lines(Path(job["source"]["source_valid"])); target_names = read_lines(Path(job["target_train_list"])); full_names = names_for_job(job)
    supplementary_config = job.get("supplementary", {})
    supplementary_names = read_lines(Path(supplementary_config["names_file"])) if supplementary_config.get("enabled") else []
    supplementary_image_root = Path(supplementary_config["image_root"]) if supplementary_names else None
    supplementary_label_root = Path(supplementary_config["label_root"]) if supplementary_names else None
    if supplementary_names:
        expected_names_hash = supplementary_config.get("selection", {}).get("names_file_sha256")
        if expected_names_hash and sha256(Path(supplementary_config["names_file"])) != expected_names_hash:
            raise RuntimeError("supplementary selection list hash mismatch")
        expected_count = int(supplementary_config.get("count", len(supplementary_names)))
        if len(supplementary_names) != expected_count:
            raise RuntimeError(f"supplementary count mismatch: {len(supplementary_names)} != {expected_count}")
        for name in supplementary_names:
            image_path = supplementary_image_root / name
            label_path = supplementary_label_path(supplementary_label_root, name)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
    dataset = gan.StrategyDataset(source_names, target_names, "none")
    supplementary_mode = supplementary_config.get("mode", "supervised_segmentation")
    supplementary_loss_names = {
        "supervised_segmentation": "four_class_weighted_cross_entropy",
        "hard_negative_suppression": "thresholded_class2_suppression",
        "weighted_negative_supervision": "balanced_binary_source_loss_with_scaled_supplementary_negatives",
        "class2_suppression": "continuous_known_nonclass2_suppression",
        human.MODE: human.VERSION,
    }
    if supplementary_mode not in supplementary_loss_names:
        raise ValueError(f"unknown supplementary mode: {supplementary_mode}")
    variant = job["variant"]
    config = config_anchor(variant)
    config.update({
        "local_mode": variant["local_mode"],
        "hard_threshold": variant["hard_threshold"],
        "class_knowledge": {"temperature": config["temperature"], "confidence_cap": config["confidence_cap"]},
        "supplementary_training": {
            "enabled": bool(supplementary_names), "count": len(supplementary_names),
            "mode": supplementary_mode, "loss": supplementary_loss_names[supplementary_mode],
            "loss_weight": float(supplementary_config.get("loss_weight", 1.0)),
            "negative_weight_scale": float(supplementary_config.get("negative_weight_scale", 0.5)),
            "hard_threshold": float(supplementary_config.get("hard_threshold", 0.5)),
            "spatial_augmentation": dict(supplementary_config.get("spatial_augmentation", {"enabled": False})),
            "update_interval_steps": int(supplementary_config.get("update_interval_steps", 1)),
            "max_occurrences_per_epoch": supplementary_config.get("max_occurrences_per_epoch"),
            "planet_class2_exempt": True, "planet_unknown_10_ignored": True,
            "enters": "segmentation_classifier_only",
            "excluded_from": ["local_fada_discriminator", "global_fada_discriminator", "domain_adaptation_loss"],
        },
    })
    negative.initialize(config, job)
    from fixed_supplementary_default_v1 import initialize as initialize_supplementary_default
    initialize_supplementary_default(config, job)
    gradient_control.initialize(config, job)
    finish.initialize(config, job)
    if distilled_policy_enabled(job):
        job["planet_decision_reference"] = False
    planet = None if job.get("planet_decision_reference", True) is False else planet_summary(job["job_id"].split("_", 1)[-1])
    model, load_stats = make_architecture_model(gan, job, device)
    local_d = FineGrainedDomainDiscriminator(768, 4, (256, 128)).to(device); global_d = GlobalDomainDiscriminator(768).to(device)
    model_opt = torch.optim.AdamW(
        model_parameter_groups(model, config["main_lr"], config["main_weight_decay"], (0.9, 0.999), 1e-8)
    ); local_opt = torch.optim.AdamW(local_d.parameters(), lr=config["D_lr"], weight_decay=config["D_weight_decay"]); global_opt = torch.optim.AdamW(global_d.parameters(), lr=config["D_lr"], weight_decay=config["D_weight_decay"])
    decisions = []; action_history = []; candidates = []
    def resolve_experiment(decision, checkpoint, current_review):
        if decision["action"] != "REQUEST_EXPERIMENT":
            return decision
        decisions.append(decision)
        result = run_branch_experiment(
            gan, common, job, dataset, source_names, source_valid_names, target_names,
            full_names, supplementary_names, supplementary_image_root, supplementary_label_root,
            checkpoint, decision["experiment"], device, current_review,
        )
        record({"type": "branch_experiment", "job_id": job["job_id"], "run_id": job["run_id"],
                "epoch": current_review["epoch"], "experiment": decision["experiment"], "result": result})
        reviewed, _ = decide(
            job, config, variant, planet, current_review["epoch"], current_review,
            action_history, candidates, previous_decision=decision,
            baseline_source_f1=baseline_source_f1, experiment_result=result, decision_history=decisions,
        )
        return reviewed
    # These are transient controller state, never Agent bookkeeping.  Only the
    # previous full-region argmax sequence is kept in memory; review JSON stores
    # compact stability summaries instead of a giant prediction array.
    previous_target_argmax = None
    stability_observations = []
    automatic_controller = None
    if job.get('automatic_negative_policy'):
        from automatic_negative_production_v4 import Controller as AutomaticNegativeController
        automatic_controller = AutomaticNegativeController(job, gan, common, model, device)
    start_epoch = 0
    resume = job.get("resume")
    if resume:
        from legacy_agent_guard_v1 import block_legacy
        block_legacy(job, 'legacy_resume_history')
        source_run = Path(resume["source_run_root"])
        resume_checkpoint = Path(resume["checkpoint"])
        start_epoch, config, restore_receipt = restore_resume_worker(
            job, resume, device, model, local_d, global_d, model_opt, local_opt, global_opt,
        )
        actual_hash = restore_receipt["source_checkpoint_sha256"]
        atomic(out / "full_state_restore.json", restore_receipt)
        for review_path in sorted(source_run.glob("review_epoch_*.json")):
            review_payload = json.loads(review_path.read_text())
            candidate = review_payload.get("candidate")
            if candidate and int(candidate["epoch"]) <= start_epoch:
                candidates.append(candidate)
                atomic(out / review_path.name, {**review_payload, "inherited_from": str(source_run.resolve())})
        if (not candidates or int(candidates[0]["epoch"]) != 0 or
                (resume.get("recovery_mode") != "same_run_committed_light" and int(candidates[-1]["epoch"]) != start_epoch)):
            raise RuntimeError("resume candidate history is incomplete")
        baseline_source_f1 = float(candidates[0]["source_class2_f1"])
        decisions_path = source_run / "agent_decisions.json"
        history_path = source_run / "action_scores.json"
        # Decision files are append-only per review and survive a failed
        # controller restart more reliably than the aggregate convenience
        # file.  Prefer them when recovering a run in place.
        recovered_decisions = []
        init_decision = source_run / "agent" / "initialization" / "decision.json"
        if init_decision.is_file():
            recovered_decisions.append(json.loads(init_decision.read_text()))
        for review_decision in sorted((source_run / "agent").glob("epoch_*/decision.json")):
            item = json.loads(review_decision.read_text())
            if int(item.get("epoch", -1)) <= start_epoch:
                recovered_decisions.append(item)
        for light_decision in sorted((source_run / "agent").glob("epoch_*/light_review/decision.json")):
            item = json.loads(light_decision.read_text())
            if int(item.get("epoch", -1)) <= start_epoch:
                recovered_decisions.append(item)
        recovered_decisions.sort(key=lambda item: int(item.get("epoch", 0)))
        decisions = resume_decisions_at_cutoff(
            recovered_decisions or (json.loads(decisions_path.read_text()) if decisions_path.is_file() else []),
            start_epoch,
        )
        if (resume.get('recovery_mode') == 'same_run_existing_checkpoint'
                and start_epoch > 0):
            # The checkpoint is saved before this epoch's decision. call_agent
            # reuses a published decision, which is then applied and appended
            # below. Exclude it here so history cannot contain it twice.
            decisions = prior_decisions_before_replay(decisions, start_epoch)
        inherited_history = json.loads(history_path.read_text()) if history_path.is_file() else []
        action_history = resume_history_at_cutoff(
            inherited_history, start_epoch,
            [int(path.stem.rsplit("_", 1)[1]) for path in source_run.glob("review_epoch_*.json")],
        )
        if resume.get("recovery_mode") == "same_run_committed_light":
            # This checkpoint is AFTER the light decision and its controls.
            # Do not replay that decision or claim a target map was measured.
            receipt_path = source_run / "light_checkpoints" / ("epoch_%03d_committed.json" % start_epoch)
            committed = json.loads(receipt_path.read_text())
            resume_decision = json.loads(Path(committed["decision"]).read_text())
            if (committed["checkpoint_sha256"] != actual_hash or
                    committed["decision_sha256"] != sha256(committed["decision"])):
                raise RuntimeError("light recovery identity mismatch")
            previous_target_argmax = None
            stability_observations = []
            atomic(out / "resume.json", {"source_run_root": str(source_run.resolve()),
                "checkpoint": str(resume_checkpoint.resolve()), "checkpoint_sha256": actual_hash,
                "resume_epoch": start_epoch, "review_mode": "committed_light",
                "decision_reapplied": False, "target_map_recomputed": False})
        elif resume.get("recovery_mode") == "same_run_existing_checkpoint" and start_epoch == 0:
            # Epoch 0 was saved after initialization was applied. Reuse its
            # durable decision and state; it is not a trained review and must
            # not launch another Agent turn or apply initialization twice.
            from memory_dispatch_v1 import publication_complete
            resume_decision, baseline_source_f1, baseline_fraction = load_epoch_zero_resume(
                source_run, publication_complete, config)
            resume_target_metric = full_target_prediction_metrics(
                gan, common, model, Path(job["target_root"]), full_names, device,
                batch_size=16, return_argmax=True,
            )
            previous_target_argmax = resume_target_metric.pop("_argmax_codes")
            stability_observations = [{"epoch": 0, "class2_fraction": baseline_fraction,
                                       "region_comparison": None}]
            action_history = [{"decision_epoch": 0, "action": resume_decision["action"],
                "changed_parameters": resume_decision["intervention"]["changed_parameters"],
                "schedule_updates": [], "model_fraction_before": baseline_fraction,
                "model_fraction_after": None,
                "gap_before": abs(baseline_fraction - planet["class2_fraction"]) if job.get("planet_decision_reference", True) else None,
                "score": None, "improved": None}]
            atomic(out / "resume.json", {"source_run_root": str(source_run.resolve()),
                "checkpoint": str(resume_checkpoint.resolve()), "checkpoint_sha256": actual_hash,
                "resume_epoch": 0, "max_epoch": max_epoch_for_job(job),
                "minimum_stop_epoch": minimum_training_floor(job),
                "initialization_reused_without_reapplication": True})
            atomic(out / "config_after_resume.json", config)
            atomic(out / "agent_decisions.json", decisions)
            atomic(out / "action_scores.json", action_history)
        else:
            resume_review_payload = json.loads((source_run / f"review_epoch_{start_epoch:03d}.json").read_text())
            resume_review = resume_review_payload["review"]
            current_review = {
                "epoch": start_epoch,
                "human_supervision_evaluation": resume_review.get('human_supervision_evaluation'),
                "source_class2_f1": float(resume_review["source_class2_f1"]),
                "target_class2_fraction": float(resume_review["target_class2_fraction"]),
                "metrics": resume_review["metrics"],
                "source_validation": resume_review.get("source"),
                "target_unsupervised": resume_review.get("target"),
                "fresh_frozen_feature_probe": resume_review.get("fresh_probe"),
                "target_region_stability": resume_review.get("target_region_stability", {
                    "scope": evaluation_scope(full_names),
                    "full_region_stable": False,
                    "stable_reason": "resume_requires_fresh_spatial_comparisons",
                    "review_points": 1,
                    "required_review_points": REGION_STABILITY_WINDOW_REVIEWS,
                }),
                "discriminator": {
                    "local": resume_review["metrics"]["local_D_acc"],
                    "global": resume_review["metrics"]["global_D_acc"],
                },
            }
            resume_target_metric = full_target_prediction_metrics(
                gan, common, model, Path(job["target_root"]), full_names, device,
                batch_size=16, return_argmax=True,
            )
            previous_target_argmax = resume_target_metric.pop("_argmax_codes")
            stability_observations = [{
                "epoch": start_epoch,
                "class2_fraction": float(resume_review["target_class2_fraction"]),
                "region_comparison": None,
            }]
            atomic(out / f"review_epoch_{start_epoch:03d}.json", {**resume_review_payload, "inherited_from": str(source_run.resolve())})
            atomic(out / "resume.json", {"source_run_root": str(source_run.resolve()), "checkpoint": str(resume_checkpoint.resolve()), "checkpoint_sha256": actual_hash, "resume_epoch": start_epoch, "max_epoch": max_epoch_for_job(job), "minimum_stop_epoch": minimum_training_floor(job)})
            # A previously published old-policy decision may have committed a
            # pool at a non-fifth epoch. Replay that exact pool on migration;
            # applying the new schedule retroactively would silently discard it.
            prior_pool_manifest = out / "automatic_negative" / ("epoch_%03d" % start_epoch) / "manifest.json"
            resume_refresh_due = (automatic_controller is not None and start_epoch < max_epoch_for_job(job)
                and (prior_pool_manifest.is_file() or automatic_refresh_due(
                    start_epoch, job["automatic_negative_policy"].get("refresh_at_trained_review", 1))))
            if resume_refresh_due:
                from legacy_agent_guard_v1 import pool_prepare
                pool_prepare(job, automatic_controller, start_epoch, resume_checkpoint, current_review, config)
            if job.get("bounded_recovery"):
                from recovery_attempt_v4 import load_resume_strategy
                resume_decision = load_resume_strategy(job, config)
            else:
                atomic(out / "status.json", {"job_id": job["job_id"], "run_id": job["run_id"], "state": "starting", "gpu": device_name, "heartbeat_epoch": start_epoch, "phase": "awaiting_resume_review"})
                resume_decision, _ = decide(job, config, variant, planet, start_epoch, current_review, action_history, candidates, previous_decision=(decisions[-1] if decisions else None), baseline_source_f1=baseline_source_f1, decision_history=decisions)
            resume_decision = resolve_experiment(resume_decision, resume_checkpoint, current_review)
            if resume.get('recovery_mode') == 'same_run_existing_checkpoint' and start_epoch > 0:
                verify_decision_checkpoint_controls(resume_decision, config, 'before')
            guard_minimum_training(job,start_epoch,resume_decision["continue_training"],source_collapsed=bool(current_review["source_class2_f1"] < max(SOURCE_COLLAPSE_ABSOLUTE_F1, baseline_source_f1-SOURCE_COLLAPSE_DROP_FROM_BASELINE)))
            if job.get("research_stopselect_policy") == "trend_judgment_v4":
                proposed_window = ((resume_decision.get("current_state_assessment") or {})
                                   .get("candidate_checkpoint_window"))
                if proposed_window:
                    job["research_candidate_window"] = [int(value) for value in proposed_window]
                    atomic(research_window_path, {
                        "candidate_checkpoint_window": job["research_candidate_window"],
                        "source_review_epoch": int(start_epoch),
                        "source_decision_sha256": hashlib.sha256(
                            json.dumps(resume_decision, sort_keys=True).encode()
                        ).hexdigest(),
                        "label_free": True,
                    })
            before_fraction = current_review["target_class2_fraction"]
            action_history.append({"decision_epoch": start_epoch, "action": resume_decision["action"], "changed_parameters": resume_decision["intervention"]["changed_parameters"], "schedule_updates": resume_decision["intervention"].get("schedule_updates", []), "model_fraction_before": before_fraction, "model_fraction_after": None, "gap_before": abs(before_fraction - planet["class2_fraction"]) if job.get("planet_decision_reference", True) else None, "score": None, "improved": None})
            decisions.append(resume_decision)
            apply_config(config, resume_decision, model_opt, local_opt, global_opt)
            if resume_refresh_due and resume_decision["continue_training"]:
                from legacy_agent_guard_v1 import pool_refresh
                pool_refresh(job, automatic_controller, start_epoch, resume_checkpoint, current_review, config)
            atomic(out / "config_after_resume.json", config); atomic(out / "agent_decisions.json", decisions); atomic(out / "action_scores.json", action_history)
    else:
        init_path = out / "agent" / "initialization" / "decision.json"
        if init_path.exists():
            # Recovery of an interrupted/stale isolated run must not create a
            # duplicate Agent turn or a second initialization record in the shared
            # knowledge base.  A runner can die before its exception handler writes
            # state=failed, so the initialization artifact is authoritative.
            init_decision = json.loads(init_path.read_text())
        else:
            baseline_review = light_review(
                gan, common, model, source_valid_names,
                diagnostic_target_names(full_names), Path(job["target_root"]),
                full_names, device, return_target_codes=True,
             supplementary_job=job)
            previous_target_argmax = baseline_review.pop("_target_argmax_codes")
            pre_run = initialization_probe(
                gan, common, model, local_d, global_d, dataset,
                source_names, target_names, supplementary_names,
                supplementary_image_root, supplementary_label_root, config, device,
            )
            if job.get('initialization_trial_policy'):
                from initialization_trials_v1 import run as run_initialization_trials
                pre_run['matched_candidate_trials']=run_initialization_trials(
                    sys.modules[__name__],gan,common,model,local_d,global_d,dataset,
                    source_names,source_valid_names,target_names,config,job,device,
                    (model_opt,local_opt,global_opt))
            atomic(out / "initialization_probe.json", pre_run)
            init_decision, _ = decide(
                job, config, variant, planet, 0, baseline_review,
                action_history, candidates, init=True, pre_run=pre_run,
            )
        apply_config(config, init_decision, model_opt, local_opt, global_opt); decisions.append(init_decision)
        atomic(out / "config_after_initialization.json", config)
        if "baseline_review" not in locals():
            baseline_review = light_review(gan, common, model, source_valid_names, diagnostic_target_names(full_names), Path(job["target_root"]), full_names, device, return_target_codes=True, supplementary_job=job)
            previous_target_argmax = baseline_review.pop("_target_argmax_codes")
        baseline_fraction = float(baseline_review["target_unsupervised"]["class_distribution"][2]);
        baseline_source_f1 = float(baseline_review["source"]["f1_per_class"][2])
        baseline_review["target_region_stability"] = {
            "scope": evaluation_scope(full_names),
            "full_region_stable": False,
            "stable_reason": "insufficient_review_window",
            "window_epochs": [0],
            "required_review_points": REGION_STABILITY_WINDOW_REVIEWS,
            "review_points": 1,
            "required_transitions": REGION_STABILITY_WINDOW_REVIEWS - 1,
        }
        stability_observations = [{
            "epoch": 0, "class2_fraction": baseline_fraction,
            "region_comparison": None,
        }]
        action_history.append({"decision_epoch": 0, "action": init_decision["action"], "changed_parameters": init_decision["intervention"]["changed_parameters"], "schedule_updates": [], "model_fraction_before": baseline_fraction, "model_fraction_after": None, "gap_before": abs(baseline_fraction - planet["class2_fraction"]) if job.get("planet_decision_reference", True) else None, "score": None, "improved": None})
        checkpoint0 = out / "checkpoints/epoch_000.pt"; checkpoint0.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(checkpoint0, checkpoint_payload(0, model, local_d, global_d, model_opt, local_opt, global_opt, config, {}))
        baseline_candidate = {"epoch": 0, "checkpoint": str(checkpoint0.resolve()), "checkpoint_sha256": sha256(checkpoint0), "target_class2_fraction": baseline_fraction, "source_class2_f1": baseline_source_f1, "review": {"epoch": 0, "source_class2_f1": baseline_source_f1, "target_class2_fraction": baseline_fraction, "source": baseline_review["source"], "target": baseline_review["target_unsupervised"]}}
        if baseline_review.get('human_supervision_evaluation') is not None:
            baseline_candidate['review']['human_supervision_evaluation']=baseline_review['human_supervision_evaluation']
        candidates.append(baseline_candidate)
        atomic(out / "review_epoch_000.json", {"epoch": 0, "source_class2_f1": baseline_source_f1, "target_class2_fraction": baseline_fraction, "review": baseline_review, "candidate": baseline_candidate})
    max_epoch = max_epoch_for_job(job); review_interval = int(job.get("review_interval", 5))
    minimum_stop_epoch = minimum_training_floor(job)
    allowed_maxima = ({max_epoch} if job.get("bounded_recovery")
                      else {100, 200})
    if job.get('pilot_only'):
        pilot = job['pilot_only']
        if (pilot.get('version') != 'fresh_E173_to_review5_controlled_auto_negative_v1'
                or pilot.get('production_delivery') is not False
                or pilot.get('branch_weights_adopted') is not False
                or job.get('source_profile') != 'source2019_benin_e173'
                or job.get('source',{}).get('source_year') != 2019
                or job.get('resume') or max_epoch != 5):
            raise ValueError('Unrecognized disposable automatic-negative pilot budget')
        allowed_maxima = {5}
    if max_epoch not in allowed_maxima or not DEFAULT_MINIMUM_STOP_EPOCH <= minimum_stop_epoch <= max_epoch:
        raise ValueError({"training_schedule_must_be": {"max_epoch": sorted(allowed_maxima), "minimum_stop_epoch": f">={DEFAULT_MINIMUM_STOP_EPOCH} and <= max_epoch"}, "received": {"max_epoch": max_epoch, "minimum_stop_epoch": minimum_stop_epoch}})
    training_epochs = range(start_epoch + 1, max_epoch + 1)
    if resume and not resume_decision["continue_training"]:
        training_epochs = ()
    for epoch in training_epochs:
        if experimental_gate:
            atomic(experimental_pending, {'epoch':epoch, 'state':'epoch_uncommitted', 'version':'experimental_gate_v1'})
        scheduled_values = apply_to_optimizers(config, epoch, model_opt, local_opt, global_opt)
        epoch_config = copy.deepcopy(config)
        epoch_config.update(scheduled_values)
        epoch_config["_negative_epoch_start"] = epoch-1
        epoch_config["_negative_seed"] = seed + epoch*100000
        automatic_pool = automatic_controller.for_epoch(epoch_config) if automatic_controller is not None else None
        if experimental_gate:
            sampling=experimental_gate.get('recorder_sampling')
            if sampling:
                from bounded_recorder_candidate_v1 import BoundedRecorder
                experimental_recorder=BoundedRecorder(count=sampling['elements_per_tensor'],max_tensors=sampling['max_tensors_per_group'])
            else:
                from experimental_gate_v1 import StepRecorder
                experimental_recorder = StepRecorder()
        else:
            experimental_recorder = None
        metrics = train_epoch(gan, common, model, local_d, global_d, model_opt, local_opt, global_opt, dataset, len(source_names), len(target_names), supplementary_names, supplementary_image_root, supplementary_label_root, epoch_config, device, np.random.RandomState(seed + epoch), automatic_negative_runtime=automatic_pool, experimental_recorder=experimental_recorder)
        if automatic_controller is not None:
            automatic_controller.after_epoch(config, metrics)
        metrics["scheduled_controls"] = scheduled_values
        if experimental_gate:
            metrics['experimental_actual_updates']['training_controls'] = {
                k: epoch_config.get(k) for k in ('temperature', 'confidence_cap', 'hard_threshold',
                    'lambda_adv', 'main_lr', 'D_lr', 'main_weight_decay', 'D_weight_decay',
                    'D_steps_per_G_step', 'adversarial_projection_strength',
                    'supplementary_loss_weight', 'supplementary_positive_scale',
                    'supplementary_occurrence_cap', 'supplementary_interval_steps',
                    'supplementary_horizon_epochs', 'main_lr_finish_epochs') if k in epoch_config}
        atomic(out / "status.json", {"job_id": job["job_id"], "run_id": job["run_id"], "state": "training", "gpu": device_name, "heartbeat_epoch": epoch, "metrics": metrics})
        atomic(out / "epoch_facts" / f"epoch_{epoch:03d}.json", {"epoch": epoch, "record": {"metrics": metrics}, "meaning": "Exact completed numerical epoch metrics; not an Agent proposal"})
        if experimental_gate:
            from experimental_gate_v1 import classify as classify_experimental, epoch_integrity, fingerprint as fingerprint_experimental, automatic_context, supplementary_context, stable_config, FIELDS
            gate_dir = out / 'experimental_gate'; gate_dir.mkdir(parents=True, exist_ok=True)
            old = []
            for ep in range(max(1, epoch-experimental_gate['lookback']), epoch):
                p = gate_dir / ('epoch_%03d.json' % ep)
                if p.is_file(): old.append(json.loads(p.read_text()))
            auto_context, auto_transition = automatic_context(automatic_controller is not None, automatic_pool is not None, config, metrics)
            # Routine, actually occurring supplementary supervision is not an
            # anomaly by itself. Review changed controls and missing expected
            # exposure; otherwise a healthy active pool would call the Agent
            # on every epoch and defeat the evidence-gated policy.
            supp_state, supp_transition = supplementary_context(metrics)
            # Scheduled values are still measured every epoch. The schedule's
            # definition lives in config; its expected numerical progression
            # must not be mistaken for an unplanned config change.
            fp = fingerprint_experimental(stable_config(config), {'automatic':auto_context,
                                            'supplementary':supp_state})
            active_aux = bool(auto_transition or supp_transition)
            # This synchronous loop reaches a new epoch only after the prior
            # action has returned; unresolved transactions block resumption.
            # An already executed adjustment is evidence to follow at the next
            # full review, not by itself an abnormal condition.
            pending_action = False
            gate_decision = classify_experimental(epoch, metrics, fp, old, experimental_gate, active_aux=active_aux, pending_action=pending_action, adversarial_active=epoch_config.get('adversarial_state')=='active')
            integrity=epoch_integrity(metrics,epoch_config,scheduled_values)
            if integrity['route']=='SAFETY_STOP':gate_decision={'route':'SAFETY_STOP','reason':'epoch execution integrity','integrity':integrity}
            elif integrity['route']=='LIGHT_REVIEW' and gate_decision['route']!='FULL_REVIEW':gate_decision={'route':'LIGHT_REVIEW','reason':'epoch execution evidence incomplete','integrity':integrity}
            gate_fact = {'epoch':epoch, 'config_hash':fp, 'facts':{k:metrics.get(k) for k in FIELDS},
                         'stable':not (active_aux or pending_action) and integrity['route']=='OK', 'gate':gate_decision,'integrity':integrity}
            atomic(gate_dir / ('epoch_%03d.json' % epoch), gate_fact)
            if gate_decision['route'] == 'SAFETY_STOP':
                raise RuntimeError('experimental gate numerical failure: safe pause')
        if epoch % review_interval and epoch < max_epoch:
            continue
        from two_tier_review_policy import mode as tier_mode, light_observation
        if (job.get("two_tier_review_policy") == "light_per_epoch_deep_every_five_v1"
                and tier_mode(epoch, int(job["deep_review_interval"])) == "light" and epoch < max_epoch):
            # No full source validation, target panel, pseudo-negative mining,
            # normal STOP or selection at light reviews. Exact pre-action state
            # permits reconciliation without repeating an uncertain model call.
            from legacy_agent_guard_v1 import light_review as dispatch_light
            light_dir = out / "light_checkpoints"
            light_dir.mkdir(parents=True, exist_ok=True)
            pre = light_dir / ("epoch_%03d_pre.pt" % epoch)
            atomic_torch_save(pre, checkpoint_payload(epoch, model, local_d, global_d,
                model_opt, local_opt, global_opt, config, metrics))
            if experimental_gate and experimental_gate['mode'] == 'enforce' and gate_decision['route'] == 'SKIP':
                # A machine gate is not an Agent response. The durable local
                # receipt permits exactly-once recovery without inventing one.
                from fixed_supplementary_default_v1 import advance as advance_supplementary_default
                default_event = advance_supplementary_default(config, epoch)
                if default_event is not None:
                    atomic(out / ('supplementary_default_epoch_%03d.json' % epoch), default_event)
                gate_receipt = gate_dir / ('epoch_%03d_skip.json' % epoch)
                atomic(gate_receipt, {'epoch':epoch, 'action':'KEEP', 'continue_training':True,
                    'decision_kind':'experimental_machine_gate', 'reason':gate_decision['reason']})
                post = light_dir / ('epoch_%03d_post.pt' % epoch)
                atomic_torch_save(post, checkpoint_payload(epoch, model, local_d, global_d,
                    model_opt, local_opt, global_opt, config, metrics))
                atomic(light_dir / ('epoch_%03d_committed.json' % epoch), {
                    'epoch':epoch, 'checkpoint':str(post.resolve()), 'checkpoint_sha256':sha256(post),
                    'decision':str(gate_receipt.resolve()), 'decision_sha256':sha256(gate_receipt),
                    'review_mode':'experimental_machine_gate', 'normal_stop_allowed':False})
                experimental_pending.unlink()
                continue
            recent = []
            for previous_epoch in range(max(1, epoch-4), epoch):
                fact = out / "epoch_facts" / ("epoch_%03d.json" % previous_epoch)
                if fact.is_file():
                    prev_metrics = json.loads(fact.read_text())["record"]["metrics"]
                    # Historical base controls are unknown here; their measured
                    # effective controls remain in scheduled_controls.
                    fact_item = light_observation(previous_epoch, prev_metrics, {})
                    previous_light = out / "agent" / ("epoch_%03d" % previous_epoch) / "light_review" / "decision.json"
                    previous_commit = out / "light_checkpoints" / ("epoch_%03d_committed.json" % previous_epoch)
                    if previous_light.is_file() and previous_commit.is_file():
                        recorded = json.loads(previous_light.read_text())
                        fact_item["accepted_light_action"] = {
                            "action": recorded.get("action"),
                            "automatic_negative_action": recorded.get("automatic_negative_action"),
                            "changed_parameters": recorded.get("intervention", {}).get("changed_parameters", []),
                            "expected_response": recorded.get("expected_response"),
                            "reversal_condition": recorded.get("reversal_condition")}
                    recent.append(fact_item)
            light_decision = dispatch_light(job, epoch, metrics, config, recent)
            apply_config(config, light_decision, model_opt, local_opt, global_opt)
            if light_decision.get("automatic_negative_action") == "DISABLE":
                state = config.get("_automatic_negative_state") or {}
                if not state.get("weight", 0) > 0:
                    raise RuntimeError("Light review cannot disable an inactive automatic-negative pool")
                state["weight"] = 0.0
                state["disabled_by_light_review_epoch"] = epoch
                atomic(out / "automatic_negative" / "state.json", state)
            # Default schedule remains independent of Agent choices.
            from fixed_supplementary_default_v1 import advance as advance_supplementary_default
            default_event = advance_supplementary_default(config, epoch)
            if default_event is not None:
                atomic(out / ("supplementary_default_epoch_%03d.json" % epoch), default_event)
            post = light_dir / ("epoch_%03d_post.pt" % epoch)
            atomic_torch_save(post, checkpoint_payload(epoch, model, local_d, global_d,
                model_opt, local_opt, global_opt, config, metrics))
            atomic(light_dir / ("epoch_%03d_committed.json" % epoch), {
                "epoch": epoch, "checkpoint": str(post.resolve()), "checkpoint_sha256": sha256(post),
                "decision": str((out / "agent" / ("epoch_%03d" % epoch) / "light_review" / "decision.json").resolve()),
                "decision_sha256": sha256(out / "agent" / ("epoch_%03d" % epoch) / "light_review" / "decision.json"),
                "review_mode": "light", "normal_stop_allowed": False})
            decisions.append(light_decision)
            atomic(out / "agent_decisions.json", decisions)
            if experimental_gate:
                experimental_pending.unlink()
            continue
        review = light_review(gan, common, model, source_valid_names, diagnostic_target_names(full_names), Path(job["target_root"]), full_names, device, return_target_codes=True, supplementary_job=job)
        current_target_argmax = review.pop("_target_argmax_codes")
        current_fraction = float(review["target_unsupervised"]["class_distribution"][2]); source_f1 = float(review["source"]["f1_per_class"][2])
        region_stability = assess_full_region_stability(
            current_target_argmax, previous_target_argmax, stability_observations,
            action_history, epoch, current_fraction, scope=evaluation_scope(full_names),
        )
        review["target_region_stability"] = region_stability
        current = {"epoch": epoch, "source_class2_f1": source_f1, "target_class2_fraction": current_fraction, "metrics": metrics, "source": review["source"], "target": review["target_unsupervised"], "target_region_stability": region_stability, "adversarial_state": config["adversarial_state"], "schedule_state": copy.deepcopy(config["schedule_state"]), "fresh_probe": review.get("independent_frozen_feature_probe"), "local_D_accuracy": metrics["local_D_acc"], "global_D_accuracy": metrics["global_D_acc"]}
        if review.get('human_supervision_evaluation') is not None:
            current['human_supervision_evaluation']=review['human_supervision_evaluation']
        if action_history:
            previous = action_history[-1]
            # The action's before value is the model state observed at the
            # decision epoch.  Its outcome is evaluated at this later review.
            # Do not use the previous action's after value as the before value.
            before_fraction = previous["model_fraction_before"]
            if job.get("planet_decision_reference", True):
                before_gap = abs(before_fraction - planet["class2_fraction"]); after_gap = abs(current_fraction - planet["class2_fraction"]); score = before_gap - after_gap
                action_history[-1].update({"outcome_epoch": epoch, "model_fraction_after": current_fraction, "model_fraction_observed": current_fraction, "gap_before": before_gap, "gap_after": after_gap, "score": score, "improved": bool(score > 0)})
                record_once({"type": "action_outcome", "job_id": job["job_id"], "run_id": job["run_id"], "variant": variant, "decision_epoch": previous["decision_epoch"], "review_epoch": epoch, "planet_class2_fraction": planet["class2_fraction"], "model_fraction_before": previous["model_fraction_before"], "model_fraction_after": current_fraction, "gap_before": previous["gap_before"], "gap_after": after_gap, "action_score": score, "improved": bool(score > 0)})
            else:
                action_history[-1].update({"outcome_epoch": epoch, "model_fraction_after": current_fraction,
                                           "model_fraction_observed": current_fraction,
                                           "score": None, "improved": None})
        from fixed_supplementary_default_v1 import advance as advance_supplementary_default
        default_event = advance_supplementary_default(config, epoch)
        if default_event is not None:
            atomic(out / f"supplementary_default_epoch_{epoch:03d}.json", default_event)
        checkpoint = out / "checkpoints" / f"epoch_{epoch:03d}.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(checkpoint, checkpoint_payload(epoch, model, local_d, global_d, model_opt, local_opt, global_opt, config, metrics))
        candidate = {"epoch": epoch, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "target_class2_fraction": current_fraction, "source_class2_f1": source_f1, "review": current}; candidates.append(candidate)
        atomic(out / f"review_epoch_{epoch:03d}.json", {"review": current, "candidate": candidate, "action_scores": action_history})
        comparison = region_stability["adjacent_comparisons"][-1] if region_stability["adjacent_comparisons"] else None
        stability_observations.append({"epoch": epoch, "class2_fraction": current_fraction, "region_comparison": comparison})
        previous_target_argmax = current_target_argmax
        current_review = {
            "epoch": epoch,
            "human_supervision_evaluation": review.get('human_supervision_evaluation'),
            "source_class2_f1": source_f1,
            "target_class2_fraction": current_fraction,
            "metrics": metrics,
            "source_validation": review["source"],
            "target_unsupervised": review["target_unsupervised"],
            "fresh_frozen_feature_probe": review.get("independent_frozen_feature_probe"),
            "metric_provenance": review.get("metric_provenance"),
            "target_region_stability": region_stability,
            "discriminator": {
                "local": metrics["local_D_acc"],
                "global": metrics["global_D_acc"],
            },
        }
        if experimental_gate:
            # Only these fixed, label-free sampled training facts are copied.
            # This is not permission to expose the legacy Agent evidence path.
            current_review['experimental_five_epoch_evidence'] = []
            for ep in range(max(1,epoch-4),epoch+1):
                ep_path = out / 'epoch_facts' / ('epoch_%03d.json' % ep)
                if ep_path.is_file():
                    ep_metrics = json.loads(ep_path.read_text())['record']['metrics']
                    updates = ep_metrics.get('experimental_actual_updates', {})
                    current_review['experimental_five_epoch_evidence'].append({
                        'epoch':ep, 'steps':updates.get('steps'), 'optimizer_groups':updates.get('optimizer_groups'),
                        'training_controls':updates.get('training_controls'), 'sample_groups':updates.get('sample_groups'),
                        'logit_response':updates.get('logit_response')})
            atomic(out / 'experimental_gate' / ('full_review_evidence_%03d.json' % epoch),
                   {'epoch':epoch,'training_only':current_review['experimental_five_epoch_evidence']})
        # All epochs retain scientific review; expensive mining is periodic.
        refresh_due = automatic_controller is not None and epoch < max_epoch and automatic_refresh_due(
            epoch, job["automatic_negative_policy"].get("refresh_at_trained_review", 1))
        if refresh_due:
            from legacy_agent_guard_v1 import pool_prepare
            pool_prepare(job, automatic_controller, epoch, checkpoint, current_review, config)
        decision, packet = decide(job, config, variant, planet, epoch, current_review, action_history, candidates, previous_decision=(decisions[-1] if decisions else None), baseline_source_f1=baseline_source_f1, decision_history=decisions)
        decision = resolve_experiment(decision, checkpoint, current_review)
        guard_minimum_training(job,epoch,decision["continue_training"],source_collapsed=bool(source_f1 < max(SOURCE_COLLAPSE_ABSOLUTE_F1, baseline_source_f1-SOURCE_COLLAPSE_DROP_FROM_BASELINE)))
        if job.get("research_stopselect_policy") == "trend_judgment_v4":
            proposed_window = ((decision.get("current_state_assessment") or {})
                               .get("candidate_checkpoint_window"))
            if proposed_window:
                job["research_candidate_window"] = [int(value) for value in proposed_window]
                atomic(research_window_path, {
                    "candidate_checkpoint_window": job["research_candidate_window"],
                    "source_review_epoch": int(epoch),
                    "source_decision_sha256": hashlib.sha256(
                        json.dumps(decision, sort_keys=True).encode()
                    ).hexdigest(),
                    "label_free": True,
                })
        # This new action is taken after observing the current checkpoint;
        # its after value will be filled by the next checkpoint review.
        before_fraction = current_fraction
        action = {"decision_epoch": epoch, "action": decision["action"], "changed_parameters": decision["intervention"]["changed_parameters"], "schedule_updates": decision["intervention"].get("schedule_updates", []), "model_fraction_before": before_fraction, "model_fraction_after": None,
                  "gap_before": abs(before_fraction - planet["class2_fraction"]) if job.get("planet_decision_reference", True) else None,
                  "score": None, "improved": None}
        action_history.append(action); decisions.append(decision); apply_config(config, decision, model_opt, local_opt, global_opt)
        atomic(out / "agent_decisions.json", decisions); atomic(out / "action_scores.json", action_history)
        if not decision["continue_training"]:
            if experimental_gate:
                experimental_pending.unlink()
            job["stopped_after_epoch"] = epoch; break
        if refresh_due:
            from legacy_agent_guard_v1 import pool_refresh
            pool_refresh(job, automatic_controller, epoch, checkpoint, current_review, config)
        if experimental_gate:
            experimental_pending.unlink()
    if not candidates: raise RuntimeError("no review candidates")
    job["stopped_after_epoch"] = int(candidates[-1]["epoch"]); selection, selection_packet = decide(job, config, variant, planet, "selection", None, action_history, candidates, previous_decision=(decisions[-1] if decisions else None), decision_history=decisions)
    selection_packet = selection_metadata_packet(
        selection_packet,
        deterministic_fixed_schedule=bool(job.get("experimental_fixed_schedule")),
    )
    selected_epoch = int(selection["selected_epoch"]); selected = next(x for x in candidates if int(x["epoch"]) == selected_epoch); payload = torch_load(Path(selected["checkpoint"]), "cpu")
    stop_epoch = int(job["stopped_after_epoch"])
    stop_reason = "policy_stop"
    stop_f1 = float(candidates[-1].get("source_class2_f1", float("nan")))
    collapse_floor = max(SOURCE_COLLAPSE_ABSOLUTE_F1, baseline_source_f1 - SOURCE_COLLAPSE_DROP_FROM_BASELINE)
    if not np.isfinite(stop_f1) or stop_f1 < collapse_floor:
        stop_reason = "source_collapse_safety_exit"
    elif stop_epoch >= max_epoch_for_job(job):
        stop_reason = "budget_exit"
    final = out / "final"; final.mkdir(parents=True, exist_ok=True); atomic_torch_save(final / "selected_checkpoint.pt", {"epoch": selected_epoch, "state_dict": payload["state_dict"], "config": payload["config"], "metrics": payload.get("metrics", {}), "variant": variant, "planet_used_in_training": bool(supplementary_names), "planet_used_as_logit_prior": False, "supplementary_segmentation_only": bool(supplementary_names), "planet_used_for_agent_risk_only": bool(job.get("planet_decision_reference", True))})
    stop_decisions = [item.get("stop_decision") for item in decisions
                      if item.get("decision_kind") in {"training_review", "experiment_review"}]
    if stop_reason in {"policy_stop", "budget_exit", "source_collapse_safety_exit"}:
        early_stop_decision = "STOP"
    else:
        early_stop_decision = stop_decisions[-1] if stop_decisions else None
    candidate_window = selection_packet.get("candidate_checkpoint_window", [int(item["epoch"]) for item in candidates])
    candidate_records = selection_packet.get("candidate_checkpoint_records", [])
    atomic(final / "selection.json", {
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str((final / 'selected_checkpoint.pt').resolve()),
        # Early stopping and checkpoint selection are separate state machines.
        # A selection turn must never be interpreted as a policy STOP.
        "early_stop_decision": early_stop_decision,
        "selection_decision": "SELECT_CHECKPOINT",
        # Kept as a compatibility alias; consumers should use the explicit
        # early_stop_decision/selection_decision fields above.
        "stop_decision": early_stop_decision,
        "stop_epoch": stop_epoch,
        "stop_reason": stop_reason,
        "candidate_checkpoint_window": candidate_window,
        "candidate_checkpoint_records": candidate_records,
        "candidate_window_provenance": selection_packet.get("candidate_window_provenance"),
        "candidate_window_artifact": (str(research_window_path.resolve())
                                      if job.get("research_stopselect_policy") == "trend_judgment_v4"
                                      and research_window_path.is_file() else None),
        "selection": selection,
        "source_profile": job.get("nas_safe_io", {}).get("original_pretrained", job["source"]["pretrained"]),
        "planet_risk": planet,
        "action_scores": action_history,
    })
    delivery = delivery_record(
        selected_epoch=selected_epoch, selected_checkpoint=final / "selected_checkpoint.pt",
        source_collapsed=stop_reason == "source_collapse_safety_exit", stop_reason=stop_reason,
        stop_epoch=stop_epoch, original_max_epoch=max_epoch_for_job(job),
        recovery_policy=job.get("recovery_policy"),
    )
    atomic(final / "delivery.json", delivery)
    atomic(out / "status.json", {"job_id": job["job_id"], "run_id": job["run_id"], "state": "training_complete", "selected_epoch": selected_epoch, "stop_epoch": stop_epoch, "stop_reason": stop_reason, "gpu": device_name,
                               "metrics": copy.deepcopy(candidates[-1].get("review", {}).get("metrics", {})),
                               "metrics_epoch": stop_epoch,
                               "selected_negative_controls": copy.deepcopy(payload.get("metrics", {}).get("negative_controls", {})),
                               "delivery_kind": delivery["delivery_kind"], "adapted": delivery["adapted"],
                               "recovery_state": delivery["recovery"]["state"]})
    print(json.dumps({"state": "training_complete", "run_id": job["run_id"], "selected_epoch": selected_epoch}, ensure_ascii=False))


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit('Python optimization disables required training validation; run without -O or PYTHONOPTIMIZE')
    parser = __import__("argparse").ArgumentParser(); parser.add_argument("--job", type=Path, required=True); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--resume-existing", action="store_true"); args = parser.parse_args()
    try:
        main(args.job, args.device, resume_existing=args.resume_existing)
    except Exception as exc:
        try:
            failed_job = json.loads(args.job.read_text())
            status_path = Path(failed_job["output_root"]) / "status.json"
            previous = json.loads(status_path.read_text()) if status_path.is_file() else {}
            atomic(status_path, {**previous, "job_id": failed_job["job_id"], "state": "failed",
                                "stop_reason": "infrastructure_failure", "gpu": args.device,
                                "error_type": type(exc).__name__, "error": str(exc)})
        finally:
            raise
