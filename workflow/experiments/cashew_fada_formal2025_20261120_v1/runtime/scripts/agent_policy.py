#!/usr/bin/env python3
"""Pure policy, bounds, and validation for persistent Luna FADA control."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from protocol import (
    ANCHOR_CONFIG,
    MAX_EPOCH,
    MIN_EARLY_STOP_EPOCH,
    PARAMETER_BOUNDS,
    PROTOCOL_VERSION,
    REVIEW_INTERVAL,
    candidate_score,
    target_distribution_valid,
)


AGENT_SCHEMA_VERSION = "cashew-fada-persistent-luna-agent-decision-v1"
AGENT_LLM_SCHEMA_VERSION = "cashew-fada-persistent-luna-semantic-decision-v1"
AGENT_MODEL = "gpt-5.6-luna"
AGENT_REASONING_EFFORT = "high"
AGENT_SANDBOX = "read-only"
AGENT_SKILL_NAME = "cashew-fada-agent-controller"
MIN_LAMBDA_ADV = float(PARAMETER_BOUNDS["lambda_adv"][0])
MAX_LAMBDA_ADV = float(PARAMETER_BOUNDS["lambda_adv"][1])
SELECTION_ACTION = "SELECT_CHECKPOINT"
PRIMARY_PARAMETERS = {
    "lambda_adv", "main_lr", "D_lr", "D_steps_per_G_step", "temperature",
    "confidence_cap", "main_weight_decay", "D_weight_decay", "prior_strength",
}
ACTION_SET = {"ADJUST", "KEEP", "PAUSE_ADV", "RESUME_ADV", "CONTINUE", "STOP", SELECTION_ACTION}
DECISION_FIELDS = {
    "schema_version", "decision_kind", "request_sha256", "epoch", "action",
    "continue_training", "selected_epoch", "reason_codes", "analysis_summary",
    "planet_used", "previous_hypothesis_review", "current_state_assessment",
    "intervention", "next_stage_expectation",
}
SEMANTIC_DECISION_FIELDS = {
    "action", "continue_training", "selected_epoch", "reason_codes", "analysis_summary",
    "previous_hypothesis_review", "current_state_assessment", "intervention",
    "next_stage_expectation",
}
SEMANTIC_INTERVENTION_FIELDS = {"adversarial_state_after", "parameter_updates", "emergency", "rationale"}
OPTIMIZER_PARAMETERS = {
    "model_adam_beta1", "model_adam_beta2", "model_adam_eps",
    "D_adam_beta1", "D_adam_beta2", "D_adam_eps",
}


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _contract() -> dict[str, Any]:
    return {
        "model": AGENT_MODEL,
        "reasoning_effort": AGENT_REASONING_EFFORT,
        "sandbox": AGENT_SANDBOX,
        "persistent_thread": True,
        "fresh_ephemeral_session": False,
        "skill_name": AGENT_SKILL_NAME,
        "review_interval": REVIEW_INTERVAL,
        "tools_required": False,
    }


def _flat_config(config: dict[str, Any]) -> dict[str, Any]:
    result = {key: config.get(key, ANCHOR_CONFIG[key]) for key in PRIMARY_PARAMETERS}
    knowledge = config.get("class_knowledge", {})
    result["temperature"] = knowledge.get("temperature", result["temperature"])
    result["confidence_cap"] = knowledge.get("confidence_cap", result["confidence_cap"])
    result["adversarial_state"] = config.get("adversarial_state", "active")
    for key in ("model_adam_beta1", "model_adam_beta2", "model_adam_eps", "D_adam_beta1", "D_adam_beta2", "D_adam_eps"):
        if key in config:
            result[key] = config[key]
    return result


def parameter_payload(config: dict[str, Any]) -> dict[str, Any]:
    values = _flat_config(config)
    return dict(values)


def _validate_parameter_envelope(parameters: dict[str, Any], *, initialization: bool = False) -> None:
    optimizer_parameters = {"model_adam_beta1", "model_adam_beta2", "D_adam_beta1", "D_adam_beta2", "model_adam_eps", "D_adam_eps"}
    required = set(PRIMARY_PARAMETERS) | optimizer_parameters | {"adversarial_state"}
    missing = sorted(required - set(parameters))
    if missing:
        raise ValueError({"missing_parameters_after": missing})
    if parameters["adversarial_state"] not in {"active", "paused"}:
        raise ValueError("adversarial_state must be active or paused")
    for key in PRIMARY_PARAMETERS:
        value = parameters[key]
        if key == "D_steps_per_G_step":
            if isinstance(value, bool) or int(value) != value:
                raise ValueError("D_steps_per_G_step must be an integer")
            value = int(value)
        else:
            value = _finite(value, key)
        lower, upper = PARAMETER_BOUNDS[key]
        if not lower <= value <= upper:
            raise ValueError({"parameter_outside_envelope": key, "value": value, "bounds": [lower, upper]})
    for key in optimizer_parameters:
        lower, upper = PARAMETER_BOUNDS[key]
        value = _finite(parameters[key], key)
        if not lower <= value <= upper:
            raise ValueError({"parameter_outside_envelope": key, "value": value, "bounds": [lower, upper]})


def _changed(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    return [key for key in sorted(PRIMARY_PARAMETERS) if old.get(key) != new.get(key)]


def normalize_semantic_decision(request: dict[str, Any], semantic: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    """Turn the small LLM response into the audited, derived decision record.

    The model supplies an action and a list of requested parameter patches.  It
    never supplies request binding, a complete after-image, or a changed list;
    all three are derived here from the request's before-image.
    """
    if not isinstance(semantic, dict) or set(semantic) != SEMANTIC_DECISION_FIELDS:
        raise ValueError({"semantic_decision_fields_mismatch": sorted(set(semantic) if isinstance(semantic, dict) else []), "required": sorted(SEMANTIC_DECISION_FIELDS)})
    intervention = semantic["intervention"]
    if not isinstance(intervention, dict) or set(intervention) != SEMANTIC_INTERVENTION_FIELDS:
        raise ValueError({"semantic_intervention_fields_mismatch": sorted(set(intervention) if isinstance(intervention, dict) else []), "required": sorted(SEMANTIC_INTERVENTION_FIELDS)})
    if semantic["action"] not in ACTION_SET:
        raise ValueError("invalid Agent action")
    if not isinstance(semantic["continue_training"], bool):
        raise ValueError("continue_training must be boolean")
    if semantic["action"] == "STOP" and semantic["continue_training"] or semantic["action"] != "STOP" and semantic["action"] != SELECTION_ACTION and not semantic["continue_training"]:
        raise ValueError("action/continue_training mismatch")
    updates = intervention["parameter_updates"]
    if not isinstance(updates, list) or len(updates) > len(PRIMARY_PARAMETERS) + len(OPTIMIZER_PARAMETERS):
        raise ValueError("parameter_updates must be a short list")
    update_map: dict[str, Any] = {}
    for update in updates:
        if not isinstance(update, dict) or set(update) != {"parameter", "value"}:
            raise ValueError("each parameter update must contain only parameter and value")
        key = update["parameter"]
        if key not in PRIMARY_PARAMETERS | OPTIMIZER_PARAMETERS:
            raise ValueError({"parameter_update_not_allowed": key})
        if key in update_map:
            raise ValueError({"duplicate_parameter_update": key})
        update_map[key] = update["value"]
    kind = request["decision_kind"]
    if kind == "checkpoint_selection" and update_map:
        raise ValueError("checkpoint selection cannot contain parameter updates")
    if kind == "initialization" and "lambda_adv" not in update_map:
        raise ValueError("initialization must explicitly select lambda_adv from the pre-run gradient evidence")
    epoch = 0 if kind == "initialization" else int(request.get("current_review", {}).get("epoch", request.get("epoch", 0)))
    if kind == "initialization":
        before = parameter_payload(request["historical_anchor_config"])
    else:
        before = parameter_payload(request.get("current_config", {}))
    after = copy.deepcopy(before)
    after.update(update_map)
    after["adversarial_state"] = intervention["adversarial_state_after"]
    canonical = {
        "schema_version": AGENT_SCHEMA_VERSION,
        "decision_kind": kind,
        "request_sha256": request_sha256,
        "epoch": epoch,
        "action": semantic["action"],
        "continue_training": semantic["continue_training"],
        "selected_epoch": semantic["selected_epoch"],
        "reason_codes": copy.deepcopy(semantic["reason_codes"]),
        "analysis_summary": semantic["analysis_summary"],
        "planet_used": False,
        "previous_hypothesis_review": copy.deepcopy(semantic["previous_hypothesis_review"]),
        "current_state_assessment": copy.deepcopy(semantic["current_state_assessment"]),
        "intervention": {
            "continue_training": semantic["continue_training"],
            "adversarial_state_after": intervention["adversarial_state_after"],
            "parameters_after": after,
            "changed_parameters": _changed(before, after),
            "emergency": bool(intervention["emergency"]),
            "rationale": intervention["rationale"],
        },
        "next_stage_expectation": copy.deepcopy(semantic["next_stage_expectation"]),
    }
    return canonical


def _validate_step_rules(request: dict[str, Any], decision: dict[str, Any], *, initialization: bool = False) -> None:
    if initialization:
        return
    before = parameter_payload(request["current_config"])
    after = decision["intervention"]["parameters_after"]
    for locked in ("model_adam_beta1", "model_adam_beta2", "model_adam_eps", "D_adam_beta1", "D_adam_beta2", "D_adam_eps"):
        if after[locked] != before[locked]:
            raise ValueError(f"{locked} is initialization-only and cannot change during a normal review")
    changed = _changed(before, after)
    if decision["intervention"].get("changed_parameters") != changed:
        raise ValueError({"changed_parameters_mismatch": [changed, decision["intervention"].get("changed_parameters")]})
    if not decision["intervention"].get("emergency", False) and len(changed) > 2:
        raise ValueError({"too_many_primary_parameter_changes": changed})
    for key in changed:
        old, new = float(before[key]), float(after[key])
        if key == "lambda_adv":
            ratio = new / old
            if decision["intervention"].get("emergency", False):
                if ratio > 1.0 or ratio < 0.25:
                    raise ValueError("emergency lambda change must be a reduction no larger than x0.25")
            elif ratio < 0.5 or ratio > 1.25:
                raise ValueError("lambda change must be within x0.5..x1.25")
        elif key == "main_lr" and not 0.5 <= new / old <= 1.25:
            raise ValueError("main_lr change must be within x0.5..x1.25")
        elif key == "D_lr" and not 0.5 <= new / old <= 1.5:
            raise ValueError("D_lr change must be within x0.5..x1.5")
        elif key == "D_steps_per_G_step" and abs(int(new) - int(old)) > 1:
            raise ValueError("D:G change must be at most one step")
        elif key == "temperature" and abs(new - old) > 0.3:
            raise ValueError("temperature change must be at most 0.3")
        elif key == "confidence_cap" and abs(new - old) > 0.03:
            raise ValueError("confidence_cap change must be at most 0.03")
        elif key in {"main_weight_decay", "D_weight_decay"} and not 0.5 <= new / max(old, 1e-12) <= 2.0:
            raise ValueError("weight decay change must be within x0.5..x2")
        elif key == "prior_strength" and abs(new - old) > 0.05:
            raise ValueError("prior_strength change must be at most 0.05")
    for slow in ("main_weight_decay", "D_weight_decay", "prior_strength"):
        if slow not in changed:
            continue
        last = [item.get("epoch") for item in request.get("decision_history", []) if slow in item.get("changed_parameters", [])]
        if last and int(request["current_review"]["epoch"]) - int(last[-1]) < 10:
            raise ValueError({"slow_parameter_cadence_violation": slow, "last_epoch": last[-1]})


def _validate_emergency_direction(before: dict[str, Any], after: dict[str, Any], changed: list[str]) -> None:
    """Emergency Agent edits may only reduce pressure or make guidance softer."""
    for key in changed:
        old, new = float(before[key]), float(after[key])
        if key in {"lambda_adv", "main_lr", "D_lr", "prior_strength", "D_steps_per_G_step"} and new > old:
            raise ValueError({"emergency_change_increases_pressure": key})
        if key == "temperature" and new < old:
            raise ValueError({"emergency_change_hardens_class_knowledge": key})
        if key == "confidence_cap" and new > old:
            raise ValueError({"emergency_change_increases_confidence_pressure": key})


def _review_summary(candidate: dict[str, Any], baseline_class2_f1: float) -> dict[str, Any]:
    review = candidate["review"]
    metrics = candidate["metrics"]
    challenger = candidate.get("challenger", {})
    source = review["source"]
    target = review["target_unsupervised"]
    current_d = challenger.get("current_D_test", {})
    fresh = challenger.get("fresh_global_mlp", {}).get("test", {})
    return {
        "epoch": int(candidate["epoch"]),
        "source": {"macro_f1": _finite(source["macro_f1"], "source macro F1"), "class2_f1": _finite(source["f1_per_class"][2], "source class2 F1"), "baseline_class2_f1": float(baseline_class2_f1)},
        "source_training": copy.deepcopy(review.get("source_training")),
        "target_unlabeled": {
            "snd": _finite(target["snd"], "target SND"),
            "entropy": _finite(target["entropy"], "target entropy"),
            "class_distribution": [
                _finite(x, "target class distribution")
                for x in target["class_distribution"]
            ],
            "class_distribution_kind": target.get(
                "class_distribution_kind", "legacy_unspecified"
            ),
            "predicted_class_counts": copy.deepcopy(target.get("predicted_class_counts")),
            "valid_pixel_count": target.get("pixel_count"),
            "patch_count": target.get("patch_count"),
            "inference_batch_size": target.get("inference_batch_size"),
            "scope": target.get("scope"),
            "mean_softmax_distribution": copy.deepcopy(
                target.get("mean_softmax_distribution")
            ),
        },
        "metric_provenance": copy.deepcopy(review.get("metric_provenance", {})),
        "domain_diagnostics": {"current_discriminator_accuracy": _finite(current_d.get("accuracy", 0.5), "current discriminator accuracy"), "current_discriminator_auc": _finite(current_d.get("auc", 0.5), "current discriminator AUC"), "fresh_global_probe_accuracy": _finite(fresh.get("accuracy", 0.5), "fresh global probe accuracy"), "fresh_global_probe_auc": _finite(fresh.get("auc", 0.5), "fresh global probe AUC")},
        "optimization": {"classification_loss": _finite(metrics["cls"], "classification loss"), "adversarial_loss": _finite(metrics["adv"], "adversarial loss"), "discriminator_loss": _finite(metrics["D_loss"], "discriminator loss"), "classification_gradient_l1": _finite(metrics["cls_grad"], "classification gradient"), "adversarial_gradient_l1": _finite(metrics["adv_grad"], "adversarial gradient"), "effective_adv_to_cls_gradient_ratio": _finite(metrics["effective_adv_to_cls_grad_ratio"], "gradient ratio"), "gradient_diagnostics": copy.deepcopy(review.get("gradient_diagnostics")), "adversarial_state": metrics.get("adversarial_state", "active"), "potential_resume_gradient_ratio": metrics.get("potential_resume_gradient_ratio")},
        "deterministic_checkpoint_score": candidate_score(candidate, baseline_class2_f1),
        "eligible_for_selection": bool(candidate.get("eligible_for_selection")),
    }


def review_safety_envelope(candidate: dict[str, Any], baseline_class2_f1: float, config: dict[str, Any], proposed_early_stop: dict[str, Any], safety_fuse_applied_this_epoch: bool) -> dict[str, Any]:
    epoch = int(candidate["epoch"])
    current = parameter_payload(config)
    ratio = _finite(candidate["metrics"]["effective_adv_to_cls_grad_ratio"], "gradient ratio")
    potential_resume_ratio = candidate.get("metrics", {}).get("potential_resume_gradient_ratio")
    potential_resume_ratio = None if potential_resume_ratio is None else _finite(potential_resume_ratio, "potential resume gradient ratio")
    source_protection = _finite(candidate["review"]["source"]["f1_per_class"][2], "source class2 F1") < float(baseline_class2_f1) - 0.05
    target_valid = target_distribution_valid(candidate["review"])
    gradient_unsafe = ratio > 0.5
    plateau_gates = dict(proposed_early_stop.get("last_gates") or {})
    plateau = bool(plateau_gates and all(plateau_gates.values()))
    emergency = source_protection or gradient_unsafe or not target_valid
    stop_allowed = epoch == MAX_EPOCH or plateau or emergency
    allowed = ["KEEP", "ADJUST", "CONTINUE", "PAUSE_ADV", "RESUME_ADV"]
    if current["adversarial_state"] == "paused":
        allowed = ["KEEP", "ADJUST", "RESUME_ADV", "CONTINUE"]
        if potential_resume_ratio is not None and potential_resume_ratio >= 0.30:
            allowed.remove("RESUME_ADV")
    if emergency:
        allowed = ["ADJUST", "PAUSE_ADV"]
    if stop_allowed:
        allowed.append("STOP")
    if epoch == MAX_EPOCH:
        allowed = ["STOP"]
    return {"allowed_actions": allowed, "parameter_bounds": copy.deepcopy(PARAMETER_BOUNDS), "current_parameters": current, "green_ratio_lt": 0.30, "yellow_ratio": [0.30, 0.50], "red_ratio_gt": 0.50, "source_protection_fired": source_protection, "gradient_ratio_unsafe": gradient_unsafe, "potential_resume_gradient_ratio": potential_resume_ratio, "resume_adversarial_pressure_allowed": potential_resume_ratio is None or potential_resume_ratio < 0.30, "target_distribution_valid": target_valid, "plateau_stop_eligible": plateau, "per_epoch_safety_fuse_already_applied": bool(safety_fuse_applied_this_epoch), "terminal_epoch": epoch == MAX_EPOCH, "minimum_plateau_stop_epoch": MIN_EARLY_STOP_EPOCH, "stop_allowed": stop_allowed, "emergency_stop_allowed": emergency, "lambda_increase_allowed": bool(ratio < 0.30 and not source_protection and current["adversarial_state"] == "active")}


def _common_request(job_id: str, job_manifest_sha256: str, prior_lock_sha256: str, kind: str, epoch: int) -> dict[str, Any]:
    return {"schema_version": AGENT_SCHEMA_VERSION, "decision_kind": kind, "protocol_version": PROTOCOL_VERSION, "agent_contract": _contract(), "job": {"job_id": str(job_id), "job_manifest_sha256": str(job_manifest_sha256), "prior_lock_sha256_only": str(prior_lock_sha256)}, "class_contract": {"0": "crop_others", "1": "mixed_trees_grassland", "2": "cashew", "3": "built_up"}, "review_interval": REVIEW_INTERVAL, "epoch": int(epoch), "information_boundary": {"planet_inputs_exposed": False, "planet_values_exposed": False, "prior_contents_exposed": False, "prior_hash_is_provenance_only": True, "target_labels_exposed": False, "allowed_evidence": "source validation plus unlabeled target diagnostics only"}, "instruction": "Do not browse the web. Do not perform literature search. The controller policy has already been supplied by $cashew-fada-agent-controller. Use the structured state and the Skill only. Return only schema-valid structured output."}


def build_initialization_request(*, job_id: str, job_manifest_sha256: str, prior_lock_sha256: str, pre_run_report: dict[str, Any], anchor_config: dict[str, Any]) -> dict[str, Any]:
    request = _common_request(job_id, job_manifest_sha256, prior_lock_sha256, "initialization", 0)
    request.update({"previous_hypothesis": "not_applicable", "previous_decision_sha256": None, "pre_run": copy.deepcopy(pre_run_report), "historical_anchor_config": parameter_payload(anchor_config), "allowed_parameter_envelopes": copy.deepcopy(PARAMETER_BOUNDS), "initialization_constraints": {"lambda_adv_must_be_explicitly_selected": True, "lambda_adv_source": "pre_run.gradient_probe.lambda_candidates"}, "allowed_actions": ["ADJUST", "KEEP", "CONTINUE"], "fixed_controls": {"batch_size": 16, "seed": 16, "max_epoch": MAX_EPOCH, "review_interval": REVIEW_INTERVAL}})
    return request


def build_review_request(*, job_id: str, job_manifest_sha256: str, prior_lock_sha256: str, checkpoint_sha256: str, candidate: dict[str, Any], baseline_class2_f1: float, config: dict[str, Any], proposed_early_stop: dict[str, Any], prior_reviews: list[dict[str, Any]], prior_decisions: list[dict[str, Any]], safety_fuse_applied_this_epoch: bool, previous_decision_sha256: str | None = None, previous_expectation: dict[str, Any] | None = None) -> dict[str, Any]:
    request = _common_request(job_id, job_manifest_sha256, prior_lock_sha256, "training_review", int(candidate["epoch"]))
    history = [_review_summary(item, baseline_class2_f1) for item in prior_reviews if int(item.get("epoch", 0)) != 0 and item.get("metrics") is not None]
    decisions = [{"epoch": int(item.get("epoch", 0)), "action": item.get("action"), "changed_parameters": list(item.get("changed_parameters") or []), "continue_training": bool(item.get("continue_training", True)), "parameters_after": copy.deepcopy(item.get("parameters_after"))} for item in prior_decisions if item.get("decision_kind") in {"training_review", "initialization"}]
    request.update({"job": {**request["job"], "training_checkpoint_sha256": str(checkpoint_sha256)}, "current_config": parameter_payload(config), "baseline": {"source_class2_f1": float(baseline_class2_f1)}, "current_review": _review_summary(candidate, baseline_class2_f1), "trend_5_epoch": history[-2:], "trend_since_previous_review": history[-1:], "previous_agent_decision": decisions[-1] if decisions else None, "previous_decision_sha256": previous_decision_sha256, "previous_expectation": copy.deepcopy(previous_expectation), "recent_safety_events": [item for item in prior_decisions if "safety" in str(item.get("action", "")).lower() or "FUSE" in str(item.get("action", ""))], "candidate": {"checkpoint_sha256": candidate["checkpoint_sha256"], "eligible_for_selection": bool(candidate.get("eligible_for_selection"))}, "safety_envelope": review_safety_envelope(candidate, baseline_class2_f1, config, proposed_early_stop, safety_fuse_applied_this_epoch), "allowed_parameter_envelopes": copy.deepcopy(PARAMETER_BOUNDS), "fixed_controls": {"batch_size": 16, "seed": 16, "max_epoch": MAX_EPOCH, "review_interval": REVIEW_INTERVAL, "architecture_immutable": True}})
    request["allowed_actions"] = request["safety_envelope"]["allowed_actions"]
    return request


def build_selection_request(*, job_id: str, job_manifest_sha256: str, prior_lock_sha256: str, stopped_after_epoch: int, stop_reason: str, candidates: list[dict[str, Any]], baseline_class2_f1: float, decision_history: list[dict[str, Any]], current_config: dict[str, Any] | None = None, previous_decision_sha256: str | None = None) -> dict[str, Any]:
    eligible = [item for item in candidates if bool(item.get("eligible_for_selection"))]
    if not eligible:
        raise ValueError("checkpoint selection requires at least one hard-safe candidate")
    summaries = []
    for item in eligible:
        summary = {"epoch": int(item["epoch"]), "checkpoint_sha256": item["checkpoint_sha256"], "safe_pretrained_fallback": int(item["epoch"]) == 0}
        if int(item["epoch"]) == 0:
            summary["source"] = {"macro_f1": _finite(item["review"]["source"]["macro_f1"], "source macro F1"), "class2_f1": _finite(item["review"]["source"]["f1_per_class"][2], "source class2 F1")}
        else:
            summary.update(_review_summary(item, baseline_class2_f1))
        summaries.append(summary)
    request = _common_request(job_id, job_manifest_sha256, prior_lock_sha256, "checkpoint_selection", int(stopped_after_epoch))
    request.update({"job": {**request["job"], "stopped_after_epoch": int(stopped_after_epoch), "stop_reason": str(stop_reason)}, "current_config": parameter_payload(current_config or ANCHOR_CONFIG), "eligible_candidates": summaries, "decision_history": [{"epoch": int(x.get("epoch", 0)), "action": x.get("action"), "changed_parameters": list(x.get("changed_parameters") or [])} for x in decision_history], "previous_decision_sha256": previous_decision_sha256, "selection_constraints": {"selected_epoch_must_be_listed": True, "epoch0_is_safe_fallback": True, "deterministic_score_is_advisory_not_an_automatic_argmax": True, "model_weights_cannot_be_modified": True}, "allowed_actions": [SELECTION_ACTION]})
    request["instruction"] = "Remain in this same persistent thread under $cashew-fada-agent-controller. Select one listed hard-safe checkpoint using longitudinal source and unlabeled-target evidence. Planet evidence must not rank checkpoints. Use action SELECT_CHECKPOINT and selected_epoch from the listed candidates."
    return request


def validate_decision_shape(decision: dict[str, Any]) -> None:
    if not isinstance(decision, dict) or set(decision) != DECISION_FIELDS:
        raise ValueError({"decision_fields_mismatch": sorted(set(decision) if isinstance(decision, dict) else []), "required": sorted(DECISION_FIELDS)})
    if decision["schema_version"] != AGENT_SCHEMA_VERSION or decision["action"] not in ACTION_SET:
        raise ValueError("invalid Agent schema or action")
    if decision["decision_kind"] not in {"initialization", "training_review", "checkpoint_selection"}:
        raise ValueError("invalid Agent decision kind")
    if not isinstance(decision["epoch"], int) or isinstance(decision["epoch"], bool) or not 0 <= decision["epoch"] <= MAX_EPOCH:
        raise ValueError("invalid decision epoch")
    if not isinstance(decision["continue_training"], bool) or decision["planet_used"] is not False:
        raise ValueError("invalid continue_training or Planet usage")
    if not isinstance(decision["reason_codes"], list) or not 1 <= len(decision["reason_codes"]) <= 12 or any(not isinstance(x, str) for x in decision["reason_codes"]):
        raise ValueError("invalid reason_codes")
    if not isinstance(decision["analysis_summary"], str) or not 1 <= len(decision["analysis_summary"]) <= 2000:
        raise ValueError("invalid analysis_summary")
    for key in ("previous_hypothesis_review", "current_state_assessment", "intervention", "next_stage_expectation"):
        if not isinstance(decision[key], dict):
            raise ValueError(f"missing mandatory reasoning record: {key}")
    if not isinstance(decision["intervention"].get("parameters_after"), dict) or not isinstance(decision["intervention"].get("changed_parameters"), list):
        raise ValueError("invalid intervention record")


def validate_initialization_decision(request: dict[str, Any], decision: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    validate_decision_shape(decision)
    if decision["decision_kind"] != "initialization" or decision["request_sha256"] != request_sha256 or decision["epoch"] != 0 or decision["action"] not in request["allowed_actions"] or not decision["continue_training"]:
        raise ValueError("initialization decision is not bound to its request")
    _validate_parameter_envelope(decision["intervention"]["parameters_after"], initialization=True)
    if decision["previous_hypothesis_review"].get("status") != "not_applicable":
        raise ValueError("initialization previous_hypothesis_review must be not_applicable")
    if decision["selected_epoch"] is not None:
        raise ValueError("initialization cannot select a checkpoint")
    if decision["intervention"].get("emergency"):
        raise ValueError("initialization cannot be marked emergency")
    return copy.deepcopy(decision)


def validate_review_decision(request: dict[str, Any], decision: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    validate_decision_shape(decision)
    epoch = int(request["current_review"]["epoch"])
    if decision["decision_kind"] != "training_review" or decision["request_sha256"] != request_sha256 or decision["epoch"] != epoch:
        raise ValueError("Agent review decision is not bound to this request/epoch")
    if decision["action"] not in request["allowed_actions"]:
        raise ValueError({"action_outside_safety_envelope": decision["action"], "allowed": request["allowed_actions"]})
    if decision["selected_epoch"] is not None:
        raise ValueError("training review cannot select a checkpoint")
    if decision["action"] == "STOP" and decision["continue_training"] or decision["action"] != "STOP" and not decision["continue_training"]:
        raise ValueError("action/continue_training mismatch")
    if epoch < MIN_EARLY_STOP_EPOCH and decision["action"] == "STOP" and not request["safety_envelope"].get("emergency_stop_allowed"):
        raise ValueError("routine STOP before minimum epoch is forbidden")
    _validate_parameter_envelope(decision["intervention"]["parameters_after"])
    _validate_step_rules(request, decision)
    envelope = request["safety_envelope"]
    before_config = parameter_payload(request["current_config"])
    after_config = decision["intervention"]["parameters_after"]
    changed = _changed(before_config, after_config)
    envelope_emergency = bool(
        envelope.get("gradient_ratio_unsafe")
        or envelope.get("source_protection_fired")
        or not envelope.get("target_distribution_valid", True)
    )
    if envelope_emergency != bool(decision["intervention"].get("emergency")):
        raise ValueError("emergency flag must match the deterministic safety envelope")
    if not envelope.get("lambda_increase_allowed", True) and after_config["lambda_adv"] > before_config["lambda_adv"]:
        raise ValueError("safety envelope forbids lambda increase")
    if envelope_emergency:
        _validate_emergency_direction(before_config, after_config, changed)
    if decision["action"] == "PAUSE_ADV" and after_config["adversarial_state"] != "paused":
        raise ValueError("PAUSE_ADV must leave adversarial_state paused")
    if decision["action"] == "RESUME_ADV" and after_config["adversarial_state"] != "active":
        raise ValueError("RESUME_ADV must leave adversarial_state active")
    return copy.deepcopy(decision)


def validate_selection_decision(request: dict[str, Any], decision: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    validate_decision_shape(decision)
    if decision["decision_kind"] != "checkpoint_selection" or decision["request_sha256"] != request_sha256 or decision["epoch"] != int(request["job"]["stopped_after_epoch"]):
        raise ValueError("selection decision is not bound to this request/run")
    if decision["action"] != SELECTION_ACTION or decision["continue_training"] or decision["intervention"]["continue_training"] or decision["intervention"]["changed_parameters"]:
        raise ValueError("checkpoint selection must only select and stop")
    _validate_parameter_envelope(decision["intervention"]["parameters_after"])
    selected = decision["selected_epoch"]
    eligible = {int(item["epoch"]) for item in request["eligible_candidates"]}
    if not isinstance(selected, int) or isinstance(selected, bool) or selected not in eligible:
        raise ValueError({"selected_epoch_not_eligible": selected, "eligible": sorted(eligible)})
    return copy.deepcopy(decision)


def validate_agent_provenance(*, request_path: Path, decision_path: Path, provenance_path: Path, schema_path: Path, sha256_file, expected_previous_decision_sha256: str | None = None) -> dict[str, Any]:
    import json
    provenance = json.loads(provenance_path.read_text())
    checks = {"request_path": provenance.get("request") == str(request_path.resolve()), "request_hash": provenance.get("request_sha256") == sha256_file(request_path), "decision_path": provenance.get("decision") == str(decision_path.resolve()), "decision_hash": provenance.get("decision_sha256") == sha256_file(decision_path), "schema_hash": provenance.get("output_schema_sha256") == sha256_file(schema_path), "model": provenance.get("model") == AGENT_MODEL, "reasoning_effort": provenance.get("reasoning_effort") == AGENT_REASONING_EFFORT, "sandbox": provenance.get("sandbox") == AGENT_SANDBOX, "persistent_thread": provenance.get("persistent_thread") is True, "fresh_ephemeral_session": provenance.get("fresh_ephemeral_session") is False, "skill_name": provenance.get("skill_name") == AGENT_SKILL_NAME, "codex_exit_code": provenance.get("codex_exit_code") == 0, "planet_inputs_exposed": provenance.get("planet_inputs_exposed") is False, "target_labels_exposed": provenance.get("target_labels_exposed") is False}
    if expected_previous_decision_sha256 is not None:
        checks["previous_decision_hash"] = provenance.get("previous_decision_sha256") == expected_previous_decision_sha256
    if not all(checks.values()):
        raise ValueError({"invalid_agent_provenance": str(provenance_path), "checks": checks})
    return provenance


def apply_agent_training_decision(config: dict[str, Any], model_optimizer, d_optimizer, decision: dict[str, Any], epoch: int) -> dict[str, Any]:
    before = parameter_payload(config)
    after = copy.deepcopy(decision["intervention"]["parameters_after"])
    _validate_parameter_envelope(after)
    for key in PRIMARY_PARAMETERS:
        config[key] = after[key]
    for key in ("model_adam_beta1", "model_adam_beta2", "model_adam_eps", "D_adam_beta1", "D_adam_beta2", "D_adam_eps"):
        config[key] = after[key]
    config["class_knowledge"]["temperature"] = after["temperature"]
    config["class_knowledge"]["confidence_cap"] = after["confidence_cap"]
    config["adversarial_state"] = after["adversarial_state"]
    if model_optimizer is not None:
        for group in model_optimizer.param_groups:
            # A3's zero-initialized spatial blocks train at a fixed multiplier
            # of the Agent-controlled base LR; legacy one-group jobs retain 1.
            group["lr"] = float(config["main_lr"]) * float(group.get("lr_multiplier", 1.0))
            group["weight_decay"] = float(config["main_weight_decay"])
            group["betas"] = (float(config["model_adam_beta1"]), float(config["model_adam_beta2"]))
            group["eps"] = float(config["model_adam_eps"])
    if d_optimizer is not None:
        for group in d_optimizer.param_groups:
            group["lr"] = float(config["D_lr"])
            group["weight_decay"] = float(config["D_weight_decay"])
            group["betas"] = (float(config["D_adam_beta1"]), float(config["D_adam_beta2"]))
            group["eps"] = float(config["D_adam_eps"])
    final = parameter_payload(config)
    return {"epoch": int(epoch), "before": before, "after": final, "changed_parameters": _changed(before, final), "adversarial_state": config["adversarial_state"]}
