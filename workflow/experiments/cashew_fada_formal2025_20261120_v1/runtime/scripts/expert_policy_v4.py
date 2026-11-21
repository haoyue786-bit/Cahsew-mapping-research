#!/usr/bin/env python3
"""Minimal semantic policy for the shared v4 expert thread."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from schedule_v4 import CURVES, default_state
from agentguide_policy_v1 import enabled as distilled_policy_enabled
from minimum_training_v1 import floor as minimum_training_floor
import supplementary_control_v4 as negative
import gradient_control_v1 as gradient_control
import cosine_finish_v1 as finish


# One durable gpt-6.1-sol/high thread per tile under the current formal template.
# Existing sessions from another configured model are archived by the dispatcher.
MODEL = "gpt-6.1-sol"
EFFORT = "high"
SANDBOX = "danger-full-access"
SKILL_NAME = "cashew-fada-training-expert"
MAX_EPOCH = 200
MINIMUM_STOP_EPOCH = 0
ALLOWED = {"ADJUST", "KEEP", "PAUSE_ADV", "RESUME_ADV", "CONTINUE", "STOP", "SELECT_CHECKPOINT", "REQUEST_EXPERIMENT"}
PARAM_BOUNDS = {
    "lambda_adv": [1e-4, 5e-2], "main_lr": [2e-7, 1e-5], "D_lr": [5e-7, 2e-5],
    "D_steps_per_G_step": [1, 4], "temperature": [1.2, 3.0], "confidence_cap": [0.8, 0.95],
    "main_weight_decay": [1e-4, 3e-2], "D_weight_decay": [0.0, 3e-2],
}
PARAMETERS = set(PARAM_BOUNDS)
SEMANTIC_FIELDS = {"action", "continue_training", "selected_epoch", "reason_codes", "analysis_summary", "previous_hypothesis_review", "current_state_assessment", "intervention", "experiment", "next_stage_expectation"}
INTERVENTION_FIELDS = {"adversarial_state_after", "parameter_updates", "schedule_choices", "schedule_updates", "emergency", "rationale"}


def derive_stop_decision(kind, action, continue_training, reason_codes=None,
                         current_state_assessment=None):
    """Keep stopping state independent from checkpoint selection state.

    STOP_CANDIDATE is a research annotation: it can preserve a shortlist while
    the worker continues for one informative review. The Agent semantic schema
    stays unchanged; the normalized decision is the durable boundary.
    """
    if kind == "checkpoint_selection":
        # Selecting a historical checkpoint is not an early-stop decision.
        # The worker records the two decisions in separate artifact fields.
        return None
    if action == "STOP" or not continue_training:
        return "STOP"
    reasons = {str(value).upper() for value in (reason_codes or [])}
    state = str((current_state_assessment or {}).get("state", "")).upper()
    if "STOP_CANDIDATE" in reasons or state in {"STOP_CANDIDATE", "PROVISIONAL_STOP"}:
        return "STOP_CANDIDATE"
    return "KEEP"


def apply_agent_early_stopping_policy(request):
    """Apply the formal stability gate or an explicitly versioned research rule."""
    result = copy.deepcopy(request)
    if result.get("decision_kind") not in {"training_review", "experiment_review"}:
        return result
    epoch = int(result.get("epoch", 0))
    safety = result.setdefault("safety_envelope", {})
    minimum_epoch = minimum_training_floor(result.get("job", {}))
    maximum_epoch = int(safety.get("max_epoch", MAX_EPOCH))
    review = result.get("current_review") or {}
    metrics = review.get("metrics") or {}
    stable = bool((review.get("target_region_stability") or {}).get("full_region_stable"))
    active = (result.get("current_config") or {}).get("adversarial_state") == "active"
    adv = metrics.get("adv")
    ratio = metrics.get("effective_gradient_ratio")
    measured = (isinstance(adv, (float, int)) and math.isfinite(float(adv)) and float(adv) > 0
                and isinstance(ratio, (float, int)) and math.isfinite(float(ratio)) and float(ratio) > 0)
    emergency = bool((result.get("source_protection_policy") or {}).get("source_collapsed"))
    research_policy = result.get("job", {}).get("research_stopselect_policy")
    flexible_research = research_policy in {"trend_judgment_v2", "trend_judgment_v3", "trend_judgment_v4"}
    stop_allowed = bool((stable and active and measured) or epoch >= maximum_epoch or emergency
                        or (flexible_research and epoch > 0))
    safety.update({
        "stop_allowed": bool(stop_allowed),
        "minimum_stop_epoch": minimum_epoch,
        "stop_decision_owner": ("agent_after_versioned_research_trend_judgment_or_safety_exit" if flexible_research else
                                "agent_after_verified_plateau_or_safety_exit"),
        "full_region_stability_required_for_stop": not (
            epoch >= maximum_epoch or emergency or flexible_research
        ),
        "stable_active_adversarial_window": bool(stable and active and measured),
    })
    actions = [action for action in result.get("allowed_actions", []) if action != "STOP"]
    if stop_allowed:
        actions.append("STOP")
    result["allowed_actions"] = actions
    result["early_stopping_policy"] = {
        "mode": (f"research_flexible_{research_policy}" if flexible_research else
                 "verified_plateau_with_budget_and_collapse_exits"),
        "minimum_epoch": minimum_epoch,
        "maximum_epoch": maximum_epoch,
        "full_region_stability_is_advisory": bool(flexible_research),
        "instruction": ((
            ("Research-only trend judgment v4 is active: judge whether another review has meaningful expected "
             "value, maintain a provisional candidate window, and distinguish STOP_CANDIDATE from STOP. "
             "A quiet segment need not be perfectly stable; coherent overlap decline with area erosion may "
             "justify a safety stop after considering one informative confirmation. STOP never selects the "
             "current checkpoint. Selection is a separate review over trained non-collapsed candidates. "
             "Do not use target labels, Planet area, source F1, SND, or one stability metric as a quality reward. "
             "This is not a formal production policy.") if research_policy == "trend_judgment_v4" else
            "Research-only trend judgment v2 is active: at any trained review the Agent may propose STOP "
            "when the recent class-specific spatial trajectory looks level, or stop for a documented safety "
            "exit. Judge area and shape together, distinguish a brief lull from a settled trend, and inspect "
            "source collapse and active measured adversarial learning. No fixed number of stable reviews is "
            "required. This is not a formal production policy; choose a trained checkpoint from the eligible "
            "set and explain uncertainty. Planet agreement alone is not a stop signal."
        ) if flexible_research else (
            "Normally STOP only after a complete stable full-region review window under active measured "
            "adversarial training. Stability is not accuracy: inspect losses, gradients and source collapse "
            "before deciding. At max_epoch STOP is allowed without stability; a measured catastrophic "
            "source collapse also permits a documented emergency STOP. Never stop on Planet agreement alone."
        )),
    }
    if distilled_policy_enabled(result.get("job", {})):
        normal = epoch >= max(1, minimum_epoch) and active and measured
        safety.update({"stop_allowed": bool(normal or epoch >= maximum_epoch or emergency),
                       "stop_decision_owner": "agent_after_marginal_review_value_judgment",
                       "full_region_stability_required_for_stop": False})
        actions = [a for a in result["allowed_actions"] if a != "STOP"]
        if safety["stop_allowed"]:
            actions.append("STOP")
        result["allowed_actions"] = actions
        result["early_stopping_policy"].update({
            "mode": "formal_marginal_review_value_v1", "full_region_stability_is_advisory": True,
            "instruction": "Judge whether the next review interval has meaningful expected value from repeated comparable losses, gradients, map movement and intervention response. A perfect four-review stability window is not required. Name two measured signals, the strongest contrary signal, what another review could resolve and a reversal condition. If a short confirmation could distinguish a lull from recovery, continue with STOP_CANDIDATE in reason_codes; do not add automatic fixed patience. Stop when further useful learning is poorly supported, or document a budget/collapse exit. Keep STOP separate from SELECT; target labels, Planet area, source F1 and single consistency metrics are not quality rewards. This is the user-authorized formal procedure; geographic quality and compute savings remain under evaluation."})
    if result.get('job', {}).get('evidence_based_stop_policy') == 'agent_judgment_v1':
        # Permission to consider STOP is not evidence for stopping. Preserve
        # the trained-epoch floor and let the Agent judge actual observations,
        # including whether paused or absent adversarial signals need further
        # diagnosis. Budget/collapse remain explicit exits below.
        stop_allowed = epoch >= max(1, minimum_epoch) or emergency
        safety.update({'stop_allowed': bool(stop_allowed),
                       'stop_decision_owner': 'agent_evidence_judgment_v1',
                       'full_region_stability_required_for_stop': False})
        result['allowed_actions'] = [a for a in result.get('allowed_actions', []) if a != 'STOP']
        if stop_allowed: result['allowed_actions'].append('STOP')
        result['early_stopping_policy'].update({
            'mode': 'agent_evidence_judgment_v1', 'full_region_stability_is_advisory': True,
            'instruction': 'STOP is available for a trained checkpoint; availability is not evidence to stop. '
                'Use actual observations, uncertainty, alternatives and expected value of another interval. '
                'Paused or unmeasured adversarial signals are not by themselves proof of convergence. '
                'STOP changes no controls; SELECT is separate. Never use sealed target-point results.'})
    if epoch < minimum_epoch and not emergency:
        safety['stop_allowed'] = False
        result['allowed_actions'] = [a for a in result['allowed_actions'] if a != 'STOP']
    result['early_stopping_policy']['instruction'] += (
        f" Explicit user horizon: ordinary STOP is forbidden before total epoch {minimum_epoch}. "
        "Continue useful permitted training/tuning until that floor. Missing accuracy evidence, weak marginal-value judgment, prior STOP and completion of an early cosine finish cannot override it. "
        "Only measured source collapse permits the existing safety exit. STOP and SELECT remain separate; all eligible trained checkpoints remain comparable after training."
    )
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_bounds(config):
    from transfer_parameter_policy import bounds
    from module_controls_v2 import bounds as module_bounds
    from transfer_regularization_v3 import bounds as regularization_bounds
    from hard_sample_training import bounds as annotation_bounds
    return {**PARAM_BOUNDS, **negative.bounds(config), **gradient_control.bounds(config), **finish.bounds(config), **bounds(config), **module_bounds(config), **regularization_bounds(config), **annotation_bounds(config)}


def schedule_curves(config):
    return dict(CURVES, **(negative.CURVES if negative.enabled(config) else {}))


def flat(config):
    result = {key: config[key] for key in parameter_bounds(config)}
    if config.get('module_control_policy'):
        result['module_control_policy'] = config['module_control_policy']
    if config.get('parameter_group_policy'):
        result['parameter_group_policy'] = copy.deepcopy(config['parameter_group_policy'])
    if gradient_control.enabled(config):
        result['gradient_control_policy'] = copy.deepcopy(config['gradient_control_policy'])
    if finish.enabled(config):
        result['cosine_finish_policy'] = copy.deepcopy(config['cosine_finish_policy'])
        if config.get('cosine_finish_state'):
            result['cosine_finish_state'] = copy.deepcopy(config['cosine_finish_state'])
    if negative.enabled(config):
        result["negative_control_policy"] = copy.deepcopy(config["negative_control_policy"])
        if config.get('polygon_supervision_policy'):
            result['polygon_supervision_policy']=copy.deepcopy(config['polygon_supervision_policy'])
        result["negative_pool_count"] = int(config.get("negative_pool_count", config.get("supplementary_training", {}).get("count", 0)))
        if config.get("conditional_negative_policy"):
            result["conditional_negative_policy"] = copy.deepcopy(config["conditional_negative_policy"])
    result["adversarial_state"] = config.get("adversarial_state", "active")
    return result


def _check_config(values):
    from module_controls_v2 import validate as validate_modules
    validate_modules(values)
    finish.validate(values)
    if gradient_control.enabled(values):
        gradient_control.strength(values)
    missing = sorted((set(parameter_bounds(values)) | {"adversarial_state"}) - set(values))
    if missing:
        raise ValueError({"missing_parameters": missing})
    if values["adversarial_state"] not in {"active", "paused"}:
        raise ValueError("invalid adversarial state")
    for key, bounds in parameter_bounds(values).items():
        if key in negative.BOUNDS and type(values[key]) not in (int, float):
            raise ValueError("negative controls must be real numeric values")
        value = float(values[key])
        if not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
            raise ValueError({"parameter_outside_bounds": [key, value, bounds]})
        if (key == "D_steps_per_G_step" or key in negative.INTEGER_CONTROLS) and (isinstance(values[key],bool) or int(value) != value):
            raise ValueError(key + " must be an integer")


def normalize(request, semantic, request_hash, *, planet_used=True):
    request = apply_agent_early_stopping_policy(request)
    reference = request["historical_anchor_config"] if request["decision_kind"] == "initialization" else request["current_config"]
    parameters = set(parameter_bounds(reference))
    curves = schedule_curves(reference)
    if not isinstance(semantic, dict) or set(semantic) != SEMANTIC_FIELDS:
        raise ValueError({"semantic_fields": sorted(set(semantic) if isinstance(semantic, dict) else [])})
    intervention = semantic["intervention"]
    if not isinstance(intervention, dict) or set(intervention) != INTERVENTION_FIELDS:
        raise ValueError("invalid semantic intervention")
    request_allowed = set(request.get("allowed_actions", ALLOWED))
    if (
        semantic["action"] not in ALLOWED
        or semantic["action"] not in request_allowed
        or not isinstance(semantic["continue_training"], bool)
    ):
        raise ValueError("invalid action")
    if kind := request["decision_kind"]:
        if kind != "checkpoint_selection" and semantic["action"] == "STOP" and semantic["continue_training"]:
            raise ValueError("STOP must set continue_training=false")
        if kind != "checkpoint_selection" and semantic["action"] != "STOP" and not semantic["continue_training"]:
            raise ValueError("non-STOP action must continue training")
    experiment = semantic["experiment"]
    kind = request["decision_kind"]
    requesting = semantic["action"] == "REQUEST_EXPERIMENT"
    if requesting:
        if kind != "training_review" or not isinstance(experiment, dict):
            raise ValueError("experiment request is invalid")
        if set(experiment) != {"kind", "hypothesis", "parameter_updates", "adversarial_state_after", "steps", "expected_observation", "falsification"}:
            raise ValueError("invalid experiment fields")
        advertised_steps = request.get("experiment_capability", {}).get("steps", [32, 64, 128])
        if experiment["kind"] not in {"paired_training", "discriminator_probe", "custom_code"} or type(experiment["steps"]) is not int or experiment["steps"] not in advertised_steps:
            raise ValueError("invalid experiment kind or steps outside advertised capability")
        if experiment["steps"] > 128 and experiment["kind"] != "paired_training":
            raise ValueError("long review horizons require paired_training")
        proposed = experiment["parameter_updates"]
        if not isinstance(proposed, list) or len(proposed) != (0 if experiment["kind"] == "custom_code" else 1):
            raise ValueError("experiment requires one parameter update")
        item = proposed[0] if proposed else {"parameter": "lambda_adv", "value": request["current_config"]["lambda_adv"]}
        if not isinstance(item, dict) or set(item) != {"parameter", "value"} or item["parameter"] not in parameters:
            raise ValueError("invalid experiment parameter")
        if item['parameter'] in negative.BOUNDS:
            if not reference.get('supplementary_training',{}).get('count'):
                raise ValueError('empty pool cannot identify a negative treatment effect')
            if item['parameter'] != negative.WEIGHT and reference[negative.WEIGHT] == 0:
                raise ValueError('negative exposure controls are inactive at zero weight')
            if item['parameter'] == negative.HORIZON and reference['schedule_state'][negative.WEIGHT]['curve'] == 'constant':
                raise ValueError('horizon is inactive under a constant negative curve')
        if experiment["kind"] == "discriminator_probe" and item["parameter"] != "D_lr":
            raise ValueError("D-only experiment may change only D_lr")
        capability = request.get("experiment_capability", {})
        if experiment["kind"] not in capability.get("kinds", ["paired_training", "discriminator_probe", "custom_code"]):
            raise ValueError("experiment kind is outside advertised capability")
        if "max_per_tile" in capability:
            used, limit = capability.get("attempts_used", 0), capability["max_per_tile"]
            if type(used) is not int or type(limit) is not int or used < 0 or used >= limit:
                raise ValueError("advertised experiment capability budget exhausted or invalid")
        if capability.get("enabled") is False or experiment["steps"] not in capability.get("steps", [32, 64, 128]):
            raise ValueError("experiment exceeds advertised capability")
        trial = flat(request["current_config"])
        trial[item["parameter"]] = item["value"]
        trial["adversarial_state"] = experiment["adversarial_state_after"]
        _check_config(trial)
        if experiment["kind"] != "custom_code" and trial == flat(request["current_config"]):
            raise ValueError("experiment treatment must differ from the formal control")
        if experiment["kind"] == "custom_code" and not capability.get("custom_code", {}).get("script_path"):
            raise ValueError("custom-code workspace was not advertised")
        if experiment["kind"] == "custom_code":
            script = Path(capability["custom_code"]["script_path"])
            if not script.is_file():
                raise ValueError("write the advertised custom_experiment.py before requesting execution")
            compile(script.read_text(), str(script), "exec")
        if experiment["kind"] == "discriminator_probe" and trial["adversarial_state"] != request["current_config"].get("adversarial_state", "active"):
            raise ValueError("D-only experiment may not change adversarial state")
    elif experiment is not None:
        raise ValueError("experiment payload only allowed with REQUEST_EXPERIMENT")
    updates = intervention["parameter_updates"]
    max_updates = len(parameters) if request["decision_kind"] == "initialization" else 4
    if not isinstance(updates, list) or len(updates) > max_updates:
        raise ValueError("too many updates")
    update_map = {}
    for item in updates:
        if not isinstance(item, dict) or set(item) != {"parameter", "value"} or item["parameter"] not in parameters:
            raise ValueError({"invalid_update": item})
        if item["parameter"] in update_map:
            raise ValueError("duplicate update")
        update_map[item["parameter"]] = item["value"]
    kind = request["decision_kind"]
    before = flat(request["historical_anchor_config"] if kind == "initialization" else request["current_config"])
    after = copy.deepcopy(before)
    after.update(update_map)
    after["adversarial_state"] = intervention["adversarial_state_after"]
    if requesting and (update_map or after != before):
        raise ValueError("experiment request may not mutate formal configuration")
    _check_config(after)
    changed = sorted(key for key in parameters if before[key] != after[key])
    if kind == "checkpoint_selection" and (after != before or semantic["action"] != "SELECT_CHECKPOINT"):
        raise ValueError("invalid selection mutation")
    if kind == "initialization" and set(update_map) != parameters:
        raise ValueError("initialization must explicitly choose all exposed controls")
    choices = intervention["schedule_choices"]
    schedule_updates = intervention["schedule_updates"]
    if not isinstance(choices, list) or not isinstance(schedule_updates, list):
        raise ValueError("invalid schedule fields")
    if kind == "initialization":
        if len(choices) != len(curves) or schedule_updates:
            raise ValueError("initialization must choose all exposed schedules")
        names = []
        for item in choices:
            if not isinstance(item, dict) or set(item) != {"component", "curve"} or item.get("curve") not in curves.get(item.get("component"), ()):
                raise ValueError("invalid schedule choice")
            names.append(item["component"])
        if set(names) != set(curves) or len(names) != len(set(names)):
            raise ValueError("duplicate or missing schedule choice")
    elif choices:
        raise ValueError("schedule family is immutable after initialization")
    if kind != "initialization" and len(schedule_updates) > 1:
        raise ValueError("at most one schedule phase update per review")
    current_states = (request.get("current_config") or {}).get("schedule_state") or default_state()
    for item in schedule_updates:
        if not isinstance(item, dict) or set(item) != {"component", "operation", "phase"}:
            raise ValueError("invalid schedule update")
        name, operation, new_phase = item["component"], item["operation"], item["phase"]
        operations = {"hold", "resume", "set_phase", "set_phase_hold"}
        if finish.enabled(reference): operations.add(finish.OPERATION)
        if name not in curves or operation not in operations:
            raise ValueError("invalid schedule operation")
        if operation == finish.OPERATION:
            maximum = int(reference['cosine_finish_policy'].get('maximum_epoch', 200))
            if name != 'main_lr' or reference.get('cosine_finish_state') or not 0 < int(request.get('epoch', 0)) < maximum:
                raise ValueError('cosine finish requires main_lr, a trained state, remaining budget and no previous finish')
            if 'main_lr' in changed:
                raise ValueError('start finish from the unchanged current effective main learning rate')
        elif name == 'main_lr' and reference.get('cosine_finish_state'):
            raise ValueError('ordinary main_lr phase operations are inactive during cosine finish')
        if operation in {"set_phase", "set_phase_hold"}:
            if not isinstance(new_phase, (float, int)) or not math.isfinite(float(new_phase)) or not 0 <= float(new_phase) <= 1:
                raise ValueError("schedule phase must be in [0, 1]")
        elif new_phase is not None:
            raise ValueError("hold/resume phase must be null")
        if operation == "resume" and current_states[name]["mode"] != "hold":
            raise ValueError("resume requires a held schedule")
    if kind in {"checkpoint_selection", "initialization"} and schedule_updates:
        raise ValueError("schedule phase updates are review-only")
    if schedule_updates and semantic["action"] != "ADJUST":
        raise ValueError("schedule phase updates require ADJUST")
    if semantic["action"] == "STOP" and (update_map or after != before):
        raise ValueError("STOP may not mutate formal configuration")
    if requesting and schedule_updates:
        raise ValueError("experiment request may not mutate formal schedule")
    from conditional_negatives_v2 import validate_activation
    validate_activation(request, semantic, before, after)
    if kind in {"training_review", "experiment_review"} and len(changed) + len(schedule_updates) > 2 and not intervention["emergency"]:
        raise ValueError("at most two normal changes including schedule")
    if kind != "initialization" and len(changed) > 2 and not intervention["emergency"]:
        raise ValueError("at most two normal changes")
    selected = semantic["selected_epoch"]
    if kind == "checkpoint_selection" and selected not in {int(x) for x in request["eligible_candidates"]}:
        raise ValueError({"selected_epoch": selected})
    if kind != "checkpoint_selection" and selected is not None:
        raise ValueError("only selection may select an epoch")
    epoch = 0 if kind == "initialization" else int(request.get("epoch", request.get("job", {}).get("stopped_after_epoch", 0)))
    if kind in {"training_review", "experiment_review"} and semantic["action"] == "STOP":
        safety = request.get("safety_envelope", {})
        minimum_epoch = int(safety.get("minimum_stop_epoch", MINIMUM_STOP_EPOCH))
        stop_allowed = bool(safety.get("stop_allowed", epoch >= minimum_epoch))
        if (epoch < minimum_epoch and not bool((request.get("source_protection_policy") or {}).get("source_collapsed"))) or not stop_allowed:
            raise ValueError("STOP violates user minimum epoch or current safety allowance")
    stop_decision = derive_stop_decision(
        kind, semantic["action"], semantic["continue_training"],
        semantic["reason_codes"], semantic["current_state_assessment"],
    )
    return {
        "schema_version": "cashew-fada-training-expert-v4",
        "decision_kind": kind, "request_sha256": request_hash, "epoch": epoch,
        "action": semantic["action"], "continue_training": semantic["continue_training"], "experiment": copy.deepcopy(experiment),
        "selected_epoch": selected, "selected_checkpoint": None,
        "stop_decision": stop_decision,
        "reason_codes": copy.deepcopy(semantic["reason_codes"]),
        "analysis_summary": semantic["analysis_summary"], "planet_used": bool(planet_used),
        "previous_hypothesis_review": copy.deepcopy(semantic["previous_hypothesis_review"]),
        "current_state_assessment": copy.deepcopy(semantic["current_state_assessment"]),
        "intervention": {"continue_training": semantic["continue_training"], "adversarial_state_after": after["adversarial_state"], "parameters_before": before, "parameters_after": after, "changed_parameters": changed, "schedule_choices": copy.deepcopy(choices), "schedule_updates": copy.deepcopy(schedule_updates), "emergency": bool(intervention["emergency"]), "rationale": intervention["rationale"]},
        "next_stage_expectation": copy.deepcopy(semantic["next_stage_expectation"]),
    }


def request_base(job, kind, epoch, mode, threshold, planet_risk, experience):
    research_policy = job.get("research_stopselect_policy")
    research_label_free = research_policy == "trend_judgment_v4" or distilled_policy_enabled(job)
    if research_label_free:
        # v4 research judgments must not receive target Planet prevalence or
        # risk summaries; posthoc labels remain outside the Agent request.
        planet_risk = None
        experience = []
    result = {
        "schema_version": "cashew-fada-training-expert-v4", "decision_kind": kind, "epoch": int(epoch),
        "agent_skill": SKILL_NAME, "job": job, "fada_variant": {"local_mode": mode, "hard_threshold": threshold, "global_branch": "binary_pooled_stage4"},
        "planet_risk": planet_risk, "experience": experience,
        "information_boundary": {"planet_summary_exposed": bool(not research_label_free and planet_risk is not None), "planet_raw_pixels_exposed": False, "target_labels_exposed": False},
        "instruction": (
            "Use $cashew-fada-training-expert. Return only semantic JSON. Planet is advisory only. "
            "Source F1 is retention/safety evidence, not a target-quality ranking objective; ordinary "
            "source degradation is acceptable. Follow source_protection_policy when it is present. "
            "Python derives parameters_after and changed_parameters. You have full-access runtime "
            "permission to correct a confirmed execution-code bug, but only with a separate "
            "experiment_reports correction report and focused verification; never silently alter "
            "training methodology, historical decisions, checkpoints, data, or class mapping."
        ),
    }
    if gradient_control.enabled(job):
        result['instruction'] = result['instruction'].replace(
            'Source F1 is retention/safety evidence, not a target-quality ranking objective; ordinary '
            'source degradation is acceptable. Follow source_protection_policy when it is present. ',
            'Allow ordinary source-validation fluctuations. Diagnose sustained material degradation '
            'alongside repeated weighted gradient conflict before collapse. Source training loss '
            'is not a generalization target. Follow source_protection_policy; source F1 alone '
            'does not rank target quality. ')
    return result


def semantic_schema(schema_path, request=None):
    """Advertise only controls and schedules supported by this pinned packet."""
    schema = json.loads(schema_path.read_text())
    reference = ((request.get("historical_anchor_config") or request.get("current_config") or {}) if request else {})
    parameters = set(parameter_bounds(reference))
    curves = schedule_curves(reference)
    defs = schema["$defs"]
    if finish.enabled(reference):
        defs['schedule_update']['properties']['operation']['enum'].append(finish.OPERATION)
    defs["parameter_update"]["properties"]["parameter"]["enum"] = sorted(parameters)
    for name in ("schedule_choice", "schedule_update"):
        defs[name]["properties"]["component"]["enum"] = sorted(curves)
    defs["schedule_choice"]["properties"]["curve"]["enum"] = sorted(set().union(*curves.values()))
    intervention = defs["intervention"]["properties"]
    intervention["parameter_updates"]["maxItems"] = len(parameters)
    intervention["schedule_choices"]["maxItems"] = len(curves)
    if request is not None and request.get("decision_kind") == "initialization":
        intervention["parameter_updates"].update(minItems=len(parameters), maxItems=len(parameters))
        intervention["schedule_choices"].update(minItems=len(curves), maxItems=len(curves))
        intervention["schedule_updates"]["maxItems"] = 0
    if request is not None:
        capability = request.get("experiment_capability", {})
        advertised = capability.get("steps", [32, 64, 128])
        if (capability.get("enabled") is False or not advertised or
                "REQUEST_EXPERIMENT" not in request.get("allowed_actions", [])):
            # The nullable experiment is the only legal output in this turn.
            # Do not send impossible empty-enum branches to structured decoding.
            schema["properties"]["experiment"] = {"type": "null"}
            defs.pop("experiment")
            return schema
        variants = []
        for variant in defs["experiment"]["anyOf"]:
            for kind in variant["properties"]["kind"]["enum"]:
                scoped = copy.deepcopy(variant)
                scoped["properties"]["kind"]["enum"] = [kind]
                steps = advertised if kind == "paired_training" else [x for x in advertised if x <= 128]
                if not steps:
                    continue
                scoped["properties"]["steps"]["enum"] = steps
                variants.append(scoped)
        defs["experiment"]["anyOf"] = variants
    return schema
