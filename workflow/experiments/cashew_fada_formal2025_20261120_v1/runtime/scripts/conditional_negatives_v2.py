"""Evidence-bound late activation, separate from negative sampling mechanics."""
from __future__ import annotations

import copy
import math

VERSION = "conditional_negatives_v2"


def policy_for(job):
    policy = job.get("conditional_negative_policy", {})
    if policy and policy.get("version") != VERSION:
        raise ValueError("unsupported conditional negative policy")
    return policy


def initialize(config, job):
    policy = policy_for(job)
    if not policy:
        return
    config["conditional_negative_policy"] = copy.deepcopy(policy)
    config["supplementary_loss_weight"] = 0.0
    config["supplementary_occurrence_cap"] = 1


def validate_activation(request, semantic, before, after):
    """Enforce a zero start; an explicit later decision owns every activation.

    Evidence strings are an audit trail, not an automatic target-quality score.
    Source/target novelty and predicted area never confer a negative label.
    """
    policy = request.get("job", {}).get("conditional_negative_policy")
    policy = policy or before.get("conditional_negative_policy")
    if not policy:
        return
    if policy.get("version") != VERSION:
        raise ValueError("unsupported conditional negative policy")
    if request["decision_kind"] == "initialization":
        if after["supplementary_loss_weight"] != 0:
            raise ValueError("new formal runs must initialize supplementary weight at zero")
        return
    if request["decision_kind"] == "checkpoint_selection":
        return
    if policy.get('spatial_prior_guidance')=='spatial_prior_negative_guidance_v2':
        collapsed=(request.get('source_protection_policy') or {}).get('source_collapsed',False)
        collapsed=collapsed or (request.get('safety_envelope') or {}).get('source_collapsed',False)
        experiment=semantic.get('experiment') or {}
        positive_trial=any(x.get('parameter')=='supplementary_loss_weight' and x.get('value',0)>0
                           for x in experiment.get('parameter_updates',[]))
        if collapsed and (after['supplementary_loss_weight']>0 or positive_trial):
            raise ValueError('source collapse requires negatives off; spatial prior cannot override source safety')
    # New jobs pin a late boundary. Historical jobs without this field retain
    # their recorded contract; never reinterpret their earlier interventions.
    if policy.get("minimum_activation_fraction") is not None:
        from area_alert_v1 import late_start
        summary = request.get("job", {})
        minimum = late_start(policy, int(summary["max_epoch"]), int(summary.get("review_interval", 5)))
        epoch = int(request.get("epoch", 0))
        experiment = semantic.get("experiment") or {}
        requests_positive_trial = any(
            item.get("parameter") == "supplementary_loss_weight" and item.get("value", 0) > 0
            for item in experiment.get("parameter_updates", []))
        if epoch < minimum and (after["supplementary_loss_weight"] > 0 or requests_positive_trial):
            raise ValueError("supplementary negatives are restricted to late reviews at epoch >= %d" % minimum)
    # A positive base can still be truly off at cosine phase1. Moving its
    # phase earlier or extending its horizon can activate loss without a
    # scalar weight change. Check the same effective controls as the worker,
    # after the caller has validated the schedule operations.
    import supplementary_control_v4 as negative
    from schedule_v4 import update_state
    current = request["current_config"]
    prospective = copy.deepcopy(current)
    prospective.update(after)
    epoch = int(request.get("epoch", 0))
    update_state(prospective, semantic["intervention"], epoch)
    activates_base = before[negative.WEIGHT] == 0 and after[negative.WEIGHT] > 0
    activates_effective = negative.weight(current, epoch) == 0 and negative.weight(prospective, epoch) > 0
    if activates_base or activates_effective:
        if request["decision_kind"] not in {"training_review", "experiment_review"}:
            raise ValueError("negative activation requires a recorded training review")
        if semantic["action"] != "ADJUST":
            raise ValueError("negative activation must be an explicit ADJUST")
        assessment = semantic.get("current_state_assessment", {})
        expectation = semantic.get("next_stage_expectation", {})
        evidence = assessment.get("evidence", [])
        if not evidence or not all(isinstance(x, str) and x.strip() for x in evidence):
            raise ValueError("negative activation requires recorded supporting and contrary evidence")
        if not semantic["intervention"].get("rationale", "").strip():
            raise ValueError("negative activation requires a mechanism and label-reliability rationale")
        if not expectation.get("hypothesis", "").strip() or not expectation.get("failure_triggers"):
            raise ValueError("negative activation requires a hypothesis and reversal condition")


def packet_evidence(job):
    policy = policy_for(job)
    if not policy:
        return {}
    import hashlib
    import json
    from pathlib import Path
    selection = job["supplementary"]["selection"]
    path = Path(selection["report"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != selection["report_sha256"]:
        raise ValueError("negative selection evidence hash changed")
    report = json.loads(path.read_text())
    if job.get('polygon_supervision_policy'):
        from human_polygon_v1 import packet
        return dict(policy=copy.deepcopy(policy),candidate_pool_count=report['selected_count'],
                    geographic_filter=report.get('geographic_filter'),human_polygon_supervision=packet(job,report),
                    label_reliability=dict(origin='Current user replacement polygons',independently_verified=False,
                        instruction='Unmarked pixels are ignored. This cohort uses human polygons, not the disputed old weak negatives. Assess local transfer and retention; scale0 permits negative-only; thirteen scalar choices and four schedule families must be explicitly initialized.'),
                    instruction=(('For fixed_supplementary_default_v1: both signs start off; after E50 the fixed baseline activates them when samples exist, unless an earlier Agent intervention overrides the pending default. Assess training-label exposure, source safety and unlabelled target behavior; do not claim independent polygon feedback.' if job.get('supplementary_default_policy') == 'fixed_supplementary_default_v1' else 'Both signs start off; Agent may activate at any trained review from current human training labels and spatial/feature relevance. No fixed50-epoch restriction. Similarity ranks evidence without rejecting novel types; current polygons do not provide independent transfer accuracy. Respect source collapse and empty pools.')
                                 if job.get('human_samples_policy',{}).get('version')=='human_samples_agent_timing_v1' else
                                 'Both signs must start off and remain off before E50; later Agent reviews own dose, positive support and bounded exposure. Respect source collapse and empty pools. Do not treat labels as automatic proof of transfer performance.'))
    return {
        "policy": copy.deepcopy(policy),
        "candidate_pool_count": report["selected_count"],
        "geographic_filter": report.get("geographic_filter"),
        "source_coverage": report.get("source_coverage"),
        "label_reliability": {
            "origin": "supplementary Planet-derived weak labels",
            "independently_verified": False,
            "unknown_and_class2_pixels_excluded": True,
            "instruction": "Feature novelty/similarity and geography do not validate a non-class2 label. State uncertainty before using supplementary suppression.",
        },
        "instruction": "Start weight0. When pinned, respect minimum_activation_fraction for both formal activation and positive-dose paired trials. A whole run may stay off for a concrete reason, but label uncertainty alone must not perpetually replace an informative late trial when area alarms persist. Identify a background mechanism, candidate relevance, competing explanations, a measurable expectation and a safety reversal condition. With confirmed exposure and source safety, small effects motivate testing a larger dose rather than disabling. Area rebound alone is not evidence of harm. Reduce/disable for source collapse, demonstrated damage to plausible cashew, invalid labels or invalid exposure. Area and source F1 alone are not target quality.",
    }


def epoch_steps(source_count, target_count, batch_size=16):
    return max(math.ceil(source_count / batch_size), math.ceil(target_count / batch_size))


def experiment_steps(job):
    """Long arms use actual epoch lengths, with +5/+10 review endpoints."""
    policy = policy_for(job)
    if not policy:
        return [32, 64, 128]
    steps = epoch_steps(job["source"]["source_train_expected_count"], job["target_train_count"])
    return [32, 64, 128, 5 * steps, 10 * steps]
