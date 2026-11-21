"""Pinned late-negative diagnostic priorities, including explicit user error reports.

No coefficient, sample schedule, pixel label, stopping or selection is chosen here.
"""
import hashlib
import json
import math
from pathlib import Path

VERSION = "late_negative_response_v3"


def feedback_evidence(job):
    pointer = job.get("user_reported_target_error")
    if pointer is None:
        return None
    if not isinstance(pointer, dict) or not pointer.get("manifest") or not pointer.get("sha256"):
        raise ValueError("user feedback must have pinned manifest path/hash")
    path = Path(pointer["manifest"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pointer["sha256"]:
        raise ValueError("user feedback manifest identity changed")
    data = json.loads(raw)
    tile = str(job.get("target_tile", job["job_id"])).rsplit("_", 1)[-1]
    cases = [case for case in data.get("cases", []) if case.get("tile_id") == tile]
    if len(cases) != 1:
        raise ValueError("user feedback must identify this exact tile once")
    case = cases[0]
    parent = json.loads(Path(case["job"]).read_text())
    if (Path(parent["target_root"]).resolve() != Path(job["target_root"]).resolve()
            or parent["target_year"] != job["target_year"]):
        raise ValueError("user feedback is for a different target/year")
    return {"tile_id": tile, "feedback": case["feedback"], "manifest": str(path),
            "manifest_sha256": pointer["sha256"], "original_selected_epoch": case.get("original_selected_epoch"),
            "role": "user-observed quality concern; not pixel labels or measured accuracy",
            "below_area_warning_still_requires_diagnosis": True}


def packet_evidence(packet, job):
    policy = job.get("negative_response_policy")
    if policy is None:
        return None
    if policy.get("version") != VERSION:
        raise ValueError("unsupported negative response policy")
    feedback = feedback_evidence(job)
    review = packet.get("current_review") or {}
    metrics = review.get("metrics") or {}
    controls = metrics.get("negative_controls") or {}
    total, updates = metrics.get("steps"), controls.get("negative_updates")
    fraction = (updates / total if type(updates) is int and type(total) is int and total > 0 else None)
    area = packet.get("area_alert_evidence") or {}
    alarm = (area.get("current") or {}).get("level") in ("warning", "high", "severe")
    kind = packet["decision_kind"]
    source_unsafe = ((packet.get("source_protection_policy") or {}).get("source_collapsed", False)
                     or (packet.get("safety_envelope") or {}).get("source_collapsed", False))
    from area_alert_v1 import late_start
    start = late_start(job.get("conditional_negative_policy", {}), int(job["max_epoch"]), int(job.get("review_interval", 5)))
    epoch = packet.get("epoch", 0)
    late = type(epoch) is int and epoch >= start
    pool = int(job.get("supplementary", {}).get("count", 0))
    exposed = (controls.get("negative_updates", 0) > 0 and controls.get("effective_weight_max", 0) > 0
               and metrics.get("supp_eligible_pixels", 0) > 0)
    focus = "assess_negative_mechanism_and_actual_pressure"
    if source_unsafe:
        focus = "disable_and_follow_bounded_source_repair"
    elif kind == "initialization" or (kind != "checkpoint_selection" and not late):
        focus = "retain_initial_off_before_pinned_late_boundary"
    elif kind == "checkpoint_selection":
        focus = "compare_all_safe_candidates_and_reopen_reported_error_without_minimum_area_ranking"
    elif pool == 0:
        focus = "retain_empty_branch_and_assess_checkpoint_or_decoding_repair"
    elif kind == "experiment_review":
        focus = "interpret_completed_pair_and_adopt_or_escalate_safe_weak_response_one_factor"
    elif feedback is not None or alarm:
        focus = ("test_meaningfully_stronger_coefficient_or_finite_exposure_if_weak_safe_response"
                 if exposed else "prioritize_informative_late_negative_trial_or_record_concrete_blocker")
    return {"version": VERSION, "user_reported_target_error": feedback,
            "recommended_review_focus": focus, "minimum_activation_review": start,
            "late_activation_eligible": late, "selected_pool_count": pool,
            "measured_epoch_pressure": {"negative_updates": updates, "total_model_updates": total,
                                        "negative_update_fraction": fraction,
                                        "base_weight": controls.get("base_weight"),
                                        "effective_weight_min": controls.get("effective_weight_min"),
                                        "effective_weight_max": controls.get("effective_weight_max"),
                                        "finite_occurrence_cap": controls.get("occurrence_cap_per_epoch"),
                                        "eligible_pixels": metrics.get("supp_eligible_pixels"),
                                        "pool_coverage_fraction": metrics.get("supp_pool_coverage_fraction")},
            "confirmed_actual_exposure": exposed,
            "rules": ["User-reported error triggers diagnosis below15% area; no extensive precise labels required",
                      "Actual coverage once is not sufficient sustained exposure; assess coefficient and finite frequency separately",
                      "Safe weak response motivates a meaningful higher dose or bounded exposure contrast, not automatic off",
                      "One area rebound, ordinary source F1 change, temporal movement or gradient opposition alone is not cashew harm",
                      "Empty/invalid negatives remain off; actual source collapse and demonstrated harm override escalation",
                      "No automatic threshold, coefficient, label, checkpoint, unlimited reuse or training restart"],
            "automatic_parameter_change": False, "target_accuracy_proven": False}
