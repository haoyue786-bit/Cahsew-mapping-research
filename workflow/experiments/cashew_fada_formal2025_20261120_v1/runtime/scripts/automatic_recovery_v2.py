"""Register one bounded source-only repair after a new-policy collapse.

The safety strategy is the user's explicit off/off procedure, not a learned
numeric optimum. Subsequent tuning, stopping and selection remain Agent turns.
No failed trajectory or selected checkpoint is rewritten.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from protocol import atomic_json, sha256


def semantic_repair(config, anchor, terminal, floor):
    from expert_policy_v4 import flat
    updates = []
    if config["supplementary_loss_weight"] > 0:
        updates.append({"parameter": "supplementary_loss_weight", "value": 0.0})
    active = config["adversarial_state"] == "active"
    if not active and not updates:
        raise ValueError("healthy anchor is already source-only; a new Agent strategy is required")
    return {
        "action": "PAUSE_ADV" if active else "ADJUST", "continue_training": True,
        "selected_epoch": None, "experiment": None,
        "reason_codes": ["registered_source_only_safety_repair", "original_collapse_preserved"],
        "analysis_summary": "Restore the verified healthy full state and explicitly disable supplementary and target-adversarial pressure. This is the user-authorized bounded safety procedure, not a target-quality claim or a causal experiment.",
        "previous_hypothesis_review": {"status": "not_applicable", "summary": "Separate recovery lineage starts from a frozen healthy state.", "evidence": []},
        "current_state_assessment": {"state": "bounded_source_repair", "summary": "The original trajectory collapsed; this anchor was noncollapsed.",
            "evidence": ["Healthy anchor epoch %s; terminal source F1 %s below floor %s." % (anchor, terminal, floor),
                         "The collapse may reflect adversarial pressure, supplementary pressure, optimizer dynamics or combinations; off/off repair does not identify the cause."],
            "candidate_checkpoint_window": None, "candidate_window_rationale": None},
        "intervention": {"adversarial_state_after": "paused", "parameter_updates": updates,
            "schedule_choices": [], "schedule_updates": [], "emergency": True,
            "rationale": "Apply the explicitly registered source-only safety bundle while preserving data, loss, architecture and optimizer/schedule/RNG restoration. Another collapse or nonfinite state ends this bounded attempt; an earlier healthy state is not evidence of better target predictions."},
        "next_stage_expectation": {"hypothesis": "Source-only continuation avoids the confirmed source collapse while measuring recovery from the complete healthy anchor.",
            "expected_source": "Remain finite and outside the recorded catastrophic-collapse floor.",
            "expected_target": "Record prediction movement; source restoration and map consistency do not establish target accuracy.",
            "expected_discriminator": "Target adversarial updates are paused; their zero response is unmeasured.",
            "expected_gradient_ratio": "Negative and adversarial pressure are off; no active ratio target.",
            "failure_triggers": ["Source collapse recurs", "Nonfinite state", "Registered epoch/call/time allowance reached"]},
    }


def register(job_path):
    from train_expert_v4 import load_resumable_checkpoint, active_saved_cuda_ordinal
    from bounded_recovery_v4 import verify_full_state
    from expert_policy_v4 import normalize
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text())
    policy = job.get("recovery_policy", {})
    if job.get("bounded_recovery") or policy.get("version") != "automatic_bounded_source_repair_v2" or not policy.get("enabled"):
        return None
    root = Path(job["output_root"])
    selection_path = root / "final/selection.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("stop_reason") != "source_collapse_safety_exit":
        return None
    remaining = int(job["max_epoch"]) - int(selection["stop_epoch"])
    allowance = min(int(policy["max_extra_epochs"]), remaining)
    if allowance <= 0:
        raise ValueError("collapse has no remaining registered recovery epochs")
    workspace = root / "recovery_registration"
    plan_path = workspace / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["source_job_sha256"] != sha256(job_path) or plan["source_selection_sha256"] != sha256(selection_path):
            raise ValueError("existing recovery plan belongs to a changed original trajectory")
        return {"path": str(plan_path), "sha256": sha256(plan_path)}
    baseline = json.loads((root / "review_epoch_000.json").read_text())["candidate"]["source_class2_f1"]
    floor = max(.35, float(baseline) - .30)
    terminal = json.loads((root / ("review_epoch_%03d.json" % selection["stop_epoch"])).read_text())["candidate"]["source_class2_f1"]
    failures = []
    candidate = None
    for record in sorted(selection["candidate_checkpoint_records"], key=lambda x: x["epoch"], reverse=True):
        epoch = int(record["epoch"])
        if epoch <= 0 or record.get("source_collapsed"):
            continue
        path = Path(record["checkpoint"])
        try:
            if sha256(path) != record["checkpoint_sha256"]:
                raise ValueError("healthy checkpoint hash changed")
            payload = load_resumable_checkpoint(path)
            anchor_review = json.loads((root / ("review_epoch_%03d.json" % epoch)).read_text())["candidate"]
            health = float(anchor_review["source_class2_f1"])
            if not math.isfinite(health) or health < floor or Path(anchor_review["checkpoint"]).resolve() != path.resolve():
                raise ValueError("anchor lacks its recorded finite healthy review")
            ordinal = active_saved_cuda_ordinal(payload, {}, root)
            verify_full_state(payload, expected_epoch=epoch, saved_cuda_ordinal=ordinal)
            config = payload["config"]
            semantic = semantic_repair(config, epoch, terminal, floor)
            candidate = record
            break
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            failures.append({"epoch": epoch, "error": str(exc)})
    if candidate is None:
        atomic_json(workspace / "registration_failure.json", {"state": "NO_VERIFIED_REPAIRABLE_TRAINED_ANCHOR", "candidates": failures})
        raise ValueError("no complete trained healthy anchor for source-only repair")
    review = json.loads((root / ("review_epoch_%03d.json" % epoch)).read_text())["review"]
    from train_expert_v4 import agent_job_summary
    request = {"decision_kind": "training_review", "epoch": epoch, "job": agent_job_summary(job),
        "current_config": copy.deepcopy(config), "current_review": review,
        "planet_risk": None, "target_labels": None,
        "source_protection_policy": {"source_collapsed": False, "effective_floor": floor},
        "allowed_actions": ["PAUSE_ADV", "ADJUST"],
        "recovery_trigger": {"original_stop_epoch": selection["stop_epoch"], "terminal_source_f1": terminal,
                             "strategy_owner": "user_authorized_deterministic_safety_procedure_not_an_Agent_tuning_choice"}}
    request_path, semantic_path = workspace / "request.json", workspace / "semantic.json"
    atomic_json(request_path, request)
    atomic_json(semantic_path, semantic)
    normalize(request, semantic, sha256(request_path), planet_used=False)
    plan = {"protocol": "bounded_healthy_recovery_v1", "strategy_kind": "source_only_repair",
        "source_job": str(job_path), "source_job_sha256": sha256(job_path),
        "source_selection_sha256": sha256(selection_path), "anchor_epoch": epoch,
        "anchor_checkpoint": str(Path(candidate["checkpoint"]).resolve()), "anchor_sha256": candidate["checkpoint_sha256"],
        "request": str(request_path.resolve()), "request_sha256": sha256(request_path),
        "semantic": str(semantic_path.resolve()), "semantic_sha256": sha256(semantic_path),
        "output_root": str((root.parent / (root.name + "__bounded_recovery")).resolve()),
        "max_attempts": int(policy["max_attempts"]), "max_extra_epochs": int(policy["max_extra_epochs"]),
        "additional_epochs": allowance, "max_agent_submissions": int(policy["max_agent_submissions"]),
        "max_model_calls": int(policy["max_model_calls"]), "timeout_seconds": int(policy["timeout_seconds"]),
        "mechanism": "Explicit source-only continuation from a verified healthy full state; negative/adversarial causes remain unseparated.",
        "reversal_condition": "A recurrent collapse/nonfinite state ends recovery; resuming either pressure requires a subsequent recorded Agent decision.",
        "rejected_anchor_records": failures, "automatic_strategy_provenance": request["recovery_trigger"]}
    atomic_json(plan_path, plan)
    return {"path": str(plan_path.resolve()), "sha256": sha256(plan_path)}
