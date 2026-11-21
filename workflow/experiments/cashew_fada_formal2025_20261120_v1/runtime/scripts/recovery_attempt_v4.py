#!/usr/bin/env python3
"""Execute an explicitly registered bounded recovery in a separate lineage.

The strategy is supplied by a normal schema/normalizer-valid review. This
entry chooses no parameter value and retains normal subsequent reviews and
SELECT. Preparation and CPU full-state compatibility checks never start a
training or Agent process.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import torch

from bounded_recovery_v4 import BoundedRecovery, _finite
from expert_policy_v4 import flat, normalize
import supplementary_control_v4 as negative
from healthy_recovery_runtime_v4 import verify_resume_job_identity
from protocol import atomic_json, sha256
from train_tile import torch_load

PROTOCOL = "bounded_healthy_recovery_v1"
HERE = Path(__file__).resolve().parent


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Recovery requires an explicit integer: " + name)
    return value


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_files(files):
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing frozen runtime/input identities")
    for name, expected in files.items():
        path = Path(name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("Frozen recovery identity changed: " + name)


def verify_source_runtime(source_job, snapshots):
    """A recorded historical file may be supplied as an exact archived copy."""
    provenance = source_job.get("runtime_provenance", {})
    files = provenance.get("files", {})
    if files:
        actual = {str(Path(snapshots.get(name, name)).resolve()): expected for name, expected in files.items()}
    elif provenance.get("runtime_tree_sha256") and source_job.get("runtime_root"):
        # Historical paired packages froze a canonical relative-name map,
        # rather than the newer per-file absolute-path field. Keep its exact
        # original algorithm and verify the whole specifically named root.
        original_root = source_job["runtime_root"]
        root = Path(snapshots.get(original_root, original_root)).resolve()
        if not root.is_dir():
            raise ValueError("Recorded source runtime tree is missing")
        tree = {path.relative_to(root).as_posix(): sha256(path) for path in sorted(root.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
        tree_hash = hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()
        if not tree or tree_hash != provenance["runtime_tree_sha256"]:
            raise ValueError("Frozen recovery runtime tree changed")
        actual = {str(root / name): expected for name, expected in tree.items()}
        for skill, key in [("cashew-fada-training-expert", "training_skill_tree_sha256"),
                           ("cashew-fada-stop-select-experiment", "stopselect_skill_tree_sha256")]:
            if provenance.get(key):
                skill_root = root / ".codex/skills" / skill
                subset = {name: h
                          for name, h in tree.items() if skill_root in (root / name).parents}
                if not subset or hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest() != provenance[key]:
                    raise ValueError("Frozen recovery skill tree changed: " + skill)
        if provenance.get("common_runtime_outside_package_components_sha256"):
            prefixes = tuple(".codex/skills/" + name + "/" for name in
                             ["cashew-fada-training-expert", "cashew-fada-stop-select-experiment"])
            common = {name: h for name, h in tree.items()
                      if name != "scripts/expert_dispatch_v4.py" and not name.startswith(prefixes)}
            if (hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
                    != provenance["common_runtime_outside_package_components_sha256"]):
                raise ValueError("Frozen recovery common runtime changed")
        if (provenance.get("dispatcher_sha256")
                and sha256(root / "scripts/expert_dispatch_v4.py") != provenance["dispatcher_sha256"]):
            raise ValueError("Frozen recovery dispatcher changed")
    else:
        raise ValueError("Source runtime provenance is missing")
    verify_files(actual)
    source = source_job["source"]
    for name, hash_name in [("pretrained", "pretrained_sha256"), ("source_train", "source_train_sha256"),
                            ("source_valid", "source_valid_sha256")]:
        if not source.get(hash_name) or sha256(Path(source[name])) != source[hash_name]:
            raise ValueError("Source data/weights provenance changed: " + name)
    shared = provenance.get("shared_inputs", {})
    supplementary = source_job.get("supplementary", {})
    shared_paths = {"target_full_list": source_job.get("target_full_list"),
                    "target_train_list": source_job.get("target_train_list"),
                    "supplementary_list": supplementary.get("names_file"),
                    "supplementary_report": supplementary.get("selection", {}).get("report")}
    for key, expected in shared.items():
        if key not in shared_paths or not shared_paths[key] or sha256(Path(shared_paths[key])) != expected:
            raise ValueError("Frozen source shared input changed: " + key)
        actual[str(Path(shared_paths[key]).resolve())] = expected
    return actual


def anchor_objects(job, config):
    """Construct the maintained architecture and optimizer topology on CPU."""
    import train_expert_v4 as worker
    device = torch.device("cpu")
    gan, _ = worker.configure_runtime(job, device)
    model, _ = worker.make_architecture_model(gan, job, device)
    local = worker.FineGrainedDomainDiscriminator(768, 4, (256, 128)).to(device)
    global_d = worker.GlobalDomainDiscriminator(768).to(device)
    model_opt = torch.optim.AdamW(worker.model_parameter_groups(
        model, config["main_lr"], config["main_weight_decay"], (0.9, 0.999), 1e-8))
    local_opt = torch.optim.AdamW(local.parameters(), lr=config["D_lr"], weight_decay=config["D_weight_decay"])
    global_opt = torch.optim.AdamW(global_d.parameters(), lr=config["D_lr"], weight_decay=config["D_weight_decay"])
    return model, local, global_d, model_opt, local_opt, global_opt


def checked_strategy(plan, payload, source_job):
    request_path, semantic_path = Path(plan["request"]), Path(plan["semantic"])
    for name, path in [("request", request_path), ("semantic", semantic_path)]:
        if not plan.get(name + "_sha256") or sha256(path) != plan[name + "_sha256"]:
            raise ValueError("Frozen strategy input changed: " + name)
    request, semantic = read_json(request_path), read_json(semantic_path)
    if request.get("decision_kind") not in {"training_review", "experiment_review"}:
        raise ValueError("Recovery strategy must be a review, not initialization")
    if request.get("epoch") != payload["epoch"] or flat(request["current_config"]) != flat(payload["config"]):
        raise ValueError("Strategy is not based on the exact restored config/epoch")
    if request["current_config"].get("schedule_state") != payload["config"]["schedule_state"]:
        raise ValueError("Strategy schedule state differs from the checkpoint")
    if request.get("planet_risk") or request.get("target_labels"):
        raise ValueError("Recovery decisions must retain the label-free boundary")
    if request.get("job", {}).get("job_id") != source_job["job_id"]:
        raise ValueError("Recovery strategy belongs to another job")
    if type(request.get("epoch")) is not int:
        raise ValueError("Recovery strategy epoch must be an integer")
    decision = normalize(request, semantic, sha256(request_path), planet_used=False)
    intervention = decision["intervention"]
    count = len(intervention["changed_parameters"]) + len(intervention["schedule_updates"])
    count += int(intervention["adversarial_state_after"] != payload["config"]["adversarial_state"])
    allowed_off = {negative.WEIGHT} if payload["config"].get(negative.WEIGHT, 0) > 0 else set()
    source_only = (plan.get("strategy_kind") == "source_only_repair"
                   and negative.enabled(payload["config"])
                   and decision["action"] in {"PAUSE_ADV", "ADJUST"}
                   and intervention["adversarial_state_after"] == "paused"
                   and set(intervention["changed_parameters"]) == allowed_off
                   and intervention["parameters_after"][negative.WEIGHT] == 0
                   and not intervention["schedule_updates"] and count in {1, 2})
    if decision["action"] not in {"ADJUST", "PAUSE_ADV", "RESUME_ADV"} or (count != 1 and not source_only):
        raise ValueError("Recovery requires one normalizer-valid changed control or schedule phase")
    if plan.get("strategy_kind") == "source_only_repair" and not source_only:
        raise ValueError("Source-only repair requires the explicit paired off controls")
    for name in ["mechanism", "reversal_condition"]:
        if not isinstance(plan.get(name), str) or not plan[name].strip():
            raise ValueError("Recovery strategy needs " + name)
    strategy = {"validated_by_normalizer": True, "changed_control_count": count,
                "decision_sha256": sha256(semantic_path), "mechanism": plan["mechanism"],
                "reversal_condition": plan["reversal_condition"]}
    if source_only:
        strategy.update(strategy_kind="source_only_repair", source_only_controls_verified=True,
                        repair_controls={"adversarial_state": "paused", negative.WEIGHT: 0},
                        matched_causal_experiment=False)
    return decision, strategy


def ceiling_for_job(job):
    """Bound the child loop without changing the pinned original max_epoch."""
    limits = job["bounded_recovery"]
    if limits.get("protocol") != PROTOCOL:
        raise ValueError("Unrecognized bounded recovery protocol")
    plan_path = Path(limits["plan"])
    if sha256(plan_path) != limits["plan_sha256"]:
        raise ValueError("Recovery plan hash mismatch")
    plan = read_json(plan_path)
    start = integer(limits["restore_epoch"], "restore_epoch", 1)
    count = integer(limits["additional_epochs"], "additional_epochs", 1)
    original = integer(limits["original_max_epoch"], "original_max_epoch", 1)
    stopped = integer(limits["original_stop_epoch"], "original_stop_epoch")
    attempt = integer(limits["attempt"], "attempt", 1)
    if (not start <= stopped <= original or attempt > integer(plan["max_attempts"], "max_attempts")
            or count > min(integer(plan["max_extra_epochs"], "max_extra_epochs"), original - stopped)
            or count != plan["additional_epochs"] or start != plan["anchor_epoch"]
            or job["max_epoch"] != original or job["resume"]["epoch"] != start
            or job["resume"]["checkpoint_sha256"] != plan["anchor_sha256"]):
        raise ValueError("Recovery loop exceeds its frozen original/extra/attempt allowance")
    return start + count


def load_resume_strategy(job, config):
    limits = job["bounded_recovery"]
    ceiling_for_job(job)
    plan = read_json(limits["plan"])
    payload = {"epoch": job["resume"]["epoch"], "config": config}
    decision, _ = checked_strategy(plan, payload, read_json(plan["source_job"]))
    root = Path(job["output_root"]) / "agent/frozen_recovery_strategy"
    atomic_json(root / "request.json", read_json(plan["request"]))
    atomic_json(root / "decision.json", decision)
    atomic_json(root / "provenance.json", {"plan_sha256": limits["plan_sha256"],
                "frozen_semantic_sha256": plan["semantic_sha256"], "new_model_calls": 0})
    return decision


def reserve_submission(job, stage):
    """Persist an admission before dispatch, including failed submissions."""
    limits = job["bounded_recovery"]
    path = Path(job["output_root"]) / "recovery_agent_submissions.json"
    entries = read_json(path) if path.is_file() else []
    cap = integer(limits["max_agent_submissions"], "max_agent_submissions")
    if len(entries) >= cap:
        raise RuntimeError("Bounded recovery Agent submission allowance exhausted")
    entries.append({"number": len(entries) + 1, "stage": stage})
    atomic_json(path, entries)


def reserve_model_call(request, attempt):
    """Admission for each real turn, including repairs and failed attempts."""
    envelope = request.get("bounded_recovery_runtime")
    if envelope is None:
        return
    root = Path(envelope["output_root"]).resolve()
    job = read_json(root / "job.json")
    if Path(job["output_root"]).resolve() != root or job["bounded_recovery"] != envelope["limits"]:
        raise ValueError("Recovery call-budget envelope does not match its frozen job")
    ceiling_for_job(job)
    with (root / "recovery_model_calls.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "recovery_model_calls.json"
        entries = read_json(path) if path.is_file() else []
        cap = integer(job["bounded_recovery"]["max_model_calls"], "max_model_calls")
        if len(entries) >= cap:
            raise RuntimeError("Bounded recovery actual model-call allowance exhausted")
        entries.append({"number": len(entries) + 1, "decision_kind": request["decision_kind"],
                        "epoch": request.get("epoch"), "attempt": attempt})
        atomic_json(path, entries)


def live_process_group_members(pgid):
    """Inspect only Linux process metadata for this owned process group."""
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == int(pgid) and fields[0] != "Z":
                result.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return result


def reap_worker(process):
    from run_year_country_v4 import terminate_process_group
    if process.pid <= 1 or process.pid == os.getpgrp():
        raise ValueError("Refuse unrelated/current process-group cleanup")
    terminate_process_group(process.pid)
    process.wait(timeout=10)
    members = live_process_group_members(process.pid)
    if members:
        raise RuntimeError("Owned recovery helper processes remain live: " + str(members))


def prepare(plan_path):
    import train_expert_v4 as worker
    from formal_v4 import max_epoch_for_job
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan.get("protocol") != PROTOCOL:
        raise ValueError("Unrecognized bounded recovery plan")
    source_path = Path(plan["source_job"]).resolve()
    source_job = read_json(source_path)
    source_root = Path(source_job["output_root"]).resolve()
    if source_path != source_root / "job.json" or sha256(source_path) != plan["source_job_sha256"]:
        raise ValueError("Source job identity mismatch")
    output = Path(plan["output_root"]).resolve()
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError("Recovery must use a separate output lineage")
    selection_path = source_root / "final/selection.json"
    selection = read_json(selection_path)
    if sha256(selection_path) != plan["source_selection_sha256"]:
        raise ValueError("Frozen original selection changed")
    stopped = integer(selection["stop_epoch"], "original_stop_epoch")
    original = max_epoch_for_job(source_job)
    anchor_epoch = integer(plan["anchor_epoch"], "anchor_epoch", 1)
    if not anchor_epoch <= stopped <= original:
        raise ValueError("Recovery checkpoint is outside the stopped trajectory")
    baseline_path = source_root / "review_epoch_000.json"
    anchor_review_path = source_root / ("review_epoch_%03d.json" % anchor_epoch)
    terminal_path = source_root / ("review_epoch_%03d.json" % stopped)
    baseline = float(read_json(baseline_path)["candidate"]["source_class2_f1"])
    floor = max(worker.SOURCE_COLLAPSE_ABSOLUTE_F1, baseline - worker.SOURCE_COLLAPSE_DROP_FROM_BASELINE)
    if not math.isfinite(baseline):
        raise ValueError("Source baseline health is unknown")
    terminal_f1 = float(read_json(terminal_path)["candidate"]["source_class2_f1"])
    if (selection["stop_reason"] != "source_collapse_safety_exit"
            or (math.isfinite(terminal_f1) and terminal_f1 >= floor)):
        raise ValueError("An ordinary source drop is not a collapse recovery trigger")
    candidate = read_json(anchor_review_path)["candidate"]
    anchor_f1 = float(candidate["source_class2_f1"])
    checkpoint = Path(candidate["checkpoint"]).resolve()
    if (not math.isfinite(anchor_f1) or anchor_f1 < floor
            or checkpoint != Path(plan["anchor_checkpoint"]).resolve()
            or sha256(checkpoint) != plan["anchor_sha256"]):
        raise ValueError("Recovery anchor is not the recorded finite healthy checkpoint")
    runtime_files = verify_source_runtime(source_job, plan.get("source_runtime_snapshot", {}))
    payload = torch_load(checkpoint, "cpu")
    decision, strategy = checked_strategy(plan, payload, source_job)
    resume = {"source_run_root": str(source_root), "checkpoint": str(checkpoint),
              "checkpoint_sha256": plan["anchor_sha256"], "epoch": anchor_epoch}
    if "saved_active_cuda_ordinal" in plan:
        resume["saved_active_cuda_ordinal"] = plan["saved_active_cuda_ordinal"]
    objects = anchor_objects(source_job, payload["config"])
    _, _, proof = worker.restore_resume_worker(source_job, resume, torch.device("cpu"), *objects)
    del objects
    attempts = integer(plan["max_attempts"], "max_attempts")
    extra = integer(plan["max_extra_epochs"], "max_extra_epochs")
    additional = integer(plan["additional_epochs"], "additional_epochs", 1)
    submissions = integer(plan["max_agent_submissions"], "max_agent_submissions", 1)
    model_calls = integer(plan["max_model_calls"], "max_model_calls", 1)
    if model_calls < submissions:
        raise ValueError("Actual model-call allowance cannot be below semantic submissions")
    integer(plan["timeout_seconds"], "timeout_seconds", 1)
    if additional > min(extra, original - stopped) or attempts < 1:
        raise ValueError("Recovery exceeds its original remaining or registered extra budget")
    anchor = {"epoch": anchor_epoch, "checkpoint_sha256": plan["anchor_sha256"],
              "source_collapsed": False, "finite_state_verified": True,
              "full_state_verified": True, "checkpoint_hash_verified": True,
              "job_data_runtime_provenance_verified": True, "strict_load_verified": True}
    protected_paths = [source_path, selection_path, checkpoint, baseline_path, anchor_review_path, terminal_path]
    protected_paths.append(source_root / "final/selected_checkpoint.pt")
    # Delivery may legitimately advance status after frozen training. Preserve
    # its terminal snapshot in the receipt; scientific artifacts remain locked.
    protected_paths += [path for path in [source_root / "agent_decisions.json",
                                        source_root / "action_scores.json"] if path.is_file()]
    protected_paths += list((source_root / "agent").glob("**/decision.json"))
    protected = {str(path): sha256(path) for path in protected_paths}
    protected.update(runtime_files)
    protected.update({str(plan_path): sha256(plan_path), plan["request"]: plan["request_sha256"],
                      plan["semantic"]: plan["semantic_sha256"]})
    child = copy.deepcopy(source_job)
    for key in ["resume", "bounded_recovery_plan", "recovery_policy", "stopped_after_epoch",
                "research_epoch_extension", "research_epoch_budget"]:
        child.pop(key, None)
    child["resume"] = resume
    child["planet_decision_reference"] = False
    child["branch_experiment"] = {"enabled": False, "max_per_tile": 0}
    child["bounded_recovery"] = {"protocol": PROTOCOL, "plan": str(plan_path),
                "plan_sha256": sha256(plan_path), "restore_epoch": anchor_epoch,
                "additional_epochs": additional, "original_max_epoch": original,
                "original_stop_epoch": stopped, "max_agent_submissions": submissions,
                "max_model_calls": model_calls}
    # Pin the current maintained runtime separately from the historical runtime.
    # This is an explicit recovery lineage, not a bitwise historical replay.
    current = {str(path.resolve()): sha256(path) for path in HERE.glob("*.py")}
    current.update({str(path.resolve()): sha256(path)
                    for path in (HERE.parent / ".codex/skills/cashew-fada-training-expert").rglob("*") if path.is_file()})
    current.update({str(path.resolve()): sha256(path)
                    for path in (HERE.parent / "references").glob("*.json")})
    child["recovery_lineage"] = {
        "source_job": str(source_path), "source_job_sha256": plan["source_job_sha256"],
        "source_output_root": str(source_root),
        "source_runtime_provenance": copy.deepcopy(source_job.get("runtime_provenance", {})),
        "source_research_design": copy.deepcopy(source_job.get("research_design")),
        "first_strategy_semantic_sha256": plan["semantic_sha256"],
        "subsequent_decisions": "normal_online_reviews_and_separate_SELECT",
        "matched_causal_experiment": False,
    }
    child["research_design"] = {
        "kind": PROTOCOL, "source_job_sha256": plan["source_job_sha256"],
        "matched_causal_experiment": False,
    }
    child["runtime_provenance"] = {
        "schema_version": 1, "files": current,
        **{key: value for key, value in source_job.get("runtime_provenance", {}).items()
           if key in {"source_pretrained_sha256", "source_train_sha256", "source_valid_sha256",
                      "target_full_list_sha256", "target_train_list_sha256"}},
    }
    protected.update(current)
    return {"plan": plan, "source_root": source_root, "output_root": output, "anchor": anchor,
            "strategy": strategy, "decision": decision, "child_job": child,
            "protected": protected, "restore_proof": proof,
            "original_remaining_epochs": original - stopped}


def bind_lineage_root(child, root):
    """Record the same isolated locations the launcher actually uses."""
    child["output_root"] = str(root)
    child["thread_key"] = child["run_id"]
    child["agent_thread_root"] = str(root / "agent_threads")
    child["agent_experience_root"] = str(root / "training_experience")
    child["runtime_root"] = str(HERE.parent.resolve())


def outcome(child):
    import train_expert_v4 as worker
    root = Path(child["output_root"])
    status = read_json(root / "status.json") if (root / "status.json").is_file() else {}
    start, cap = child["resume"]["epoch"], child["bounded_recovery"]["additional_epochs"]
    observed = status.get("stop_epoch", status.get("heartbeat_epoch", start))
    if type(observed) is not int or not start <= observed <= start + cap:
        raise ValueError("Worker recovery cost is missing or outside the reserved loop")
    completed = observed - start
    terminal = status.get("state") == "training_complete"
    delivery = {}
    if not terminal or completed == 0:
        return completed, False, False, delivery, cap
    checkpoint = root / ("checkpoints/epoch_%03d.pt" % observed)
    payload = worker.load_resumable_checkpoint(checkpoint)
    finite = _finite(payload)
    review = read_json(root / ("review_epoch_%03d.json" % observed))["candidate"]
    baseline = float(read_json(root / "review_epoch_000.json")["candidate"]["source_class2_f1"])
    floor = max(worker.SOURCE_COLLAPSE_ABSOLUTE_F1, baseline - worker.SOURCE_COLLAPSE_DROP_FROM_BASELINE)
    f1 = float(review["source_class2_f1"])
    collapsed = not math.isfinite(f1) or f1 < floor
    selected = read_json(root / "final/selection.json")["selected_epoch"]
    selected_path = root / "final/selected_checkpoint.pt"
    selected_payload = torch_load(selected_path, "cpu")
    selected_review = read_json(root / ("review_epoch_%03d.json" % selected))["candidate"]
    selected_f1 = float(selected_review["source_class2_f1"])
    if (payload["epoch"] != observed or selected_payload["epoch"] != selected
            or not _finite(selected_payload) or not math.isfinite(selected_f1) or selected_f1 < floor):
        raise ValueError("Selected recovery delivery is not a finite healthy checkpoint")
    delivery = {"epoch": selected, "checkpoint": str(selected_path.resolve()),
                "checkpoint_sha256": sha256(selected_path), "checkpoint_hash_verified": True,
                "outcome_epoch": observed, "outcome_checkpoint_sha256": sha256(checkpoint),
                "outcome_checkpoint_verified": finite}
    return completed, collapsed, finite, delivery, completed


def fallback(prepared, machine):
    """Keep the frozen original fallback in a new delivery root."""
    import shutil
    source = prepared["source_root"]
    selection = read_json(source / "final/selection.json")
    epoch = selection["selected_epoch"]
    checkpoint = source / "final/selected_checkpoint.pt"
    payload = torch_load(checkpoint, "cpu")
    review = read_json(source / ("review_epoch_%03d.json" % epoch))["candidate"]
    baseline = float(read_json(source / "review_epoch_000.json")["candidate"]["source_class2_f1"])
    import train_expert_v4 as worker
    floor = max(worker.SOURCE_COLLAPSE_ABSOLUTE_F1, baseline - worker.SOURCE_COLLAPSE_DROP_FROM_BASELINE)
    f1 = float(review["source_class2_f1"])
    if (type(epoch) is not int or epoch < 0 or payload["epoch"] != epoch
            or not _finite(payload) or not math.isfinite(f1) or f1 < floor):
        raise ValueError("Frozen original fallback is not a finite noncollapsed checkpoint")
    candidate = {"epoch": epoch, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                 "source_collapsed": False, "finite_state_verified": True, "checkpoint_hash_verified": True}
    machine.fallback(reason="Recovery attempt ended without a verified usable trained result", verified_candidate=candidate)
    root = prepared["output_root"] / "fallback_delivery"
    final = root / "final"
    final.mkdir(parents=True)
    shutil.copy2(checkpoint, final / "selected_checkpoint.pt")
    shutil.copy2(source / "final/selection.json", final / "selection.json")
    child = copy.deepcopy(prepared["child_job"])
    for name in ["resume", "bounded_recovery"]:
        child.pop(name, None)
    child["run_id"] += "__recovery_fallback"
    bind_lineage_root(child, root)
    atomic_json(root / "job.json", child)
    status = read_json(source / "status.json")
    status.update({"state": "training_complete", "selected_epoch": epoch,
                   "delivery_kind": machine.state["delivery"]["kind"], "adapted": epoch > 0})
    atomic_json(root / "status.json", status)
    from healthy_recovery_runtime_v4 import delivery_record
    record = delivery_record(selected_epoch=epoch, selected_checkpoint=final / "selected_checkpoint.pt",
        source_collapsed=True, stop_reason=selection["stop_reason"],
        stop_epoch=selection["stop_epoch"], original_max_epoch=child["max_epoch"])
    record["recovery"].update({"state": "FALLBACK_DELIVERED", "receipt": str(prepared["output_root"] / "recovery_state.json"),
                              "reason": machine.state["delivery"]["reason"]})
    atomic_json(final / "delivery.json", record)
    return root / "job.json"


def deliver_state(state, plan, gpu):
    from finalize_v4 import main as finalize
    finalize(Path(state["delivery_job"]), "cuda:" + str(gpu),
             prediction_only=not bool(plan.get("posthoc_score", False)))


def execute(plan_path, gpu, *, deliver=False, slots_held=False):
    """Run at most one reserved attempt; retries require another explicit turn."""
    from run_year_country_v4 import (acquire_gpu_slot, acquire_machine_slot, acquire_job_lock,
        release_lease, launch_worker, terminate_process_group)
    prepared = prepare(plan_path)
    plan, root = prepared["plan"], prepared["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "recovery.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "recovery_state.json"
        machine = BoundedRecovery(source_collapsed=True, max_attempts=plan["max_attempts"],
            max_extra_epochs=plan["max_extra_epochs"],
            original_remaining_epochs=prepared["original_remaining_epochs"], anchor=prepared["anchor"])
        if state_path.is_file():
            previous = read_json(state_path)
            if (previous["anchor"] != machine.state["anchor"] or previous["epoch_limit"] != machine.state["epoch_limit"]
                    or previous.get("plan_sha256") != sha256(Path(plan_path))
                    or previous["max_attempts"] != machine.state["max_attempts"]):
                raise ValueError("Recovery state/plan changed")
            verify_files(previous["protected"])
            if previous["status"] in {"HEALTHY_COMPLETED", "FALLBACK_DELIVERED"}:
                verify_files(prepared["protected"])
                if deliver:
                    deliver_state(previous, plan, gpu)
                return previous
            machine.state = previous
        machine.state["plan_sha256"] = sha256(Path(plan_path))
        machine.state["protected"] = prepared["protected"]
        machine.state.setdefault("source_terminal_status", read_json(prepared["source_root"] / "status.json"))
        verify_files(prepared["protected"])
        number = len(machine.state["attempts"]) + 1
        child = prepared["child_job"]
        child_root = root / ("attempt_%02d" % number)
        if child_root.exists():
            raise ValueError("Recovery attempt lineage already exists; reconcile it before retry")
        child["run_id"] = str(child["run_id"]) + "__recovery_%02d" % number
        bind_lineage_root(child, child_root)
        child["bounded_recovery"]["attempt"] = number
        ceiling_for_job(child)
        from formal_v4 import validate_formal_job
        if "formal_training" in child:
            validate_formal_job(child)
        child_path = child_root / "job.json"
        item = {"job_id": child["job_id"], "job": str(child_path), "output_root": str(child_root),
                "thread_key": child["run_id"], "agent_thread_root": str(child_root / "agent_threads")}
        leases = []
        process = None
        try:
            leases.append(acquire_job_lock(item))
            if not slots_held:
                leases.append(acquire_gpu_slot(str(gpu), child["job_id"], 3))
                leases.append(acquire_machine_slot(child["job_id"], 24))
            if any(lease is None for lease in leases):
                raise RuntimeError("No GPU/machine slot available for this recovery attempt")
            machine.begin(additional_epochs=plan["additional_epochs"],
                          strategy=prepared["strategy"], verified_anchor=prepared["anchor"])
            atomic_json(state_path, machine.state)
            child_root.mkdir()
            atomic_json(child_path, child)
            atomic_json(child_root / "recovery_preflight.json",
                        {"restore_proof": prepared["restore_proof"], "protected": prepared["protected"],
                         "original_failure_root": str(prepared["source_root"]),
                         "normal_online_review_and_SELECT_preserved": True})
            process = launch_worker(item, str(gpu), root, number)
            try:
                code = process.wait(timeout=plan["timeout_seconds"])
            except subprocess.TimeoutExpired:
                terminate_process_group(process.pid)
                code = process.wait(timeout=10)
            finally:
                reap_worker(process)
            atomic_json(child_root / "launch_result.json",
                        {"pid": process.pid, "returncode": code, "worker_reaped": True})
            verify_files(prepared["protected"])
            completed, collapsed, finite, result, charge = outcome(child)
            if code != 0 or result.get("epoch") == 0:
                collapsed, finite, charge = True, False, plan["additional_epochs"]
            machine.finish(completed_epochs=completed, worker_reaped=True,
                source_collapsed=collapsed, finite_state=finite, delivery=result, charged_epochs=charge)
            if machine.state["status"] == "HEALTHY_COMPLETED":
                record_path = child_root / "final/delivery.json"
                record = read_json(record_path)
                record["delivery_kind"] = machine.state["delivery"]["kind"]
                record["recovery"].update({"state": "HEALTHY_COMPLETED", "receipt": str(state_path)})
                atomic_json(record_path, record)
                machine.state["delivery_job"] = str(child_path)
            else:
                machine.state["delivery_job"] = str(fallback(prepared, machine))
            atomic_json(state_path, machine.state)
            verify_files(prepared["protected"])
            if deliver:
                deliver_state(machine.state, plan, gpu)
            return machine.state
        finally:
            if process is not None and process.poll() is None:
                reap_worker(process)
            for lease in reversed(leases):
                release_lease(lease)


def configured_recovery(item, gpu):
    """Supervisor hook, used only for an explicitly pinned recovery plan."""
    job = read_json(item["job"])
    # Attach an amendment to the launch manifest after the failure is frozen;
    # putting a plan hash into its own source job creates a hash dependency cycle.
    configured = item.get("bounded_recovery_plan") or job.get("bounded_recovery_plan")
    if not configured:
        if job.get("recovery_policy", {}).get("version") != "automatic_bounded_source_repair_v2":
            return None
        from automatic_recovery_v2 import register
        try:
            configured = register(Path(item["job"]))
        except Exception as exc:
            # Failed registration is explicit; deliver the original fallback.
            atomic_json(Path(item["output_root"]) / "automatic_recovery_failure.json",
                        {"state": "REGISTRATION_FAILED", "error_type": type(exc).__name__, "error": str(exc),
                         "original_checkpoint_preserved": True})
            return {"status": "REGISTRATION_FAILED", "error": str(exc)}
        if not configured:
            return None
        atomic_json(Path(item["output_root"]) / "recovery_plan_amendment.json", configured)
    plan_path = Path(configured["path"])
    if sha256(plan_path) != configured["sha256"]:
        raise ValueError("Configured recovery plan changed")
    if Path(read_json(plan_path)["source_job"]).resolve() != Path(item["job"]).resolve():
        raise ValueError("Configured recovery belongs to another stopped job")
    status = read_json(Path(item["output_root"]) / "status.json")
    if status.get("stop_reason") != "source_collapse_safety_exit":
        return None
    try:
        result = execute(plan_path, str(gpu), deliver=True, slots_held=True)
    except Exception as exc:
        atomic_json(Path(item["output_root"]) / "automatic_recovery_failure.json",
                    {"state": "EXECUTION_FAILED", "error_type": type(exc).__name__, "error": str(exc),
                     "original_checkpoint_preserved": True, "plan": configured})
        result = {"status": "EXECUTION_FAILED", "error": str(exc)}
    atomic_json(Path(item["output_root"]) / "recovery_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--deliver", action="store_true")
    args = parser.parse_args()
    if args.gpu is None:
        result = prepare(args.plan)
        print(json.dumps({"state": "PREPARED_NO_TRAINING_OR_AGENT_CALL",
                          "restore_proof": result["restore_proof"], "strategy": result["strategy"]}, indent=2))
    else:
        print(json.dumps(execute(args.plan, args.gpu, deliver=args.deliver), indent=2))
