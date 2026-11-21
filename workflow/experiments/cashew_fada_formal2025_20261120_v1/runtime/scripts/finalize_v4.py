#!/usr/bin/env python3
"""Deliver the frozen selected checkpoint, then optionally score post-hoc."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluate_planet_v4 import evaluate
from formal_v4 import max_epoch_for_job
from healthy_recovery_runtime_v4 import delivery_record
from predict_raw_argmax_v4 import predict
from protocol import atomic_json, sha256
from train_tile import torch_load


DELIVERY_READY_STATES = {"training_complete", "prediction_complete"}


def protected_identities(job_path, root):
    paths = [job_path, root / "final/selected_checkpoint.pt", root / "final/selection.json"]
    paths += [root / name for name in ["agent_decisions.json", "action_scores.json"]]
    paths += list((root / "agent").glob("**/decision.json"))
    return {str(path.resolve()): sha256(path) for path in paths if path.is_file()}


def assert_protected(identities):
    for name, expected in identities.items():
        path = Path(name)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError("Frozen training/selection artifact changed during delivery: " + name)


def verified_prediction_outputs(root, selected_hash, selected_epoch, raw, scores):
    """Bind every delivered product to the actual frozen weights and file bytes."""
    selected = (root / "final/selected_checkpoint.pt").resolve()
    outputs = {}
    for qa in [raw, scores]:
        if (qa.get("checkpoint_sha256") != selected_hash
                or type(qa.get("selected_epoch")) is not int
                or qa.get("selected_epoch") != selected_epoch
                or Path(qa.get("checkpoint", "")).resolve() != selected):
            raise ValueError("Prediction QA checkpoint/epoch does not match frozen selection")
    if (raw.get("planet_prior_applied_to_logits") is not False
            or raw.get("fada_discriminator_loaded_for_inference") is not False
            or scores.get("pre_argmax") is not True):
        raise ValueError("Prediction mode violates raw-argmax/pre-argmax delivery contract")
    for qa, name, hash_key in [
            (raw, "four_class_tif", "four_class_sha256"),
            (raw, "only_class2_tif", "only_class2_sha256"),
            (raw, "kmz", "kmz_sha256"),
            (scores, "softmax_score_tif", "softmax_score_tif_sha256")]:
        path = Path(qa.get(name, "")).resolve()
        try:
            path.relative_to((root / "final/outputs").resolve())
        except ValueError:
            raise ValueError("Prediction artifact is outside this delivery: " + name)
        if (not path.is_file() or not qa.get(hash_key)
                or sha256(path) != qa[hash_key]):
            raise ValueError("Prediction artifact is missing, changed or outside this delivery: " + name)
        outputs[name] = {"path": str(path), "sha256": qa[hash_key]}
    return outputs


def main(job_path: Path, device: str, *, prediction_only=False) -> dict:
    job_path = Path(job_path)
    job = json.loads(job_path.read_text())
    root = Path(job["output_root"])
    final = root / "final"
    status_path = root / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("state") not in DELIVERY_READY_STATES | {"evaluation_complete"}:
        raise ValueError("Deliver only a completed, frozen training result")
    selection_path = final / "selection.json"
    selection = json.loads(selection_path.read_text())
    selected = final / "selected_checkpoint.pt"
    selected_epoch = selection["selected_epoch"]
    payload = torch_load(selected, "cpu")
    if (type(selected_epoch) is not int or selected_epoch < 0
            or type(payload.get("epoch")) is not int or payload["epoch"] != selected_epoch
            or status.get("selected_epoch") != selected_epoch):
        raise ValueError("Selection/status/checkpoint epoch mismatch")
    selected_hash = sha256(selected)
    protected = protected_identities(job_path, root)
    delivery_path = final / "delivery.json"
    if delivery_path.is_file():
        delivery = json.loads(delivery_path.read_text())
        if (delivery.get("checkpoint_sha256") != selected_hash
                or delivery.get("selected_epoch") != selected_epoch
                or delivery.get("adapted") is not (selected_epoch > 0)
                or Path(delivery.get("checkpoint", "")).resolve() != selected.resolve()):
            raise ValueError("Existing delivery identity does not match selected checkpoint")
    else:
        # Older completed jobs have no delivery record. Describe their real
        # frozen selection without changing their STOP/SELECT or weights.
        stop_reason = selection["stop_reason"]
        delivery = delivery_record(
            selected_epoch=selected_epoch, selected_checkpoint=selected,
            source_collapsed=stop_reason == "source_collapse_safety_exit",
            stop_reason=stop_reason, stop_epoch=selection["stop_epoch"],
            original_max_epoch=max_epoch_for_job(job), recovery_policy=job.get("recovery_policy"),
        )
    allowed_kinds = {"trained_selected", "trained_fallback", "trained_recovery"} if selected_epoch else {"unadapted_fallback"}
    if delivery.get("delivery_kind") not in allowed_kinds or not isinstance(delivery.get("recovery"), dict):
        raise ValueError("Delivery kind/recovery state contradicts the actual trained epoch")

    prediction_done = delivery.get("prediction_complete") is True
    if prediction_done:
        raw = json.loads((final / "outputs/raw_argmax/prediction_raw_argmax.json").read_text())
    else:
        raw = predict(job_path, device, 16)
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("predict_softmax_scores_v4.py")),
             "--job", str(job_path), "--device", device], check=True,
        )
    scores_path = final / "outputs/raw_argmax/softmax_scores_qa.json"
    scores = json.loads(scores_path.read_text())
    outputs = verified_prediction_outputs(root, selected_hash, selected_epoch, raw, scores)
    assert_protected(protected)
    if prediction_done and delivery.get("prediction_outputs") != outputs:
        raise ValueError("Previously delivered prediction identities changed")
    if status.get("state") == "evaluation_complete":
        if not prediction_done:
            raise ValueError("Completed evaluation has no verified prediction receipt")
        return delivery
    delivery.update({
        "prediction_complete": True, "prediction_outputs": outputs,
        "frozen_selection_sha256": protected[str(selection_path.resolve())],
        "target_quality_proven": False,
    })
    atomic_json(delivery_path, delivery)
    status.update({
        "job_id": job["job_id"], "run_id": job["run_id"],
        "state": "prediction_complete", "prediction": raw["four_class_tif"],
        "delivery_kind": delivery["delivery_kind"], "adapted": delivery["adapted"],
        "recovery_state": delivery["recovery"]["state"], "prediction_complete": True,
        "selected_checkpoint_sha256": selected_hash,
    })
    atomic_json(status_path, status)
    if not prediction_only:
        evaluation = evaluate(job_path)
        assert_protected(protected)
        if (evaluation.get("prediction_sha256") != outputs["four_class_tif"]["sha256"]
                or Path(evaluation.get("prediction", "")).resolve() != Path(raw["four_class_tif"]).resolve()
                or evaluation.get("used_for_training_or_selection") is not False):
            raise ValueError("Post-hoc score is not bound to the frozen prediction")
        status.update({"state": "evaluation_complete", "planet_posthoc_f1": evaluation["f1"]})
        atomic_json(status_path, status)
    return delivery

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--prediction-only", action="store_true")
    args = parser.parse_args()
    main(args.job, args.device, prediction_only=args.prediction_only)
