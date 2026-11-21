"""Execute the tile expert's self-contained diagnostic script and archive evidence."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from protocol import sha256


def _terminate_residual_process_group(pgid: int) -> None:
    """A custom diagnostic is foreground-only; kill descendants it left behind."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return
    os.killpg(pgid, signal.SIGTERM)
    for _ in range(50):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        __import__("time").sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_custom_branch(job, checkpoint, experiment, device, current_review):
    from train_expert_v4 import atomic
    epoch = int(current_review["epoch"])
    source = Path(job["output_root"]) / "agent" / f"epoch_{epoch:03d}" / "custom_experiment.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    source_code = source.read_bytes()
    signature = hashlib.sha256(source_code + json.dumps(experiment, sort_keys=True).encode()).hexdigest()[:12]
    root = Path(job["output_root"]) / "experiments" / f"epoch_{epoch:03d}" / signature
    root.mkdir(parents=True, exist_ok=True)
    comparison = root / "comparison.json"
    if comparison.is_file():
        return json.loads(comparison.read_text())
    # An uncertain execution must be inspected, not blindly submitted twice.
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("state") == "running":
        raise RuntimeError(f"Unresolved custom experiment; reconcile {manifest_path}")
    script = root / "custom_experiment.py"
    script.write_bytes(source_code)
    result_path = root / "result.json"
    tracked = [checkpoint, Path(job["output_root"]) / "final/selected_checkpoint.pt"]
    before = {str(p): sha256(p) for p in tracked if p.is_file()}
    context = root / "context.json"
    atomic(context, {"job": job, "checkpoint": str(checkpoint.resolve()), "review": current_review,
                     "experiment": experiment, "device": str(device), "seed": 314159,
                     "output_root": str(root.resolve()), "formal_checkpoint_hashes": before})
    command = [sys.executable, str(script.resolve()), "--context", str(context.resolve()), "--output", str(result_path.resolve())]
    manifest = {"state": "running", "command": command, "script_sha256": sha256(script),
                "source_script": str(source.resolve()), "checkpoint_sha256": sha256(checkpoint)}
    atomic(manifest_path, manifest)
    try:
        with (root / "execution.log").open("w") as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            atomic(manifest_path, {**manifest, "pid": process.pid})
            try:
                code = process.wait(timeout=int(job.get("branch_experiment", {}).get("timeout_seconds", 1800)))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError("custom experiment exceeded execution budget")
            finally:
                _terminate_residual_process_group(process.pid)
        if code != 0:
            raise RuntimeError(f"custom experiment exit={code}; see {root / 'execution.log'}")
        result = json.loads(result_path.read_text())
        json.dumps(result, allow_nan=False)
        if result.get("state") != "completed" or not result.get("measurements") or not result.get("limitations"):
            raise ValueError("custom result needs state=completed, measurements and limitations")
        if {p: sha256(Path(p)) for p in before} != before:
            raise RuntimeError("custom experiment changed a formal checkpoint")
        completed = {"state": "completed", "experiment": experiment, "custom_result": result,
                     "main_checkpoint_replaced": False, "formal_checkpoint_hashes_verified": True,
                     "script_sha256": manifest["script_sha256"], "artifacts": str(root.resolve()),
                     "interpretation_limit": "custom diagnostic; inspect methodology and limitations before adoption"}
        atomic(comparison, completed)
        atomic(manifest_path, {**manifest, "state": "completed"})
        return completed
    except Exception as exc:
        unchanged = all(Path(p).is_file() and sha256(Path(p)) == digest for p, digest in before.items())
        failure = {"state": "failed", "experiment": experiment, "error_type": type(exc).__name__,
                   "error": str(exc), "main_checkpoint_replaced": not unchanged, "artifacts": str(root.resolve())}
        atomic(root / "failure.json", failure)
        atomic(manifest_path, {**manifest, **failure})
        if not unchanged:
            raise RuntimeError("formal checkpoint integrity failed; stop for recovery") from exc
        return failure
