"""Install the research workflow and its bound skills in a fresh workspace."""

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile, rmtree


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "workflow"
APP = Path("experiments/cashew_fada_formal2025_20261120_v1")
ACCOUNT = Path(".codex_training_current_account")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_slots(value):
    try:
        slots = {str(int(k)): int(v) for k, v in (item.split(":") for item in value.split(","))}
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a GPU:slots list such as 0:1,1:2") from error
    if not slots or any(int(gpu) < 0 or cap < 1 for gpu, cap in slots.items()):
        raise argparse.ArgumentTypeError("GPU indices must be nonnegative and slot counts positive")
    return slots


def install(args):
    workspace, assets = args.workspace.resolve(), args.assets.resolve(strict=True)
    if not assets.is_dir():
        raise ValueError("The asset root must be a directory")
    if any(c in str(p) for p in (workspace, assets) for c in ('"', "'", "\\", "\n", "\r")):
        raise ValueError("Use directory names without quotes, backslashes, or line breaks")
    if workspace.exists() or workspace == REPO or REPO in workspace.parents:
        raise ValueError("Choose a fresh workspace outside this repository")
    workspace.mkdir(parents=True, mode=0o700)
    try:
        original = read_json(SOURCE / "provenance.json")["sha256_before_relocation"]
        installed = {}
        for source in sorted(SOURCE.rglob("*")):
            if not source.is_file() or source.name == "provenance.json":
                continue
            if source.suffix not in {".py", ".txt", ".csv", ".json", ".yaml", ".md"}:
                continue
            target = workspace / source.relative_to(SOURCE)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix in {".txt", ".csv"}:
                copyfile(source, target)
            else:
                body = source.read_text(encoding="utf-8")
                body = body.replace("@WORKSPACE@", str(workspace)).replace("@ASSETS@", str(assets))
                target.write_text(body, encoding="utf-8")
            installed[source.relative_to(REPO).as_posix()] = target

        root = workspace / APP
        plan_path = root / "cohort_2025.json"
        plan = read_json(plan_path)
        if args.all_groups:
            plan["jobs"].extend(read_json(root / "config/extension_2025.json"))
        if args.single_host:
            for job in plan["jobs"]:
                job["host"] = "local"
            plan.pop("prediction_gpu_overrides", None)
        if args.slots:
            plan["hosts"][args.host]["gpu_slots"] = args.slots
            selected_jobs = {j["name"] for j in plan["jobs"] if j["host"] == args.host}
            plan["prediction_gpu_overrides"] = {name: gpu
                for name, gpu in plan.get("prediction_gpu_overrides", {}).items()
                if name not in selected_jobs or str(gpu) in args.slots}
        plan["hosts"][args.host]["sam_unit"] = args.sam_unit
        write_json(plan_path, plan)

        # Retie the runtime integrity checks to path-rendered model code and its registry.
        digests = {digest: sha(installed[name]) for name, digest in original.items()
                   if name in installed and (name.startswith("workflow/code/") or name.endswith("/model_registry.json"))}
        scripts = root / "runtime/scripts"
        for script in (scripts / "train_tile.py", *scripts.glob("source_panel_attestation*.py")):
            body = script.read_text(encoding="utf-8")
            for upstream, local in digests.items():
                if upstream != local:
                    body = body.replace(upstream, local)
            script.write_text(body, encoding="utf-8")

        skills = workspace / ACCOUNT / "skills"
        training = skills / "cashew-fada-training-expert"
        pins = read_json(root / "skill_identity.json")
        pins["skill_sha256"] = sha(training / "SKILL.md")
        pins["verifier_sha256"] = sha(training / "scripts/verify_explicit_skill_binding.py")
        pins["reference_hashes"] = {name: sha(training / "references" / name)
                                    for name in pins["reference_hashes"]}
        write_json(root / "skill_identity.json", pins)
        annotation = skills / "cashew-sample-assist"
        version_path = annotation / "current_reference_version.json"
        version = read_json(version_path)
        version["version_skill_sha256"] = sha(annotation / "SKILL.md")
        write_json(version_path, version)
        pins = read_json(root / "annotation_identity.json")
        pins["files"] = {p.relative_to(annotation).as_posix(): sha(p)
                         for p in annotation.rglob("*") if p.is_file()}
        write_json(root / "annotation_identity.json", pins)
        workspace.joinpath(ACCOUNT).chmod(0o700)

        # Registration and dispatch verify these files before using any inputs.
        manifest = {p.relative_to(root).as_posix(): sha(p)
                    for p in root.rglob("*") if p.is_file()}
        write_json(root / "runtime_identity.json",
                   {"version": "cashew_fada_formal2025_v1", "files": manifest})
    except BaseException:
        rmtree(workspace)
        raise
    print(f"Workspace: {workspace}")
    print(f"Run directory: {root}")
    print(f"Codex account directory: {workspace / ACCOUNT}")
    print(f"Group definitions: {len(plan['jobs'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--all-groups", action="store_true")
    parser.add_argument("--single-host", action="store_true")
    parser.add_argument("--host", choices=("local", "124"), default="local")
    parser.add_argument("--slots", type=gpu_slots)
    parser.add_argument("--sam-unit")
    install(parser.parse_args())


if __name__ == "__main__":
    main()
