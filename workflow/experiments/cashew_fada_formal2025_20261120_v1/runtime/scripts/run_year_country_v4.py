#!/usr/bin/env python3
"""Prepare and launch v4 FADA jobs by target year and country.

Examples
--------
    python run_year_country_v4.py --year 2025 --country TOGO
    python run_year_country_v4.py --year 2019 --country TOGO --tile 000000000000000001fa

Without ``--tile`` every tile directory under the requested country is queued.
The supervisor honors the requested slots up to three per GPU and 24 total,
spreading jobs across available GPUs before stacking multiple jobs per card.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from select_tile_specific_supplementary import read_names, select_supplementary
from formal_v4 import load_template, template_identity, validate_formal_job
from runtime_account_v4 import ACCOUNT_ENV_VAR, codex_home


HERE = Path(__file__).resolve().parent
SKILL_ROOT = HERE.parent
REGISTRY = SKILL_ROOT / "references/model_registry.json"
DATA_ROOT = Path("@ASSETS@/alphaearth_data_split")
PLANET_ROOT = Path("@WORKSPACE@/planetmap/antile")
DEFAULT_OUTPUT_ROOT = Path("@WORKSPACE@/fada_expert_runs/year_country_v4")
SUPPLEMENTARY_CANDIDATE_NAMES = Path(
    "@WORKSPACE@/fada_expert_runs/20260916_supplementary_all_1012/data/"
    "supplementary_segmentation_only_all_1012.txt"
)
SUPPLEMENTARY_IMAGE_ROOT = Path("@ASSETS@/code/mydataset2020/newdataset/supplementary_images")
SUPPLEMENTARY_LABEL_ROOT = Path("@ASSETS@/code/mydataset2020/newdataset/supplementary_labels")
FORMAL_TEMPLATE = load_template()
MAX_EPOCH = FORMAL_TEMPLATE["job_defaults"]["max_epoch"]
MINIMUM_STOP_EPOCH = FORMAL_TEMPLATE["job_defaults"]["minimum_stop_epoch"]
REVIEW_INTERVAL = FORMAL_TEMPLATE["job_defaults"]["review_interval"]
TARGET_TRAIN_LIMIT = 1000
RSENV_PYTHON = Path("python")
# Inputs are read directly from NAS; there is no /dev/shm staging step.
# Admission ceilings are three tasks per GPU and 24 machine-wide. These are
# upper bounds; the supervisor fills idle GPUs first and stops admitting work
# if the operator observes NAS degradation.
NAS_SAFE_MAX_WORKERS_PER_GPU = 3
MAX_TOTAL_CONCURRENT_TRAINING_TASKS = 24
SUPERVISOR_POLL_SECONDS = 15
SUPERVISOR_DURABLE_HEARTBEAT_SECONDS = 300
MAX_RESTARTS_PER_JOB = 2
LEASE_ROOT = Path("/tmp/cashew_fada_v4_leases")


def require_rsenv() -> str:
    """Return the one supported Python runtime for training and inference."""
    if not RSENV_PYTHON.is_file():
        raise FileNotFoundError(f"Required rsenv Python executable is missing: {RSENV_PYTHON}")
    return str(RSENV_PYTHON)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def _lease_payload(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _lease_is_live(path: Path) -> bool:
    value = _lease_payload(path)
    pid = int(value.get("pid", 0) or 0)
    ticks = value.get("process_start_ticks")
    return bool(pid and process_alive(pid) and (ticks is None or process_start_ticks(pid) == int(ticks)))


def acquire_job_lock(item: dict) -> Path:
    """Atomically claim one manifest job across all v4 supervisors."""
    LEASE_ROOT.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(Path(item["job"]).resolve()).encode()).hexdigest()
    path = LEASE_ROOT / f"job_{key}.json"
    if path.exists() and not _lease_is_live(path):
        path.unlink(missing_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError({"job_already_owned_by_another_supervisor": item["job_id"], "lease": str(path)}) from exc
    with os.fdopen(fd, "w") as handle:
        json.dump({"job_id": item["job_id"], "job": str(Path(item["job"]).resolve()),
                   "pid": os.getpid(), "process_start_ticks": process_start_ticks(os.getpid()),
                   "created_at": datetime.now(timezone.utc).isoformat()}, handle)
    return path


def acquire_gpu_slot(gpu: str, job_id: str, slots_per_gpu: int) -> Path | None:
    """Acquire a machine-wide atomic GPU slot; stale supervisor leases expire."""
    return _acquire_slot_lease("gpu_" + str(gpu) + "_slot", job_id, slots_per_gpu,
                               gpu=str(gpu))


def _acquire_slot_lease(prefix: str, job_id: str, max_slots: int,
                        gpu: str | None = None) -> Path | None:
    """Serialize stale cleanup and lease creation across local supervisors."""
    LEASE_ROOT.mkdir(parents=True, exist_ok=True)
    guard_path = LEASE_ROOT / ".slot-allocation.lock"
    with guard_path.open("a") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
        for index in range(max_slots):
            path = LEASE_ROOT / f"{prefix}_{index}.json"
            if path.exists() and not _lease_is_live(path):
                path.unlink(missing_ok=True)
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w") as handle:
                payload = {"job_id": job_id, "slot": index, "pid": os.getpid(),
                           "process_start_ticks": process_start_ticks(os.getpid()),
                           "created_at": datetime.now(timezone.utc).isoformat()}
                if gpu is not None:
                    payload["gpu"] = gpu
                json.dump(payload, handle)
            return path
    return None


def acquire_machine_slot(job_id: str, max_slots: int = MAX_TOTAL_CONCURRENT_TRAINING_TASKS) -> Path | None:
    """Acquire a machine-wide atomic task slot shared by all supervisors."""
    return _acquire_slot_lease("machine_slot", job_id, max_slots)


def release_lease(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def normalized_country(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    aliases = {
        "cote_divoire": "Ivoire",
        "cote_d_ivoire": "Ivoire",
        "ivory_coast": "Ivoire",
        "sierra_leone": "Sierra_Leone",
        "guinea_bissau": "Guinea-Bissau",
    }
    if key in aliases:
        return aliases[key]
    root = DATA_ROOT / "2025"
    for path in sorted(root.iterdir()):
        if path.is_dir() and re.sub(r"[^a-z0-9]+", "_", path.name.casefold()).strip("_") == key:
            return path.name
    raise ValueError({
        "unknown_country": value,
        "available": sorted(path.name for path in root.iterdir() if path.is_dir()),
    })


def load_source_profile(target_year: int) -> tuple[str, dict]:
    registry = json.loads(REGISTRY.read_text())
    try:
        profile_name = registry["automatic_selection"][str(int(target_year))]
        profile = dict(registry["profiles"][profile_name])
    except KeyError as exc:
        raise ValueError(f"no registered source profile for target year {target_year}") from exc
    if int(target_year) not in {int(year) for year in profile["compatible_target_years"]}:
        raise RuntimeError(f"source profile {profile_name} is not compatible with {target_year}")
    if int(profile["source_year"]) != 2019:
        raise RuntimeError(
            "production v4 requires the validated 2019 Benin source profile; "
            f"registry resolved source_year={profile['source_year']}"
        )
    pretrained = Path(profile["pretrained"])
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    if sha256(pretrained) != profile["pretrained_sha256"]:
        raise RuntimeError(f"epoch173 checkpoint hash mismatch: {pretrained}")
    train = Path(profile["source_train"])
    valid = Path(profile["source_valid"])
    for split, path in (("train", train), ("valid", valid)):
        expected = int(profile[f"source_{split}_expected_count"])
        actual = len([x for x in path.read_text().splitlines() if x.strip()])
        if actual != expected:
            raise RuntimeError(f"source {split} list count mismatch: {actual} != {expected}: {path}")
        if sha256(path) != profile[f"source_{split}_sha256"]:
            raise RuntimeError(f"source {split} list hash mismatch: {path}")
    profile["entrypoint_pretrained_epoch"] = 173
    return profile_name, profile


def validate_supplementary(names_file: Path | None = None) -> dict:
    """Validate the candidate pool; training lists are selected per tile."""
    names_path = Path(names_file or SUPPLEMENTARY_CANDIDATE_NAMES)
    if not names_path.is_file():
        raise FileNotFoundError(names_path)
    names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
    missing_images = [name for name in names if not (SUPPLEMENTARY_IMAGE_ROOT / name).is_file()]
    missing_labels = []
    for name in names:
        stem = Path(name).stem
        if not stem.startswith("AlphaEarth_2025_"):
            raise ValueError(f"unexpected supplementary name: {name}")
        label = SUPPLEMENTARY_LABEL_ROOT / f"label_processed_{stem[len('AlphaEarth_2025_'):]}.tif"
        if not label.is_file():
            missing_labels.append(str(label))
    if missing_images or missing_labels:
        raise FileNotFoundError({"missing_images": missing_images[:5], "missing_labels": missing_labels[:5]})
    return {
        "enabled": True,
        "image_root": str(SUPPLEMENTARY_IMAGE_ROOT.resolve()),
        "label_root": str(SUPPLEMENTARY_LABEL_ROOT.resolve()),
        "candidate_names_file": str(names_path.resolve()),
        "candidate_count": len(names),
        "loss": "continuous_known_nonclass2_suppression",
        "mode": "class2_suppression",
        "loss_weight": float(FORMAL_TEMPLATE["supplementary"]["loss_weight"]),
        "hard_threshold": 0.5,
        "spatial_augmentation": {
            "enabled": True,
            "transforms": ["rot90", "horizontal_flip", "vertical_flip"],
            "applies_to": "supplementary_image_and_label_pair_only",
        },
        "update_interval_steps": 1,
        "max_occurrences_per_epoch": None,
        "enters": "segmentation_classifier_only",
        "excluded_from": ["local_fada_discriminator", "global_fada_discriminator", "domain_adaptation_loss"],
    }


def select_tile_specific_supplementary(tile_dir: Path, target_names: list[str],
                                       data_root: Path, candidate_pool: dict, source: dict | None = None) -> dict:
    """Materialize one auditable, variable-size similarity-filtered list."""
    names_file = data_root / "supplementary_tile_specific.txt"
    report_file = data_root / "supplementary_tile_specific_report.json"
    selection = select_supplementary(
        target_root=tile_dir, target_names=target_names,
        candidate_root=Path(candidate_pool["image_root"]),
        candidate_names=read_names(Path(candidate_pool["candidate_names_file"])),
        output_list=names_file, report=report_file,
        geographic_radius_km=FORMAL_TEMPLATE.get("supplementary_geographic_filter", {}).get("radius_km"),
        source_profile=source if FORMAL_TEMPLATE.get("conditional_negative_policy") else None,
    )
    count = int(selection["selected_count"])
    empty_policy = FORMAL_TEMPLATE.get("supplementary_empty_policy", {})
    if count == 0 and empty_policy.get("version") != "allow_empty_v1":
        raise RuntimeError({"tile": tile_dir.name, "reason": "no_feature_similar_supplementary_negatives",
                            "report": str(report_file)})
    result = {
        **{key: value for key, value in candidate_pool.items()
           if key not in {"candidate_names_file", "candidate_count"}},
        "enabled": True, "names_file": str(names_file.resolve()),
        "count": count,
        "selection": {
            "method": selection["selection_rule"], "report": str(report_file.resolve()),
            "report_sha256": sha256(report_file),
            "candidate_names_file": candidate_pool["candidate_names_file"],
            "candidate_count": int(selection["candidate_count"]),
            "selected_count": int(selection["selected_count"]),
            "names_file_sha256": sha256(names_file),
            "fixed_top_k_used": False, "geographic_distance_used_for_filtering": selection["geographic_distance_used_for_filtering"],
            "geographic_filter": selection.get("geographic_filter"),
        },
    }
    if count == 0:
        result.update({key: empty_policy[key] for key in ("enabled", "loss_weight", "disabled_reason")})
    return result


def resolve_tiles(year: int, country: str, tile: str | None) -> list[Path]:
    parent = DATA_ROOT / str(year) / country
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    directories = sorted(path for path in parent.iterdir() if path.is_dir())
    if tile:
        requested = str(tile)
        matches = [path for path in directories if path.name == requested or path.name.endswith("_" + requested)]
        if len(matches) != 1:
            raise FileNotFoundError({"requested_tile": requested, "matches": [path.name for path in matches]})
        directories = matches
    if not directories:
        raise RuntimeError(f"no tile directories under {parent}")
    return directories


def tile_suffix(tile_dir: Path) -> str:
    if "_" not in tile_dir.name:
        raise ValueError(f"tile name has no id suffix: {tile_dir.name}")
    # Country names such as ``Burkina_Faso`` contain underscores; the tile ID
    # is the final underscore-delimited field.
    return tile_dir.name.rsplit("_", 1)[1]


def make_job(tile_dir: Path, year: int, country: str, output_root: Path,
             source_profile_name: str, source: dict, supplementary: dict,
             resume: dict | None = None, agent_thread_root: Path | None = None,
             job_name: str | None = None, thread_key: str | None = None) -> dict:
    # This legacy preparer selects old tile-specific supplementary labels and
    # does not construct the current human-polygon group or point-sealed
    # short-session contract. Refuse before creating files, rather than leave
    # an unusable half-prepared directory after formal validation fails.
    if FORMAL_TEMPLATE.get("polygon_supervision_policy", {}).get("version") == "human_polygon_group_formal_v1":
        raise RuntimeError(
            "Legacy single-tile preparer does not support the current human-polygon "
            "group and point-sealed contract; use the verified per-epoch group "
            "preparer. No job files were created."
        )
    names = sorted(path.name for path in tile_dir.glob("*.tif") if path.is_file())
    if not names:
        raise RuntimeError(f"tile has no TIFF patches: {tile_dir}")
    suffix = tile_suffix(tile_dir)
    planet = PLANET_ROOT / f"{suffix}.tif"
    if not planet.is_file():
        raise FileNotFoundError({"tile": tile_dir.name, "missing_planet_risk_map": str(planet)})
    train_count = min(TARGET_TRAIN_LIMIT, len(names))
    selected = sorted(np.random.RandomState(16).choice(np.asarray(names), train_count, replace=False).tolist())
    name = str(job_name or tile_dir.name)
    job_root = output_root / name
    if job_root.exists() and any(job_root.iterdir()):
        raise FileExistsError(f"refuse to overwrite existing job: {job_root}")
    data_root = job_root / "data"
    full_list = data_root / "target_full.txt"
    train_list = data_root / "target_train_seed16.txt"
    atomic_text(full_list, "\n".join(names) + "\n")
    atomic_text(train_list, "\n".join(selected) + "\n")
    supplementary = select_tile_specific_supplementary(tile_dir, names, data_root, supplementary, source)
    job = {
        "schema_version": 1,
        "protocol_version": "cashew-fada-training-expert-v4-year-country-entry",
        "job_id": name,
        "run_id": f"{year}_{country}/{name}",
        "thread_key": str(thread_key or f"{country}{year}_{suffix}"),
        "country": country,
        "target_year": int(year),
        "target_tile": tile_dir.name,
        "target_root": str(tile_dir.resolve()),
        "target_patch_count": len(names),
        "target_train_count": len(selected),
        "target_train_list": str(train_list.resolve()),
        "target_full_list": str(full_list.resolve()),
        "source_profile": source_profile_name,
        "source": source,
        "variant": {"local_mode": "soft", "confidence_cap": 0.9, "hard_threshold": None},
        "supplementary": supplementary,
        "output_root": str(job_root.resolve()),
        "max_epoch": MAX_EPOCH,
        "agentguide_policy": copy.deepcopy(FORMAL_TEMPLATE["job_defaults"]["agentguide_policy"]),
        "planet_decision_reference": False,
        "minimum_stop_epoch": MINIMUM_STOP_EPOCH,
        "review_interval": REVIEW_INTERVAL,
        "branch_experiment": {"enabled": True, "max_per_tile": 4, "allowed_steps": [32, 64, 128],
                              "policy": "one paired control/treatment at a time; branch weights never replace main"},
        "seed": 16,
        "planet_risk_map": str(planet.resolve()),
        "planet_training_policy": "Planet is advisory to Agent only; no logit prior; supplementary labels are segmentation-only weighted negative supervision.",
    }
    # Snapshot the template before the manifest is published.  Future formal
    # upgrades must not change recovery semantics for this job.
    template_snapshot = job_root / "formal_training_v4.snapshot.json"
    atomic_text(template_snapshot, json.dumps(FORMAL_TEMPLATE, ensure_ascii=False, indent=2) + "\n")
    job["formal_training"] = template_identity(template_snapshot)
    job["architecture"] = FORMAL_TEMPLATE["architecture"]
    if FORMAL_TEMPLATE.get("negative_control_policy"):
        job["negative_control_policy"] = copy.deepcopy(FORMAL_TEMPLATE["negative_control_policy"])
    if FORMAL_TEMPLATE.get("conditional_negative_policy"):
        from conditional_negatives_v2 import experiment_steps
        job["conditional_negative_policy"] = copy.deepcopy(FORMAL_TEMPLATE["conditional_negative_policy"])
        job["recovery_policy"] = copy.deepcopy(FORMAL_TEMPLATE["recovery_policy"])
        job["branch_experiment"]["allowed_steps"] = experiment_steps(job)
    if FORMAL_TEMPLATE["job_defaults"].get("area_alert_policy"):
        job["area_alert_policy"] = copy.deepcopy(FORMAL_TEMPLATE["job_defaults"]["area_alert_policy"])
    if FORMAL_TEMPLATE["job_defaults"].get("negative_response_policy"):
        job["negative_response_policy"] = copy.deepcopy(FORMAL_TEMPLATE["job_defaults"]["negative_response_policy"])
    if FORMAL_TEMPLATE["job_defaults"].get("geographic_prior_policy"):
        from geographic_prior_v1 import prepare as prepare_geographic_prior
        job["geographic_prior_policy"] = copy.deepcopy(FORMAL_TEMPLATE["job_defaults"]["geographic_prior_policy"])
        job["geographic_prior"] = prepare_geographic_prior(job)
    runtime_files = [HERE / "conditional_negatives_v2.py", HERE / "area_alert_v1.py", HERE / "geographic_prior_v1.py", HERE / "spatial_prior_v2.py", HERE / "cashew_background_v1.py", HERE / "prior_semantic_check_v1.py", HERE / "prior_negative_policy_v2.py", HERE / "source_coverage_v2.py",
                     HERE / "select_tile_specific_supplementary.py", HERE / "automatic_recovery_v2.py", HERE / "paired_horizon_v2.py", HERE / "train_expert_v4.py", HERE / "initialization_metric_evidence_v001.py",
                     HERE / "bounded_recovery_v4.py", HERE / "healthy_recovery_runtime_v4.py",
                     HERE / "recovery_attempt_v4.py",
                     HERE / "finalize_v4.py", HERE / "predict_softmax_scores_v4.py",
                     HERE / "predict_raw_argmax_v4.py", HERE / "evaluate_planet_v4.py",
                     HERE / "architecture.py",
                     HERE / "formal_v4.py", HERE / "expert_dispatch_v4.py",
                     HERE / "expert_policy_v4.py", HERE / "schedule_v4.py", HERE / "supplementary_control_v4.py",
                     HERE / "runtime_account_v4.py", HERE / "agentguide_policy_v1.py",
                     HERE.parent / ".codex/skills/cashew-fada-training-expert/scripts/audit_decision_evidence.py",
                     HERE.parent / ".codex/skills/cashew-fada-training-expert/scripts/persistent_candidate_memory.py",
                     HERE.parent / "references/agent_decision_v4.schema.json",
                     HERE.parent / ".codex/skills/cashew-fada-training-expert/SKILL.md"]
    for name in ("formal-distilled-decisions.md", "diagnostic-reasoning.md", "schedule-and-stopping.md",
                 "experiment-library.md", "runtime-contract.md", "applied-history-procedures.md",
                 "terminal-runtime-recovery.md", "source-component-rescue.md", "healthy-restore-review.md"):
        runtime_files.append(HERE.parent / ".codex/skills/cashew-fada-training-expert/references" / name)
    runtime_files.append(HERE.parent / ".codex/skills/cashew-fada-training-expert/references/negative-availability-and-exposure.md")
    runtime_files.append(HERE.parent / ".codex/skills/cashew-fada-training-expert/references/area-alert-and-geographic-prior.md")
    runtime_files.append(HERE.parent / ".codex/skills/cashew-fada-training-expert/references/shared-planting-background-and-semantic-check.md")
    runtime_files.append(HERE.parent / ".codex/skills/cashew-fada-training-expert/references/adm1-training-sample-design.md")
    runtime_files.append(HERE.parent / ".codex/skills/cashew-fada-training-expert/scripts/audit_prior_prediction_semantics.py")
    if job.get("negative_response_policy"):
        runtime_files.append(HERE / "negative_response_policy_v3.py")
        runtime_files.append(HERE.parent / ".codex/skills/cashew-fada-training-expert/references/user-reported-negative-transfer.md")
        for name in ("audit_negative_transfer.py", "compare_threshold_decoding.py"):
            runtime_files.append(HERE.parent / ".codex/skills/cashew-fada-training-expert/scripts" / name)
    if job.get("geographic_prior"):
        runtime_files.append(Path(job["geographic_prior"]["report"]))
        if job["geographic_prior"].get("card"):
            runtime_files.append(Path(job["geographic_prior"]["card"]))
    job["runtime_provenance"] = {
        "schema_version": 1,
        "files": {str(path.resolve()): sha256(path) for path in runtime_files},
        "source_pretrained_sha256": source.get("pretrained_sha256"),
        "source_train_sha256": source.get("source_train_sha256"),
        "source_valid_sha256": source.get("source_valid_sha256"),
        "target_full_list_sha256": sha256(full_list),
        "target_train_list_sha256": sha256(train_list),
    }
    job["planet_training_policy"] = "Target Planet maps and scores are withheld from formal Agent decisions; geographic text context and unified area warnings are advisory; no logit prior; supplementary labels supply class2 suppression only."
    validate_formal_job(job)
    if resume is not None:
        job["resume"] = resume
    atomic_json(job_root / "job.json", job)
    atomic_json(job_root / "status.json", {
        "job_id": name, "run_id": job["run_id"], "state": "prepared",
        "gpu": None, "heartbeat_epoch": 0,
    })
    return {
        "job_id": name,
        "job": str((job_root / "job.json").resolve()),
        "output_root": str(job_root.resolve()),
        "thread_key": job["thread_key"],
        "planet_risk_map": str(planet.resolve()),
        "agent_thread_root": str(agent_thread_root.resolve()) if agent_thread_root else None,
    }


def configure_experience_environment(env: dict, job: dict, item: dict) -> None:
    if job.get("bounded_recovery"):
        # A recovery uses its own history even when its source was formal.
        env["FADA_TRAINING_EXPERIENCE_ROOT"] = str(
            (Path(item["output_root"]) / "training_experience").resolve()
        )
    elif "formal_training" in job:
        env["FADA_TRAINING_EXPERIENCE_ROOT"] = "@WORKSPACE@/training_experience/formal_v4"
    elif job.get("research_stopselect_policy") in {
        "trend_judgment_v2", "trend_judgment_v3", "trend_judgment_v4"
    }:
        env["FADA_TRAINING_EXPERIENCE_ROOT"] = str(
            (Path(item["output_root"]) / "training_experience").resolve()
        )
    elif job.get("research_authorization") == "user_requested_019e_sourceyear_comparison_20260924":
        # Source-year research must not inherit decisions from stale experiments
        # for the same geographic tile or write them into the global experience log.
        env["FADA_TRAINING_EXPERIENCE_ROOT"] = str(
            (Path(item["output_root"]) / "training_experience").resolve()
        )
    elif job.get("research_authorization") == "user_requested_benin_sourceyear_pair_2019_vs_2020_20260927":
        # Keep the paired Benin source-year research isolated from formal and
        # historical Agent experience. Each arm receives its own durable log.
        env["FADA_TRAINING_EXPERIENCE_ROOT"] = str(
            (Path(item["output_root"]) / "training_experience").resolve()
        )


def launch_worker(item: dict, gpu: str, launch_root: Path,
                  attempt: int) -> subprocess.Popen:
    job_root = Path(item["output_root"])
    status_path = job_root / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    if status.get("state") in {"training_complete", "prediction_complete", "evaluation_complete"}:
        raise RuntimeError("Completed training must enter delivery, never another worker launch")
    agent_root = Path(item["agent_thread_root"]) if item.get("agent_thread_root") else launch_root / "agent_threads" / item["thread_key"]
    agent_root.mkdir(parents=True, exist_ok=True)
    log = job_root / "training.log"
    handle = log.open("a", buffering=1)
    env = os.environ.copy()
    # Resolve the one project account setting, ignoring a stale CODEX_HOME.
    account = codex_home()
    env[ACCOUNT_ENV_VAR] = str(account)
    env["CODEX_HOME"] = str(account)
    job = json.loads(Path(item["job"]).read_text())
    if "formal_training" in job:
        validate_formal_job(job)
    configure_experience_environment(env, job, item)
    env["FADA_EXPERT_SESSION"] = str((agent_root / "thread.json").resolve())
    env["FADA_EXPERT_DISPATCH_LOCK"] = str((agent_root / "dispatch.lock").resolve())
    # Never inherit the caller's environment: nerfsd lacks timm.layers and
    # causes post-training prediction to fail (which used to trigger a rerun).
    python_executable = require_rsenv()
    command = [python_executable, str(HERE / "train_expert_v4.py"), "--job", item["job"], "--device", f"cuda:{gpu}"]
    # A transient Agent/app-server outage can leave a reviewed checkpoint but
    # no live worker.  Resume that exact run state rather than silently
    # restarting the tile from epoch 0.  Jobs that fail before their first
    # review (for example during initialization) deliberately retry normally.
    # epoch_000 is only the discardable initialization probe.  It has no
    # formal source review metrics and must not trigger --resume-existing;
    # otherwise a retry fails while looking for source_class2_f1.
    has_reviewed_checkpoint = any(
        int(path.stem.split("_", 1)[1]) > 0
        and (job_root / f"review_epoch_{path.stem.split('_', 1)[1]}.json").is_file()
        for path in (job_root / "checkpoints").glob("epoch_*.pt")
    )
    if status.get("state") not in {"training_complete", "prediction_complete"} and has_reviewed_checkpoint:
        command.append("--resume-existing")
    process = subprocess.Popen(command, cwd=str(HERE), env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    handle.close()
    atomic_json(job_root / "launch.json", {
        "command": command, "gpu": str(gpu), "pid": process.pid,
        "pgid": process.pid, "process_start_ticks": process_start_ticks(process.pid),
        "attempt": int(attempt),
        "thread_key": item["thread_key"], "started_at": datetime.now(timezone.utc).isoformat(),
    })
    return process


def process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        state = stat[stat.rfind(")") + 2:].split()[0]
        return state != "Z"
    except (OSError, ValueError):
        return False


def process_start_ticks(pid: int) -> int | None:
    """Return Linux /proc start ticks, which disambiguate PID reuse."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def launch_process_matches(launch: dict, job_path: str) -> bool:
    pid = int(launch.get("pid", 0) or 0)
    expected_start = launch.get("process_start_ticks")
    if not pid or not process_alive(pid):
        return False
    if expected_start is not None and process_start_ticks(pid) != int(expected_start):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return "train_expert_v4.py" in command and str(job_path) in command


def terminate_process_group(pgid: int, grace_seconds: float = 5.0) -> None:
    """Reap Agent/experiment descendants left after a worker exits."""
    if int(pgid) <= 1 or int(pgid) == os.getpgrp():
        return
    try:
        os.killpg(int(pgid), signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(int(pgid), 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(int(pgid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _local_supervisor_status_path(launch_root: Path) -> Path:
    key = hashlib.sha256(str(launch_root.resolve()).encode()).hexdigest()[:20]
    return Path("/tmp/cashew_fada_supervisors") / f"{key}.json"


def should_retry_worker(status: dict, attempt: int, max_restarts: int) -> bool:
    """Never spend automatic restarts on a terminal Agent account limit."""
    return (status.get("error_type") != "AccountUsageLimitExceeded"
            and attempt <= max_restarts)


def resolve_admission_limits(gpus: list[str], requested_slots_per_gpu: int) -> tuple[list[str], int, int]:
    """Normalize GPU IDs and return (GPUs, per-GPU slots, machine-wide slots)."""
    unique_gpus = list(dict.fromkeys(str(gpu) for gpu in gpus))
    if not unique_gpus or int(requested_slots_per_gpu) < 1:
        raise ValueError({"gpus": unique_gpus, "slots_per_gpu": requested_slots_per_gpu})
    slots_per_gpu = min(int(requested_slots_per_gpu), NAS_SAFE_MAX_WORKERS_PER_GPU)
    machine_slots = min(MAX_TOTAL_CONCURRENT_TRAINING_TASKS, len(unique_gpus) * slots_per_gpu)
    return unique_gpus, slots_per_gpu, machine_slots


def can_admit_training_task(active_total: int, active_on_gpu: int,
                            slots_per_gpu: int, machine_slots: int) -> bool:
    """Single admission predicate shared by the supervisor and regressions."""
    return active_total < machine_slots and active_on_gpu < slots_per_gpu


def round_robin_admission_gpus(gpus: list[str], active_counts: dict[str, int],
                               pending_count: int, slots_per_gpu: int,
                               active_total: int, machine_slots: int) -> list[str]:
    """Plan one fair admission round, filling one slot per GPU before stacking."""
    if pending_count <= 0 or machine_slots <= active_total:
        return []
    counts = {gpu: int(active_counts.get(gpu, 0)) for gpu in gpus}
    remaining = min(int(pending_count), machine_slots - active_total)
    result: list[str] = []
    while len(result) < remaining:
        progressed = False
        for gpu in gpus:
            if counts[gpu] >= slots_per_gpu:
                continue
            result.append(gpu)
            counts[gpu] += 1
            progressed = True
            if len(result) >= remaining:
                break
        if not progressed:
            break
    return result


def finalize_completed_job(item, gpu):
    """Keep the original result; any registered rescue has its own lineage."""
    from recovery_attempt_v4 import configured_recovery
    recovery = configured_recovery(item, str(gpu))
    command = [require_rsenv(), str(HERE / "finalize_v4.py"),
               "--job", item["job"], "--device", f"cuda:{gpu}"]
    result = subprocess.run(command, cwd=str(HERE), capture_output=True, text=True)
    if recovery:
        for path in (Path(item["output_root"]) / "status.json", Path(item["output_root"]) / "final/delivery.json"):
            if path.is_file():
                data = json.loads(path.read_text())
                data["bounded_recovery_result"] = {"state": recovery.get("status"),
                    "delivery_job": recovery.get("delivery_job"), "receipt": str(Path(item["output_root"]) / "recovery_result.json"),
                    "original_trajectory_preserved": True}
                atomic_json(path, data)
    return result


def supervise(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    gpus = [str(value) for value in manifest["gpus"]]
    requested_slots_per_gpu = int(manifest.get("slots_per_gpu", 1))
    gpus, slots_per_gpu, machine_slots = resolve_admission_limits(gpus, requested_slots_per_gpu)
    max_restarts = int(manifest.get("max_restarts_per_job", MAX_RESTARTS_PER_JOB))
    pending = []
    active: dict[int, tuple[subprocess.Popen | None, dict, str, int, int, Path, Path, Path]] = {}
    attempts: dict[str, int] = {}
    failures: list[dict] = []
    launch_root = Path(manifest["launch_root"])
    # Restarting a supervisor must adopt live workers and retain completed
    # jobs.  This prevents a scheduler change from restarting a tile or
    # creating a second Agent process for the same job.
    for item in manifest["jobs"]:
        job_root = Path(item["output_root"])
        status_path = job_root / "status.json"
        launch_path = job_root / "launch.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        if status.get("state") == "evaluation_complete":
            continue
        # Claim before any process is started.  A duplicate supervisor fails
        # closed instead of creating a second Agent/training worker.
        job_lease = acquire_job_lock(item)
        launch = json.loads(launch_path.read_text()) if launch_path.is_file() else {}
        if (status.get("state") in {"training_complete", "prediction_complete"}
                and not launch_process_matches(launch, item["job"])):
            gpu = str(manifest["gpus"][0])
            slot_lease = acquire_gpu_slot(gpu, item["job_id"], NAS_SAFE_MAX_WORKERS_PER_GPU)
            machine_lease = acquire_machine_slot(item["job_id"], MAX_TOTAL_CONCURRENT_TRAINING_TASKS)
            if slot_lease is None or machine_lease is None:
                release_lease(job_lease)
                release_lease(slot_lease)
                release_lease(machine_lease)
                pending.append((item, acquire_job_lock(item)))
                continue
            finalizer = [require_rsenv(), str(HERE / "finalize_v4.py"), "--job", item["job"], "--device", f"cuda:{gpu}"]
            finalized = finalize_completed_job(item, gpu)
            release_lease(slot_lease)
            release_lease(machine_lease)
            status = json.loads(status_path.read_text()) if status_path.is_file() else {}
            if finalized.returncode == 0 and status.get("state") == "evaluation_complete":
                release_lease(job_lease)
                continue
            if status.get("state") in {"training_complete", "prediction_complete"}:
                error = {"command": finalizer, "stdout": finalized.stdout[-4000:],
                         "stderr": finalized.stderr[-4000:]}
                atomic_json(Path(item["output_root"]) / "finalization_error.json", error)
                # A completed training run must never be restarted just because
                # inference/evaluation failed. Preserve its selected checkpoint.
                release_lease(job_lease)
                failures.append({"job_id": item["job_id"], "stage": "finalization", **error})
                continue
        launch = json.loads(launch_path.read_text()) if launch_path.is_file() else {}
        pid = int(launch.get("pid", 0) or 0)
        attempt = int(launch.get("attempt", 1 if pid else 0))
        attempts[item["job_id"]] = attempt
        # Process identity is authoritative during restart.  Agent dispatch may
        # transiently publish agent_decision_ready or another nonterminal state;
        # launching a second worker then would duplicate both training and Agent
        # work for the same tile.
        if launch_process_matches(launch, item["job"]):
            gpu = str(launch.get("gpu"))
            slot_lease = acquire_gpu_slot(gpu, item["job_id"], NAS_SAFE_MAX_WORKERS_PER_GPU)
            machine_lease = acquire_machine_slot(item["job_id"], MAX_TOTAL_CONCURRENT_TRAINING_TASKS)
            if slot_lease is None or machine_lease is None:
                release_lease(job_lease)
                release_lease(slot_lease)
                release_lease(machine_lease)
                raise RuntimeError({"no_machine_wide_gpu_slot_for_adopted_worker": item["job_id"], "gpu": gpu})
            active[pid] = (None, item, gpu, attempt, int(launch.get("pgid", pid)), job_lease, slot_lease, machine_lease)
        else:
            pending.append((item, job_lease))
    local_status = _local_supervisor_status_path(launch_root)
    local_status.parent.mkdir(parents=True, exist_ok=True)
    last_durable_signature = None
    last_durable_write = 0.0
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        while (pending or active) and not stopping:
            active_counts = {gpu: sum(value[2] == gpu for value in active.values()) for gpu in gpus}
            admission_order = round_robin_admission_gpus(
                gpus, active_counts, len(pending), slots_per_gpu, len(active), machine_slots
            )
            for gpu in admission_order:
                if not pending:
                    break
                item, job_lease = pending[0]
                slot_lease = acquire_gpu_slot(gpu, item["job_id"], NAS_SAFE_MAX_WORKERS_PER_GPU)
                if slot_lease is None:
                    continue
                machine_lease = acquire_machine_slot(item["job_id"], MAX_TOTAL_CONCURRENT_TRAINING_TASKS)
                if machine_lease is None:
                    release_lease(slot_lease)
                    continue
                pending.pop(0)
                status_path = Path(item["output_root"]) / "status.json"
                status = json.loads(status_path.read_text()) if status_path.is_file() else {}
                if status.get("state") in {"training_complete", "prediction_complete", "evaluation_complete"}:
                    try:
                        if status["state"] != "evaluation_complete":
                            finalizer = [require_rsenv(), str(HERE / "finalize_v4.py"),
                                         "--job", item["job"], "--device", f"cuda:{gpu}"]
                            finalized = finalize_completed_job(item, gpu)
                            status = json.loads(status_path.read_text()) if status_path.is_file() else {}
                            if finalized.returncode != 0 or status.get("state") != "evaluation_complete":
                                error = {"command": finalizer, "returncode": finalized.returncode,
                                         "stdout": finalized.stdout[-4000:], "stderr": finalized.stderr[-4000:]}
                                atomic_json(Path(item["output_root"]) / "finalization_error.json", error)
                                failures.append({"job_id": item["job_id"], "stage": "finalization", **error})
                    finally:
                        release_lease(slot_lease)
                        release_lease(machine_lease)
                        release_lease(job_lease)
                    continue
                attempt = attempts.get(item["job_id"], 0) + 1
                attempts[item["job_id"]] = attempt
                process = launch_worker(item, gpu, launch_root, attempt)
                active[process.pid] = (process, item, gpu, attempt, process.pid, job_lease, slot_lease, machine_lease)

            for pid, (process, item, gpu, attempt, pgid, job_lease, slot_lease, machine_lease) in list(active.items()):
                code = process.poll() if process is not None else None
                alive = code is None if process is not None else process_alive(pid)
                if alive:
                    continue
                terminate_process_group(pgid)
                status_path = Path(item["output_root"]) / "status.json"
                status = json.loads(status_path.read_text()) if status_path.is_file() else {}
                finalization_failed = False
                # Keep the GPU slot and job lock through deterministic delivery.
                # Prediction/Planet evaluation are post-hoc only and cannot
                # affect Agent decisions or checkpoint selection.
                if status.get("state") in {"training_complete", "prediction_complete"}:
                    finalizer = [require_rsenv(), str(HERE / "finalize_v4.py"), "--job", item["job"], "--device", f"cuda:{gpu}"]
                    finalized = finalize_completed_job(item, gpu)
                    if finalized.returncode != 0:
                        finalization_failed = True
                        atomic_json(Path(item["output_root"]) / "finalization_error.json", {"command": finalizer, "stdout": finalized.stdout[-4000:], "stderr": finalized.stderr[-4000:]})
                    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
                completed = status.get("state") == "evaluation_complete"
                result = {
                    "pid": pid, "pgid": pgid, "returncode": code, "gpu": gpu,
                    "attempt": attempt, "final_state": status.get("state"),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_json(Path(item["output_root"]) / "launch_result.json", result)
                del active[pid]
                release_lease(slot_lease)
                release_lease(machine_lease)
                if finalization_failed and status.get("state") in {"training_complete", "prediction_complete"}:
                    # Do not consume a restart attempt or overwrite a valid
                    # selected checkpoint to recover from an inference error.
                    release_lease(job_lease)
                    failures.append({"job_id": item["job_id"], "stage": "finalization", **result})
                elif not completed:
                    if status.get("error_type") == "AccountUsageLimitExceeded":
                        # More worker launches cannot restore an exhausted
                        # account. Preserve the run for an explicit recovery.
                        release_lease(job_lease)
                        failures.append({"job_id": item["job_id"], "stage": "agent_account_limit", **result})
                    elif should_retry_worker(status, attempt, max_restarts):
                        pending.append((item, job_lease))
                    else:
                        release_lease(job_lease)
                        failures.append({"job_id": item["job_id"], **result})
                else:
                    release_lease(job_lease)

            payload = {
                "state": "stopping" if stopping else ("running" if pending or active else ("failed" if failures else "complete")),
                "gpus": gpus,
                "requested_slots_per_gpu": requested_slots_per_gpu,
                "slots_per_gpu": slots_per_gpu,
                "machine_wide_concurrency_limit": machine_slots,
                "absolute_machine_wide_concurrency_limit": MAX_TOTAL_CONCURRENT_TRAINING_TASKS,
                "nas_safe_concurrency_enforced": requested_slots_per_gpu != slots_per_gpu,
                "pending": [item[0]["job_id"] for item in pending],
                "active": [
                    {"job_id": value[1]["job_id"], "pid": pid, "pgid": value[4],
                     "gpu": value[2], "attempt": value[3]}
                    for pid, value in active.items()
                ],
                "failures": failures,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(local_status, payload)
            signature = json.dumps({key: value for key, value in payload.items() if key != "updated_at"}, sort_keys=True)
            now = time.monotonic()
            if signature != last_durable_signature or now - last_durable_write >= SUPERVISOR_DURABLE_HEARTBEAT_SECONDS:
                atomic_json(launch_root / "supervisor_status.json", payload)
                last_durable_signature = signature
                last_durable_write = now
            if pending or active:
                time.sleep(SUPERVISOR_POLL_SECONDS)
    finally:
        if stopping or sys.exc_info()[0] is not None:
            for _pid, (_process, _item, _gpu, _attempt, pgid, job_lease, slot_lease, machine_lease) in list(active.items()):
                terminate_process_group(pgid)
                release_lease(slot_lease); release_lease(machine_lease); release_lease(job_lease)
            for _item, job_lease in pending:
                release_lease(job_lease)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    final_payload = {
        "state": "stopped" if stopping else ("failed" if failures else "complete"),
        "gpus": gpus, "requested_slots_per_gpu": requested_slots_per_gpu,
        "slots_per_gpu": slots_per_gpu,
        "machine_wide_concurrency_limit": machine_slots,
        "absolute_machine_wide_concurrency_limit": MAX_TOTAL_CONCURRENT_TRAINING_TASKS,
        "pending": [item[0]["job_id"] for item in pending],
        "active": [], "failures": failures,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(local_status, final_payload)
    atomic_json(launch_root / "supervisor_status.json", final_payload)
    if failures:
        raise RuntimeError({"training_jobs_failed": failures})
    if stopping:
        raise RuntimeError("supervisor stopped; all managed worker groups were terminated")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch v4 FADA by target year and country")
    parser.add_argument("--year", type=int, choices=[2019, 2025])
    parser.add_argument("--country", help="for example TOGO or Ivoire")
    parser.add_argument("--tile", help="optional full tile name or hexadecimal tile suffix")
    parser.add_argument("--exclude-tile", action="append", default=[],
                        help="tile full name or hexadecimal suffix to omit; may be repeated")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU ids")
    parser.add_argument("--slots-per-gpu", type=int, default=3,
                        help="requested concurrent tiles per GPU; runtime cap is 3 per GPU and 24 total")
    parser.add_argument("--supplementary-candidates-file", type=Path,
                        help="candidate pool; each tile is always selected by feature similarity")
    parser.add_argument("--supplementary-names-file", type=Path, dest="supplementary_candidates_file",
                        help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume-source-run", type=Path,
                        help="resume one tile from a prior v4 job root into a new output root")
    parser.add_argument("--resume-epoch", type=int,
                        help="checkpoint epoch used with --resume-source-run")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--foreground", action="store_true", help="keep this process as the supervisor")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--supervise", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.supervise:
        supervise(args.supervise)
        return
    if args.year is None or not args.country:
        parser.error("--year and --country are required unless --supervise is used")
    gpus = list(dict.fromkeys(value.strip() for value in args.gpus.split(",") if value.strip()))
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    if args.slots_per_gpu < 1:
        raise ValueError("--slots-per-gpu must be positive")
    if bool(args.resume_source_run) != bool(args.resume_epoch is not None):
        parser.error("--resume-source-run and --resume-epoch must be supplied together")
    if args.resume_source_run and not args.tile:
        parser.error("resume accepts exactly one tile; supply --tile")
    country = normalized_country(args.country)
    source_profile_name, source = load_source_profile(args.year)
    supplementary = validate_supplementary(args.supplementary_candidates_file)
    tiles = resolve_tiles(args.year, country, args.tile)
    # Country launches sometimes intentionally retain a completed reference
    # tile.  Resolve exclusions against the discovered directories so both a
    # full tile name and its hexadecimal suffix are accepted, just like --tile.
    for requested in args.exclude_tile:
        matches = [path for path in tiles if path.name == requested or path.name.endswith("_" + str(requested))]
        if len(matches) != 1:
            raise FileNotFoundError({"excluded_tile": requested, "matches": [path.name for path in matches]})
        tiles.remove(matches[0])
    if not tiles:
        raise RuntimeError("all requested tiles were excluded")
    missing_planet = [tile.name for tile in tiles if not (PLANET_ROOT / f"{tile_suffix(tile)}.tif").is_file()]
    if missing_planet:
        raise FileNotFoundError({"country": country, "year": args.year, "tiles_missing_planet_risk_map": missing_planet})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (args.output_root or (DEFAULT_OUTPUT_ROOT / f"{args.year}_{country}_{timestamp}")).resolve()
    if args.dry_run:
        print(json.dumps({
            "year": args.year, "country": country, "tiles": [tile.name for tile in tiles],
            "tile_count": len(tiles), "pretrained": source["pretrained"], "pretrained_epoch": 173,
            "source_profile": source_profile_name, "source_year": source["source_year"],
            "source_train": source["source_train"], "source_valid": source["source_valid"],
            "supplementary_candidate_count": supplementary["candidate_count"], "max_epoch": MAX_EPOCH,
            "minimum_stop_epoch": MINIMUM_STOP_EPOCH, "gpus": gpus,
            "slots_per_gpu": args.slots_per_gpu,
        }, ensure_ascii=False, indent=2))
        return
    if output_root.exists():
        if any(output_root.iterdir()):
            raise FileExistsError(f"refuse to overwrite non-empty output root: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    resume = None
    agent_thread_root = None
    if args.resume_source_run:
        source_run = args.resume_source_run.resolve()
        source_job_path = source_run / "job.json"
        source_checkpoint = source_run / "checkpoints" / f"epoch_{int(args.resume_epoch):03d}.pt"
        if not source_job_path.is_file() or not source_checkpoint.is_file():
            raise FileNotFoundError({"source_run": str(source_run), "job": str(source_job_path), "checkpoint": str(source_checkpoint)})
        source_job = json.loads(source_job_path.read_text())
        # A production restart must not silently inherit weights/optimizers
        # from an architecture ablation or from the historical Prior trainer.
        validate_formal_job(source_job)
        if source_job.get("job_id") != tiles[0].name or int(source_job.get("target_year", -1)) != args.year:
            raise RuntimeError({"resume_source_does_not_match_requested_tile": source_job.get("job_id"), "requested_tile": tiles[0].name})
        resume = {"source_run_root": str(source_run), "checkpoint": str(source_checkpoint),
                  "checkpoint_sha256": sha256(source_checkpoint), "epoch": int(args.resume_epoch)}
        # The detached launcher stores the durable thread beside the job
        # directory (under its launch root), not inside the job root itself.
        agent_thread_root = source_run.parent / "agent_threads" / str(source_job["thread_key"])
        if not agent_thread_root.is_dir():
            raise FileNotFoundError({"missing_persistent_agent_thread": str(agent_thread_root)})
    jobs = [make_job(tile, args.year, country, output_root, source_profile_name, source, supplementary,
                     resume=resume, agent_thread_root=agent_thread_root) for tile in tiles]
    manifest = {
        "protocol_version": "cashew-fada-training-expert-v4-year-country-entry",
        "year": args.year, "country": country, "gpus": gpus,
        "source_profile": source_profile_name, "source_year": source["source_year"],
        "slots_per_gpu": args.slots_per_gpu,
        "launch_root": str(output_root), "pretrained_epoch": 173,
        "max_epoch": MAX_EPOCH, "minimum_stop_epoch": MINIMUM_STOP_EPOCH,
        "jobs": jobs, "resume": resume,
        "codex_home": str(codex_home()),
        "codex_account_env_var": ACCOUNT_ENV_VAR,
    }
    manifest_path = output_root / "launch_manifest.json"
    atomic_json(manifest_path, manifest)
    if args.prepare_only:
        print(json.dumps({"state": "prepared", "manifest": str(manifest_path), "jobs": len(jobs)}, ensure_ascii=False, indent=2))
        return
    if args.foreground:
        supervise(manifest_path)
        print(json.dumps({"state": "complete", "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
        return
    supervisor_log = output_root / "supervisor.log"
    handle = supervisor_log.open("a", buffering=1)
    command = [require_rsenv(), str(Path(__file__).resolve()), "--supervise", str(manifest_path)]
    process = subprocess.Popen(command, cwd=str(HERE), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    handle.close()
    atomic_json(output_root / "supervisor_launch.json", {
        "pid": process.pid, "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({
        "state": "started_in_background", "manifest": str(manifest_path),
        "supervisor_pid": process.pid, "jobs": len(jobs), "gpus": gpus,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
