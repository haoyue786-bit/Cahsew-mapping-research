#!/usr/bin/env python3
"""Estimate and validate one tile's four-class PlanetMap prior lock."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from fada import calculate_prior_shift
from geometry import inspect_geometry
from protocol import NUM_CLASSES, PROTOCOL_VERSION, atomic_json, canonical_prior_config, read_lines, sha256


SKILL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANET_REGISTRY = SKILL_ROOT / "references/planetmap_registry.json"
PRIOR_LOCK_RELATIVE = Path("prior/planet_prior_lock.json")


def source_label_path(label_root: Path, name: str) -> Path:
    stem = Path(name).stem
    if stem.startswith("AlphaEarth_"):
        parts = stem.split("_", 2)
        if len(parts) == 3 and parts[1].isdigit():
            stem = parts[2]
    return label_root / f"label_processed_{stem}.tif"


def _mapping_as_int(mapping: dict[str, Any]) -> dict[int, str]:
    result = {int(key): str(value) for key, value in mapping.items()}
    if set(result) != set(range(NUM_CLASSES)):
        raise ValueError(f"class mapping must have keys 0..{NUM_CLASSES - 1}: {result}")
    return result


def resolve_planet_profile(job: dict, registry_path: Path = DEFAULT_PLANET_REGISTRY) -> tuple[dict, dict]:
    registry = json.loads(registry_path.read_text())
    country = str(job["country"])
    key = registry.get("aliases", {}).get(country, country)
    country_profile = registry.get("countries", {}).get(key)
    if country_profile is None:
        raise KeyError(f"PlanetMap has no registered country profile for {country!r}")
    profile = dict(registry.get("profile_defaults", {}))
    profile.update(country_profile)
    source_mapping = _mapping_as_int(job["source"]["class_mapping"])
    planet_mapping = _mapping_as_int(profile["class_mapping"])
    if source_mapping != planet_mapping:
        raise ValueError({
            "planet_source_class_mapping_mismatch": {
                "source": source_mapping, "planet": planet_mapping,
            }
        })
    path = Path(profile["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return registry, dict(profile, country_key=key)


def _source_counts(job: dict) -> tuple[np.ndarray, int, list[int]]:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    ignored = 0
    unexpected: set[int] = set()
    label_root = Path(job["source"]["source_label_root"])
    for name in read_lines(Path(job["source"]["source_train"])):
        path = source_label_path(label_root, name)
        with rasterio.open(path) as source:
            label = source.read(1)
        values, value_counts = np.unique(label, return_counts=True)
        for value, count in zip(values.tolist(), value_counts.tolist()):
            if 0 <= int(value) < NUM_CLASSES:
                counts[int(value)] += int(count)
            elif int(value) == int(job["source"]["ignore_index"]):
                ignored += int(count)
            else:
                unexpected.add(int(value))
    if unexpected:
        raise ValueError(f"unexpected source label values: {sorted(unexpected)}")
    if int(counts.sum()) <= 0:
        raise ValueError("source labels contain no valid class pixels")
    return counts, ignored, sorted(unexpected)


def _planet_counts(job: dict, profile: dict) -> tuple[np.ndarray, dict[str, Any]]:
    target_names = read_lines(Path(job["target_full_list"]))
    geometry = inspect_geometry(
        Path(job["target_root"]), target_names, int(job["expected_target_patch_count"]),
        allow_partial=(len(target_names) != int(job["expected_target_patch_count"])),
    )
    path = Path(profile["path"])
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    nodata_count = 0
    unexpected: set[int] = set()
    with rasterio.open(path) as planet:
        original_crs = str(planet.crs)
        source_crs = planet.crs
        source_crs_resolution = "dataset"
        if source_crs is None or not source_crs.is_valid:
            expected_name = str(profile.get("source_crs_name", ""))
            if not expected_name or expected_name not in original_crs:
                raise ValueError(f"PlanetMap CRS is invalid and is not the registered CRS: {original_crs!r}")
            try:
                source_crs = CRS.from_string(str(profile["source_crs_proj4"]))
            except Exception as exc:
                raise ValueError("registered PlanetMap fallback CRS cannot be parsed") from exc
            source_crs_resolution = "registry_proj4_for_matching_local_cs"
        metadata = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "dtype": planet.dtypes[0],
            "count": planet.count,
            "crs": original_crs,
            "source_crs_used": str(source_crs),
            "source_crs_resolution": source_crs_resolution,
            "transform": list(planet.transform),
            "bounds": list(planet.bounds),
            "nodata": planet.nodata,
            "tags": planet.tags(),
        }
        if planet.count != 1 or planet.dtypes[0] != "uint8":
            raise ValueError(f"PlanetMap must be one-band uint8: {metadata}")
        if int(planet.nodata) != int(profile["nodata"]):
            raise ValueError(f"PlanetMap nodata mismatch: {planet.nodata}/{profile['nodata']}")
        actual_rule = planet.tags().get("REMAP_RULE")
        if actual_rule != profile["expected_remap_rule"]:
            raise ValueError(f"PlanetMap REMAP_RULE mismatch: {actual_rule!r}")
        with WarpedVRT(
            planet,
            src_crs=source_crs,
            crs=geometry["crs"],
            transform=geometry["transform"],
            width=geometry["width"],
            height=geometry["height"],
            src_nodata=int(profile["nodata"]),
            nodata=int(profile["nodata"]),
            resampling=Resampling.nearest,
        ) as aligned:
            for record in geometry["records"]:
                window = Window(
                    record["col"], record["row"],
                    geometry["patch_width"], geometry["patch_height"],
                )
                array = aligned.read(1, window=window)
                values, value_counts = np.unique(array, return_counts=True)
                for value, count in zip(values.tolist(), value_counts.tolist()):
                    value = int(value)
                    if 0 <= value < NUM_CLASSES:
                        counts[value] += int(count)
                    elif value == int(profile["nodata"]):
                        nodata_count += int(count)
                    else:
                        unexpected.add(value)
    if unexpected:
        raise ValueError(f"unexpected PlanetMap values on target footprint: {sorted(unexpected)}")
    total = len(geometry["records"]) * geometry["patch_width"] * geometry["patch_height"]
    valid = int(counts.sum())
    metadata.update({
        "sha256": sha256(path),
        "target_grid_resampling": "nearest",
        "target_grid_crs": str(geometry["crs"]),
        "target_patch_count": len(geometry["records"]),
        "footprint_pixel_count": int(total),
        "valid_pixel_count": valid,
        "nodata_pixel_count": int(nodata_count),
        "valid_fraction": valid / max(1, total),
        "unexpected_values": sorted(unexpected),
    })
    return counts, metadata


def smoothed_prior(counts: np.ndarray, pseudocount: float) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    if values.shape != (NUM_CLASSES,) or (values < 0).any():
        raise ValueError("counts must be a non-negative four-vector")
    return (values + pseudocount) / (values.sum() + pseudocount * NUM_CLASSES)


def estimate_prior(job_path: Path, registry_path: Path = DEFAULT_PLANET_REGISTRY,
                   force: bool = False) -> Path:
    if registry_path.resolve() != DEFAULT_PLANET_REGISTRY.resolve():
        raise ValueError("formal jobs use only the Skill's pinned PlanetMap registry")
    job_path = job_path.resolve()
    job = json.loads(job_path.read_text())
    if job.get("protocol_version") != PROTOCOL_VERSION or job.get("state") != "queued":
        raise ValueError("prior estimation requires a queued job from this protocol")
    output = Path(job["output_root"])
    lock_path = output / PRIOR_LOCK_RELATIVE
    if lock_path.exists() and not force:
        validate_prior_lock(job_path, lock_path)
        return lock_path
    if lock_path.exists() and force:
        raise FileExistsError(
            "prior locks are immutable; use a new output root instead of --force overwrite"
        )
    config = canonical_prior_config(job.get("prior", {}).get("config"))
    registry, profile = resolve_planet_profile(job, registry_path)
    source_counts, source_ignored, _ = _source_counts(job)
    target_counts, planet_metadata = _planet_counts(job, profile)
    # Coverage is retained as provenance only.  It must not prevent a tile
    # from training: sparse Planet data is a weaker prior, not a reason to
    # discard the target domain altogether.
    source_prior = smoothed_prior(source_counts, float(config["dirichlet_pseudocount"]))
    target_prior = smoothed_prior(target_counts, float(config["dirichlet_pseudocount"]))
    raw_ratio, shift = calculate_prior_shift(
        source_prior, target_prior, np.asarray(config["tau"], dtype=np.float64),
        float(config["epsilon"]), tuple(config["log_ratio_clip"]),
    )
    clipped_ratio = np.clip(raw_ratio, float(config["log_ratio_clip"][0]), float(config["log_ratio_clip"][1]))
    zero_classes = np.flatnonzero(target_counts == 0).astype(int).tolist()
    mapping = _mapping_as_int(job["source"]["class_mapping"])
    lock = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "job_id": job["job_id"],
        "job_manifest": str(job_path),
        "job_manifest_sha256": sha256(job_path),
        "method": "PlanetMap nearest-neighbour alignment to exact target patch grid plus source-label pixel counts",
        "formula": "z'_c=z_c+tau_c*clip(log(pi_t,c+epsilon)-log(pi_s,c+epsilon))",
        "class_mapping": {str(key): value for key, value in mapping.items()},
        "source_counts": source_counts.tolist(),
        "source_ignored_pixel_count": int(source_ignored),
        "source_prior_smoothed": source_prior.tolist(),
        "target_planet_counts": target_counts.tolist(),
        "target_prior_smoothed": target_prior.tolist(),
        "raw_log_prior_ratio": raw_ratio.tolist(),
        "clipped_log_prior_ratio": clipped_ratio.tolist(),
        "effective_logit_shift": shift.tolist(),
        "config": config,
        "zero_observed_target_classes": zero_classes,
        "zero_policy_applied": "pseudocount plus finite ratio clipping; no class removed",
        "class_removed": False,
        "cashew_class_index": 2,
        "cashew_guard_active": bool(shift[2] < 0),
        "cashew_guard_explanation": (
            "negative class-2 shift reduces target FADA cashew encoding without making cashew impossible"
            if shift[2] < 0 else
            "Planet prior does not indicate lower cashew abundance than the source prior"
        ),
        "planet": planet_metadata,
        "planet_profile": profile,
        "planet_registry": str(registry_path.resolve()),
        "planet_registry_sha256": sha256(registry_path),
        "planet_registry_schema_version": registry.get("schema_version"),
        "application": {
            "training_target_fada_class_encoding": True,
            "source_class_encoding": False,
            "checkpoint_selection": False,
            "early_stopping": False,
            "coral_threshold_selection": False,
            "inference_logits": False,
            "reason": "training-only correction avoids applying the same prior ratio twice",
        },
    }
    atomic_json(lock_path, lock)
    return lock_path


def validate_prior_lock(job_path: Path, lock_path: Path | None = None) -> dict[str, Any]:
    job_path = job_path.resolve()
    job = json.loads(job_path.read_text())
    path = lock_path or Path(job["output_root"]) / PRIOR_LOCK_RELATIVE
    lock = json.loads(path.read_text())
    shift = np.asarray(lock.get("effective_logit_shift"), dtype=np.float64)
    clipped_ratio = np.asarray(lock.get("clipped_log_prior_ratio"), dtype=np.float64)
    source_prior = np.asarray(lock.get("source_prior_smoothed"), dtype=np.float64)
    target_prior = np.asarray(lock.get("target_prior_smoothed"), dtype=np.float64)
    expected_config = canonical_prior_config(job.get("prior", {}).get("config"))
    _, profile = resolve_planet_profile(job, DEFAULT_PLANET_REGISTRY)
    planet_path = Path(profile["path"])
    checks = {
        "protocol": lock.get("protocol_version") == PROTOCOL_VERSION,
        "job_id": lock.get("job_id") == job.get("job_id"),
        "job_manifest": lock.get("job_manifest") == str(job_path),
        "job_manifest_sha256": lock.get("job_manifest_sha256") == sha256(job_path),
        "class_mapping": lock.get("class_mapping") == job["source"]["class_mapping"],
        "config": lock.get("config") == expected_config,
        "shape": shift.shape == clipped_ratio.shape == source_prior.shape == target_prior.shape == (NUM_CLASSES,),
        "finite": bool(np.isfinite(shift).all() and np.isfinite(clipped_ratio).all() and np.isfinite(source_prior).all() and np.isfinite(target_prior).all()),
        "normalized": bool(abs(source_prior.sum() - 1) < 1e-9 and abs(target_prior.sum() - 1) < 1e-9),
        "no_class_removed": lock.get("class_removed") is False,
        "training_only": (
            lock.get("application", {}).get("training_target_fada_class_encoding") is True
            and lock.get("application", {}).get("inference_logits") is False
        ),
        "planet_file": planet_path.is_file(),
        "planet_sha256": planet_path.is_file() and sha256(planet_path) == lock.get("planet", {}).get("sha256"),
        "planet_registry_path": lock.get("planet_registry") == str(DEFAULT_PLANET_REGISTRY.resolve()),
        "planet_registry_sha256": sha256(DEFAULT_PLANET_REGISTRY) == lock.get("planet_registry_sha256"),
    }
    if not all(checks.values()):
        raise RuntimeError({"invalid_planet_prior_lock": str(path), "checks": checks})
    return lock
