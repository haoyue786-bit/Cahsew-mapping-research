#!/usr/bin/env python3
"""Build an auditable, tile-specific supplementary-negative list.

Candidates are retained by AlphaEarth descriptor similarity to the target
tile. New jobs also require the configured kilometre radius; historical
feature-only calls keep their recorded behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from pyproj import Geod

GEOD = Geod(ellps="WGS84")


def ensure_geographic_crs():
    """Repair an incompatible inherited PROJ path using this Python environment.

    The production shell can inherit another conda environment's PROJ_DATA.
    Only a failed EPSG lookup triggers the fallback; valid configured paths stay.
    """
    try:
        rasterio.crs.CRS.from_epsg(4326)
    except rasterio.errors.CRSError:
        path = Path(sys.prefix) / "share/proj"
        if not (path / "proj.db").is_file():
            raise
        from rasterio._env import set_proj_data_search_path
        set_proj_data_search_path(str(path))
        rasterio.crs.CRS.from_epsg(4326)


def geographic_box(path):
    ensure_geographic_crs()
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("geographic filtering requires a declared raster CRS")
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)


def geographic_distance_km(point, box):
    """WGS84 geodesic from patch centre to the target's lon/lat box.

    Latitude/longitude boundary projection defines the closest box point;
    geographic containment is zero. Never compare raw degree distances to km.
    """
    x, y = point
    left, bottom, right, top = box
    if not (-180 <= left <= right <= 180 and -90 <= bottom <= top <= 90):
        raise ValueError("unsupported geographic box (including antimeridian crossing)")
    tx, ty = min(max(x, left), right), min(max(y, bottom), top)
    return abs(float(GEOD.inv(x, y, tx, ty)[2])) / 1000.


def descriptor(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(out_shape=(src.count, 32, 32), resampling=Resampling.average).astype(np.float32)
    valid = np.isfinite(data).all(axis=0) & ~np.all(np.isclose(data, 0.0), axis=0)
    if not bool(valid.any()):
        return np.zeros(data.shape[0], dtype=np.float32)
    return data[:, valid].mean(axis=1)


def center_and_bounds(path: Path) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
    with rasterio.open(path) as src:
        b = src.bounds
    return ((float((b.left + b.right) / 2), float((b.bottom + b.top) / 2)),
            (float(b.left), float(b.bottom), float(b.right), float(b.top)))


def target_bounds(paths: list[Path]) -> tuple[float, float, float, float]:
    indexed: list[tuple[int, int, Path]] = []
    for path in paths:
        match = re.search(r"_(\d+)_(\d+)$", path.stem)
        if not match:
            raise ValueError(f"unexpected target patch name: {path.name}")
        indexed.append((int(match.group(1)), int(match.group(2)), path))
    min_x, max_x = min(x for x, _, _ in indexed), max(x for x, _, _ in indexed)
    min_y, max_y = min(y for _, y, _ in indexed), max(y for _, y, _ in indexed)
    corners = [path for x, y, path in indexed if (x, y) in {(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)}]
    boxes = [center_and_bounds(path)[1] for path in corners]
    return min(x[0] for x in boxes), min(x[1] for x in boxes), max(x[2] for x in boxes), max(x[3] for x in boxes)


def distance_to_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    x, y = point
    left, bottom, right, top = box
    return float(np.hypot(max(left - x, 0.0, x - right), max(bottom - y, 0.0, y - top)))


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def select_supplementary(*, target_root: Path, target_names: list[str],
                         candidate_root: Path, candidate_names: list[str],
                         output_list: Path, report: Path,
                         target_sample_count: int = 128,
                         geographic_radius_km: float | None = None,
                         source_profile: dict | None = None) -> dict:
    """Create a variable-size feature list, optionally intersected with geography.

    The threshold comes from the target descriptor spread. No fixed top-k or
    radius widening is applied; historical calls omit the geographic radius.
    """
    target_paths = [target_root / name for name in target_names]
    if not target_paths or any(not path.is_file() for path in target_paths):
        raise FileNotFoundError("target patches missing")
    sample_indexes = np.linspace(0, len(target_paths) - 1, min(target_sample_count, len(target_paths)), dtype=int)
    target_descriptors = np.stack([descriptor(target_paths[int(i)]) for i in sample_indexes])
    sample_names = [target_names[int(i)] for i in sample_indexes]
    geographic = geographic_radius_km is not None
    if geographic:
        ensure_geographic_crs()
    if geographic and (isinstance(geographic_radius_km, bool) or not math.isfinite(float(geographic_radius_km)) or geographic_radius_km <= 0):
        raise ValueError("geographic radius must be a positive finite kilometre value")
    center = np.median(target_descriptors, axis=0)
    scale = np.maximum(np.std(target_descriptors, axis=0), 1e-4)
    target_distances = np.sqrt(np.mean(((target_descriptors - center) / scale) ** 2, axis=1))
    threshold = float(np.quantile(target_distances, 0.95))
    bounds = target_bounds(target_paths)
    if geographic:
        with rasterio.open(target_paths[0]) as src:
            if src.crs is None:
                raise ValueError("geographic filtering requires target CRS")
            geographic_bounds = transform_bounds(src.crs, "EPSG:4326", *bounds, densify_pts=21)
    kept, rejected = [], []
    candidate_descriptors = {}
    for name in candidate_names:
        path = candidate_root / name
        if not path.is_file():
            rejected.append({"name": name, "reason": "missing"})
            continue
        point, _ = center_and_bounds(path)
        geo_distance = distance_to_box(point, bounds)
        if geographic:
            cb = geographic_box(path)
            distance_km = geographic_distance_km(((cb[0]+cb[2])/2, (cb[1]+cb[3])/2), geographic_bounds)
            if distance_km > geographic_radius_km:
                rejected.append({"name": name, "geo_distance_km": distance_km, "reason": "outside_geographic_radius"})
                continue
        values = descriptor(path)
        score = float(np.sqrt(np.mean(((values - center) / scale) ** 2)))
        record = {"name": name, "feature_distance": score, "geo_distance_deg": geo_distance}
        if geographic:
            record["geo_distance_km"] = distance_km
        if score <= threshold:
            kept.append(record)
            candidate_descriptors[name] = values
        else:
            record["reason"] = "feature_dissimilar"
            rejected.append(record)
    coverage = None
    if source_profile is not None:
        from source_coverage_v2 import diagnose
        coverage, matching = diagnose(source_profile, target_descriptors, sample_names, candidate_descriptors)
        for item in kept:
            item["distance_to_uncovered_target_features"] = matching.get(item["name"])
    kept.sort(key=lambda item: (item["feature_distance"], item["geo_distance_deg"], item["name"]))
    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text("\n".join(item["name"] for item in kept) + ("\n" if kept else ""))
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_root": str(target_root.resolve()), "target_patch_count": len(target_paths),
        "target_descriptor_sample_count": len(sample_indexes), "target_bounds": bounds,
        "feature_distance_threshold": threshold,
        "selection_rule": ("AlphaEarth_geographic_and_feature_variable_count_v2" if geographic else "AlphaEarth_feature_distance_only_variable_count"),
        "geographic_distance_used_for_filtering": geographic, "fixed_top_k_used": False,
        "candidate_count": len(candidate_names), "selected_count": len(kept), "selected": kept,
        "rejected_count": len(rejected), "rejected": rejected,
    }
    if geographic:
        payload["geographic_filter"] = {"enabled": True, "radius_km": float(geographic_radius_km),
            "target_geographic_bounds": list(geographic_bounds), "crs": "EPSG:4326",
            "distance_definition": "WGS84 geodesic from candidate patch centre to clamped target bounding-box boundary; inside=0",
            "candidate_selection": "geographic AND feature similarity; no radius widening on empty"}
    if coverage is not None:
        payload["source_coverage"] = coverage
    report.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--target-list", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-list", type=Path, required=True)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-sample-count", type=int, default=128)
    parser.add_argument("--geographic-radius-km", type=float)
    parser.add_argument("--source-profile-json", type=Path)
    args = parser.parse_args()

    result = select_supplementary(
        target_root=args.target_root, target_names=read_names(args.target_list),
        candidate_root=args.candidate_root, candidate_names=read_names(args.candidate_list),
        output_list=args.output_list, report=args.report, target_sample_count=args.target_sample_count,
        geographic_radius_km=args.geographic_radius_km,
        source_profile=json.loads(args.source_profile_json.read_text()) if args.source_profile_json else None,
    )
    print(json.dumps({key: result[key] for key in ("selected_count", "candidate_count", "feature_distance_threshold", "selection_rule")}))


if __name__ == "__main__":
    main()
