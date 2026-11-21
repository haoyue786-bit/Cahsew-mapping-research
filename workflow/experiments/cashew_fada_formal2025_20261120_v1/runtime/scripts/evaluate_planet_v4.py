#!/usr/bin/env python3
"""Post-training Planet comparison; never feeds results back into training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from protocol import atomic_json, sha256


def evaluate(job_path: Path) -> dict:
    job = json.loads(job_path.read_text())
    root = Path(job["output_root"]) / "final" / "outputs" / "raw_argmax"
    predicted_path = root / f"{job['job_id']}__four_class_raw_argmax.tif"
    planet_path = Path(job["planet_risk_map"])
    if not predicted_path.is_file() or not planet_path.is_file():
        raise FileNotFoundError({"prediction": str(predicted_path), "planet": str(planet_path)})
    with rasterio.open(predicted_path) as prediction, rasterio.open(planet_path) as planet:
        predicted = prediction.read(1)
        if planet.shape == prediction.shape and planet.transform == prediction.transform and planet.crs == prediction.crs:
            reference = planet.read(1).astype(np.int16)
        else:
            reference = np.full(predicted.shape, 10, dtype=np.int16)
            reproject(planet.read(1), reference, src_transform=planet.transform, src_crs=planet.crs,
                      dst_transform=prediction.transform, dst_crs=prediction.crs,
                      resampling=Resampling.nearest, dst_nodata=10)
    valid = (predicted != 255) & np.isin(reference, [0, 1, 2, 3])
    actual, expected = predicted[valid] == 2, reference[valid] == 2
    tp = int(np.sum(actual & expected)); fp = int(np.sum(actual & ~expected)); fn = int(np.sum(~actual & expected))
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    result = {"job_id": job["job_id"], "evaluation": "PlanetMap_class2_posthoc_only",
              "prediction": str(predicted_path.resolve()), "prediction_sha256": sha256(predicted_path),
              "planet": str(planet_path.resolve()), "planet_sha256": sha256(planet_path),
              "support": "intersection_of_raw_prediction_valid_pixels_and_Planet_classes_0_1_2_3",
              "valid_pixels": int(valid.sum()), "tp": tp, "fp": fp, "fn": fn,
              "precision": precision, "recall": recall,
              "f1": 2 * precision * recall / max(1e-12, precision + recall),
              "iou": tp / max(1, tp + fp + fn), "used_for_training_or_selection": False}
    atomic_json(root / "planet_posthoc_evaluation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(evaluate(args.job), ensure_ascii=False, indent=2))
