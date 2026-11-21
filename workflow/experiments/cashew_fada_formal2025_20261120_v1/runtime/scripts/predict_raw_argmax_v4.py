#!/usr/bin/env python3
"""Predict one completed v4 tile with the selected segmentation checkpoint.

This is the raw four-class argmax product.  It deliberately does not load a
FADA discriminator, apply a Planet logit prior, or run CORAL-P90.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from geometry import inspect_geometry
from protocol import atomic_json, read_lines, sha256
from tif_class2_to_kmz import convert_any_extent as convert_kmz
from train_tile import configure_runtime, torch_load
from architecture import make_model


def write_class2_mask(source_path: Path, destination_path: Path) -> list[int]:
    with rasterio.open(source_path) as source:
        profile = source.profile.copy()
        array = source.read(1)
    mask = np.where(array == 255, 255, np.where(array == 2, 2, 0)).astype(np.uint8)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(destination_path.name + f".tmp.{__import__('os').getpid()}")
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(mask, 1)
    temporary.replace(destination_path)
    return np.unique(mask).tolist()


def predict(job_path: Path, device_name: str, inference_batch: int) -> dict:
    if inference_batch < 1:
        raise ValueError("inference_batch must be positive")
    job = json.loads(job_path.read_text())
    output_root = Path(job["output_root"])
    target_root = Path(job["target_root"])
    names = read_lines(Path(job["target_full_list"]))
    expected_count = int(job.get("target_patch_count", len(names)))
    geometry = inspect_geometry(target_root, names, expected_count, allow_partial=False)
    if len(names) != expected_count:
        raise RuntimeError(f"target patch count mismatch: {len(names)} != {expected_count}")

    device = torch.device(device_name)
    gan, common = configure_runtime(job, device)
    model, load_stats = make_model(gan, job, device)
    selected = output_root / "final/selected_checkpoint.pt"
    if not selected.is_file():
        raise FileNotFoundError(selected)
    payload = torch_load(selected, "cpu")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError({"missing": missing, "unexpected": unexpected})
    model.eval()

    result_root = output_root / "final/outputs/raw_argmax"
    result_root.mkdir(parents=True, exist_ok=True)
    four_class = result_root / f"{job['job_id']}__four_class_raw_argmax.tif"
    profile = {
        "driver": "GTiff", "width": geometry["width"], "height": geometry["height"],
        "count": 1, "dtype": "uint8", "crs": geometry["crs"],
        "transform": geometry["transform"], "nodata": 255,
        "compress": "deflate", "predictor": 1, "tiled": True,
        "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER",
    }
    record_by_name = {item["name"]: item for item in geometry["records"]}
    counts = np.zeros(4, dtype=np.int64)
    support_pixels = 0
    temporary = four_class.with_name(four_class.name + f".tmp.{__import__('os').getpid()}")
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(np.full((geometry["height"], geometry["width"]), 255, dtype=np.uint8), 1)
        with torch.no_grad():
            for start in range(0, len(names), inference_batch):
                part = names[start:start + inference_batch]
                images = np.stack([
                    common.read_raster(target_root / name) for name in part
                ]).astype(np.float32)
                logits, _ = gan.forward_fuse(model, torch.from_numpy(images).to(device).float())
                predictions = logits.detach().argmax(dim=1).cpu().numpy().astype(np.uint8)
                for name, image, prediction in zip(part, images, predictions):
                    support = ~np.all(image == 0, axis=0)
                    prediction = prediction.copy()
                    prediction[~support] = 255
                    support_pixels += int(support.sum())
                    counts += np.bincount(prediction[support].reshape(-1), minlength=4)
                    record = record_by_name[name]
                    destination.write(
                        prediction, 1,
                        window=Window(record["col"], record["row"], geometry["patch_width"], geometry["patch_height"]),
                    )
                print(json.dumps({"stage": "raw_argmax_prediction", "done": min(start + inference_batch, len(names)), "total": len(names)}), flush=True)
    temporary.replace(four_class)

    with rasterio.open(four_class) as source:
        values = np.unique(source.read(1)).tolist()
        actual = {
            "width": source.width, "height": source.height, "crs": str(source.crs),
            "transform": list(source.transform), "bounds": list(source.bounds),
            "dtype": source.dtypes[0], "nodata": source.nodata,
        }
    if set(values) - {0, 1, 2, 3, 255}:
        raise RuntimeError(f"unexpected prediction values: {values}")

    only_class2 = result_root / f"{job['job_id']}__only_class2_raw_argmax.tif"
    only_values = write_class2_mask(four_class, only_class2)
    # Every tile receives the same deterministic six-level superoverlay;
    # boundary extents are padded transparently by convert_any_extent.
    kmz = result_root / f"{job['job_id']}__only_class2_raw_argmax.kmz"
    kmz_qa = convert_kmz(only_class2, kmz, name=f"{job['job_id']} raw argmax class2")
    class2_fraction = float(counts[2] / max(1, support_pixels))
    qa = {
        "job_id": job["job_id"],
        "prediction": "raw_four_class_argmax",
        "checkpoint": str(selected.resolve()),
        "checkpoint_sha256": sha256(selected),
        "selected_epoch": int(payload.get("epoch", -1)),
        "planet_prior_applied_to_logits": False,
        "fada_discriminator_loaded_for_inference": False,
        "target_patch_count": len(names),
        "support_pixel_count": support_pixels,
        "predicted_class_counts": counts.tolist(),
        "class_distribution": (counts / max(1, support_pixels)).tolist(),
        "class2_fraction": class2_fraction,
        "class2_percentage": class2_fraction * 100.0,
        "four_class_tif": str(four_class.resolve()),
        "four_class_sha256": sha256(four_class),
        "four_class_values": values,
        "only_class2_tif": str(only_class2.resolve()),
        "only_class2_sha256": sha256(only_class2),
        "only_class2_values": only_values,
        "kmz": str(kmz.resolve()),
        "kmz_sha256": sha256(kmz),
        "kmz_qa": kmz_qa,
        "geometry": actual,
        "load_stats": load_stats,
    }
    qa_path = result_root / "prediction_raw_argmax.json"
    atomic_json(qa_path, qa)
    atomic_json(output_root / "final/outputs/raw_argmax/class2_ratio.json", {
        "job_id": job["job_id"], "class2_fraction": class2_fraction,
        "class2_percentage": class2_fraction * 100.0,
        "support_pixel_count": support_pixels, "predicted_class_counts": counts.tolist(),
        "source": str(qa_path.resolve()),
    })
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return qa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-batch", type=int, default=16)
    args = parser.parse_args()
    # Prediction is intentionally direct from NAS.  Unlike training it makes a
    # single inference pass, and keeping the result next to its source avoids a
    # full tile copy to tmpfs before GPU work can begin.
    predict(args.job, args.device, args.inference_batch)


if __name__ == "__main__":
    main()
