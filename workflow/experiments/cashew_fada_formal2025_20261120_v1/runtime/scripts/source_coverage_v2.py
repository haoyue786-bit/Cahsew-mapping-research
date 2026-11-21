"""Frozen AlphaEarth source coverage; feature novelty never assigns a class."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np


def sample_names(path, limit):
    names = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if not names:
        raise ValueError("empty source coverage list")
    indexes = np.linspace(0, len(names) - 1, min(limit, len(names)), dtype=int)
    return [names[int(i)] for i in indexes]


def nearest_distance(values, reference, scale):
    return np.asarray([np.sqrt(np.mean(((reference - value) / scale) ** 2, axis=1)).min()
                       for value in values], dtype=np.float64)


@lru_cache(maxsize=4)
def _source_reference(root, train, valid, train_hash, valid_hash, limit):
    from select_tile_specific_supplementary import descriptor
    train_names = sample_names(train, limit)
    valid_names = sample_names(valid, limit)
    overlap = set(train_names) & set(valid_names)
    if overlap:
        raise ValueError("source coverage calibration must be held out from reference")
    reference = np.stack([descriptor(Path(root) / name) for name in train_names])
    calibration = np.stack([descriptor(Path(root) / name) for name in valid_names])
    if not np.isfinite(reference).all() or not np.isfinite(calibration).all():
        raise ValueError("nonfinite source feature descriptors")
    scale = np.maximum(np.std(reference, axis=0), 1e-4)
    distances = nearest_distance(calibration, reference, scale)
    threshold = float(np.quantile(distances, .95))
    return reference, scale, threshold, {
        "reference_list_sha256": train_hash, "calibration_list_sha256": valid_hash,
        "reference_sample_names": train_names, "calibration_sample_names": valid_names,
        "reference_count": len(reference), "calibration_count": len(calibration),
        "calibration_quantile": .95, "threshold": threshold,
    }


def diagnose(source, target_descriptors, target_sample_names, candidate_descriptors):
    required = {"source_root", "source_train", "source_valid"}
    if required - source.keys():
        raise ValueError("source coverage requires source root/train/held-out valid lists")
    train, valid = Path(source["source_train"]), Path(source["source_valid"])
    reference, scale, threshold, evidence = _source_reference(
        str(Path(source["source_root"]).resolve()), str(train.resolve()), str(valid.resolve()),
        hashlib.sha256(train.read_bytes()).hexdigest(), hashlib.sha256(valid.read_bytes()).hexdigest(), 128)
    distances = nearest_distance(target_descriptors, reference, scale)
    novel = distances > threshold
    # Candidate matching is descriptive: candidates remain weakly labelled.
    matching = {}
    if bool(novel.any()):
        unseen = target_descriptors[novel]
        for name, values in candidate_descriptors.items():
            matching[name] = float(nearest_distance(np.asarray([values]), unseen, scale)[0])
    report = {
        "version": "heldout_source_feature_coverage_v2", **evidence,
        "target_sample_count": len(target_descriptors), "uncovered_sample_count": int(novel.sum()),
        "uncovered_sample_fraction": float(novel.mean()),
        "target_samples": [{"name": name, "nearest_source_distance": float(distance), "outside_source_coverage": bool(flag)}
                           for name, distance, flag in zip(target_sample_names, distances, novel)],
        "distance_definition": "nearest source patch mean descriptor; channel standardization from source reference",
        "calibration_role": "held-out source descriptors, no target labels",
        "novelty_assigns_negative_label": False,
        "limitations": ["Patch means can hide rare or mixed land cover", "Novelty can be cashew or background and can reflect domain-wide shifts", "128 deterministic samples per source split/target; not a complete pixel-level coverage proof", "Weak negative labels are not independently verified"],
    }
    return report, matching
