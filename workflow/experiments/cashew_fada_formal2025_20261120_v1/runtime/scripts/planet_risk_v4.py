#!/usr/bin/env python3
"""Read-only PlanetMap risk summaries for the v4 Agent packet."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import rasterio


def summarize(tile_id: str, root: Path = Path("@WORKSPACE@/planetmap/antile")) -> dict:
    path = root / f"{tile_id}.tif"
    if not path.is_file():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        values = src.read(1)
        profile = {"width": src.width, "height": src.height, "crs": str(src.crs), "transform": list(src.transform)}
    # 0 is a valid remapped class even though old exports may carry nodata=0.
    known = np.isin(values, [0, 1, 2, 3])
    counts = np.bincount(values[known].astype(np.uint8), minlength=4).astype(int)
    total = int(counts.sum())
    return {
        "tile_id": tile_id,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "class2_fraction": float(counts[2] / max(1, total)),
        "known_fraction": float(total / values.size),
        "class_counts": counts.tolist(),
        "pixel_count": int(values.size),
        "valid_definition": "values 0,1,2,3; all other values ignored; TIFF nodata tag is not used",
        **profile,
    }
