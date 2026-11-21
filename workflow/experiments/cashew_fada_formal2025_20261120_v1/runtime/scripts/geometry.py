#!/usr/bin/env python3
"""Strict filename-grid and affine geometry validation for target tiles."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import rasterio
from affine import Affine


COORDINATE = re.compile(r"_(?P<x>\d+)_(?P<y>\d+)\.tif$", re.IGNORECASE)


def coordinates(name: str) -> tuple[int, int]:
    match = COORDINATE.search(Path(name).name)
    if not match:
        raise ValueError(f"missing final _(x)_(y) suffix: {name}")
    return int(match.group("x")), int(match.group("y"))


def inspect_name_grid(names: list[str], expected_patch_count: int,
                     allow_partial: bool = False) -> dict[str, Any]:
    """Cheap preparation-time check; full raster headers are checked by preflight.

    Partial tiles are trainable: their suffixes must still lie on the registered
    57x55 grid, but missing cells are allowed.  A zero-patch tile is never valid.
    """
    if len(names) == 0:
        raise RuntimeError("target tile has no TIFF patches")
    if len(names) != expected_patch_count and not allow_partial:
        raise RuntimeError(f"formal tile requires {expected_patch_count} patches, got {len(names)}")
    pairs = [coordinates(name) for name in names]
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("duplicate coordinate suffix")
    xs = sorted({x for x, _ in pairs})
    ys = sorted({y for _, y in pairs})
    if min(xs) < 0 or max(xs) > 56 or min(ys) < 0 or max(ys) > 54:
        raise RuntimeError(f"coordinate suffix outside registered x=0..56/y=0..54 grid: {xs[0]}..{xs[-1]}/{ys[0]}..{ys[-1]}")
    expected_pairs = {(x, y) for x in xs for y in ys}
    actual_pairs = set(pairs)
    if not allow_partial and actual_pairs != expected_pairs:
        raise RuntimeError(
            f"filename grid is incomplete: missing={len(expected_pairs - actual_pairs)}, "
            f"extra={len(actual_pairs - expected_pairs)}"
        )
    if expected_patch_count == 3135 and not allow_partial and (xs != list(range(57)) or ys != list(range(55))):
        raise RuntimeError(f"registered grid must use x=0..56/y=0..54, got {xs[:1]}..{xs[-1:]}/{ys[:1]}..{ys[-1:]}")
    return {"columns": len(xs), "rows": len(ys), "x_range": [xs[0], xs[-1]], "y_range": [ys[0], ys[-1]]}


def inspect_geometry(target_root: Path, names: list[str], expected_patch_count: int,
                     allow_partial: bool = False) -> dict[str, Any]:
    inspect_name_grid(names, expected_patch_count, allow_partial=allow_partial)
    records = []
    crs = None
    patch_width = patch_height = None
    resolution = None
    all_bounds = []
    suffix_pairs = set()
    for name in names:
        path = target_root / name
        x, y = coordinates(name)
        if (x, y) in suffix_pairs:
            raise RuntimeError(f"duplicate coordinate suffix {(x, y)}")
        suffix_pairs.add((x, y))
        with rasterio.open(path) as source:
            if crs is None:
                crs = source.crs
                patch_width, patch_height = source.width, source.height
                resolution = (float(source.transform.a), float(-source.transform.e))
            if source.crs != crs or (source.width, source.height) != (patch_width, patch_height):
                raise RuntimeError(f"CRS/shape mismatch: {path}")
            if abs(source.transform.b) > 1e-12 or abs(source.transform.d) > 1e-12:
                raise RuntimeError(f"rotated patch unsupported: {path}")
            if not math.isclose(source.transform.a, resolution[0], abs_tol=1e-12) or not math.isclose(-source.transform.e, resolution[1], abs_tol=1e-12):
                raise RuntimeError(f"resolution mismatch: {path}")
            records.append({"name": name, "x": x, "y": y, "transform": source.transform})
            all_bounds.append(source.bounds)
    if crs is None or resolution is None:
        raise RuntimeError("target tile has no readable rasters")
    left = min(item.left for item in all_bounds)
    bottom = min(item.bottom for item in all_bounds)
    right = max(item.right for item in all_bounds)
    top = max(item.top for item in all_bounds)
    res_x, res_y = resolution
    width_f, height_f = (right - left) / res_x, (top - bottom) / res_y
    width, height = round(width_f), round(height_f)
    if not math.isclose(width_f, width, abs_tol=1e-6) or not math.isclose(height_f, height, abs_tol=1e-6):
        raise RuntimeError("mosaic union does not have integral dimensions")
    transform = Affine(res_x, 0.0, left, 0.0, -res_y, top)
    occupied = set()
    x_to_col, y_to_row = {}, {}
    for item in records:
        affine = item["transform"]
        col_f, row_f = (affine.c - left) / res_x, (top - affine.f) / res_y
        col, row = round(col_f), round(row_f)
        if not math.isclose(col_f, col, abs_tol=1e-6) or not math.isclose(row_f, row, abs_tol=1e-6):
            raise RuntimeError(f"non-integral placement: {item['name']}")
        if col % patch_width or row % patch_height:
            raise RuntimeError(f"off-grid patch: {item['name']}")
        cell = (row // patch_height, col // patch_width)
        if cell in occupied:
            raise RuntimeError(f"overlap at grid cell {cell}")
        occupied.add(cell)
        item["col"], item["row"] = col, row
        if item["x"] in x_to_col and x_to_col[item["x"]] != col:
            raise RuntimeError("one x suffix maps to multiple columns")
        if item["y"] in y_to_row and y_to_row[item["y"]] != row:
            raise RuntimeError("one y suffix maps to multiple rows")
        x_to_col[item["x"]] = col
        y_to_row[item["y"]] = row
    if [x_to_col[key] for key in sorted(x_to_col)] != sorted(x_to_col.values()):
        raise RuntimeError("x suffix must increase eastward")
    if [y_to_row[key] for key in sorted(y_to_row)] != sorted(y_to_row.values()):
        raise RuntimeError("y suffix must increase southward")
    rows, columns = height // patch_height, width // patch_width
    expected_cells = {(row, col) for row in range(rows) for col in range(columns)}
    if (not allow_partial and occupied != expected_cells) or (allow_partial and not occupied.issubset(expected_cells)):
        raise RuntimeError(f"grid is incomplete: missing={len(expected_cells - occupied)}, extra={len(occupied - expected_cells)}")
    if expected_patch_count == 3135 and not allow_partial and (
        patch_width != 128 or patch_height != 128 or columns != 57 or rows != 55
        or width != 7296 or height != 7040
    ):
        raise RuntimeError(
            "registered 3,135-patch tile must be 57x55 patches of 128x128 "
            f"and 7296x7040 pixels, got patches={patch_width}x{patch_height}, "
            f"grid={columns}x{rows}, mosaic={width}x{height}"
        )
    return {
        "records": records, "crs": crs, "transform": transform,
        "bounds": (left, bottom, right, top), "width": width, "height": height,
        "patch_width": patch_width, "patch_height": patch_height,
        "columns": columns, "rows": rows,
        "x_range": [min(x_to_col), max(x_to_col)], "y_range": [min(y_to_row), max(y_to_row)],
    }
