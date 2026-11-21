#!/usr/bin/env python3
"""Convert a georeferenced class-2 TIFF to a deterministic six-level KMZ superoverlay."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window


TILE_WIDTH = 228
TILE_HEIGHT = 220
LEVELS = 6
CLASS_VALUE = 2
RGBA = (255, 140, 0, 210)
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_info(name: str, *, directory: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.create_system = 3
    if directory:
        info.external_attr = (0o40755 << 16) | 0x10
        info.compress_type = zipfile.ZIP_STORED
    else:
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
    return info


def write_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    archive.writestr(zip_info(name), payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def bounds_for(global_bounds, grid: int, x: int, y_south: int) -> tuple[float, float, float, float]:
    west = global_bounds.left + (global_bounds.right - global_bounds.left) * x / grid
    east = global_bounds.left + (global_bounds.right - global_bounds.left) * (x + 1) / grid
    south = global_bounds.bottom + (global_bounds.top - global_bounds.bottom) * y_south / grid
    north = global_bounds.bottom + (global_bounds.top - global_bounds.bottom) * (y_south + 1) / grid
    return west, south, east, north


def region_xml(west: float, south: float, east: float, north: float,
               min_lod: int, max_lod: int) -> str:
    return (
        "<Region><LatLonAltBox>"
        f"<north>{north:.15g}</north><south>{south:.15g}</south>"
        f"<east>{east:.15g}</east><west>{west:.15g}</west>"
        "</LatLonAltBox><Lod>"
        f"<minLodPixels>{min_lod}</minLodPixels><maxLodPixels>{max_lod}</maxLodPixels>"
        "</Lod></Region>"
    )


def tile_kml(name: str, level: int, x: int, y_south: int, global_bounds) -> bytes:
    grid = 2 ** level
    west, south, east, north = bounds_for(global_bounds, grid, x, y_south)
    min_lod = 0 if level == 0 else 128
    max_lod = -1 if level == LEVELS - 1 else 1024
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
        f"<name>{escape(name)} L{level} X{x} Y{y_south}</name>",
        region_xml(west, south, east, north, min_lod, max_lod),
        "<GroundOverlay>",
        f"<name>{escape(name)}</name><drawOrder>{level}</drawOrder>",
        f"<Icon><href>{y_south}.png</href></Icon>",
        "<LatLonBox>",
        f"<north>{north:.15g}</north><south>{south:.15g}</south>"
        f"<east>{east:.15g}</east><west>{west:.15g}</west>",
        "</LatLonBox></GroundOverlay>",
    ]
    if level < LEVELS - 1:
        child_level = level + 1
        child_grid = 2 ** child_level
        for child_x in (2 * x, 2 * x + 1):
            for child_y in (2 * y_south, 2 * y_south + 1):
                child_w, child_s, child_e, child_n = bounds_for(
                    global_bounds, child_grid, child_x, child_y
                )
                parts.extend([
                    "<NetworkLink>",
                    f"<name>L{child_level} X{child_x} Y{child_y}</name>",
                    region_xml(child_w, child_s, child_e, child_n, 128,
                               -1 if child_level == LEVELS - 1 else 1024),
                    "<Link>",
                    f"<href>../../{child_level}/{child_x}/{child_y}.kml</href>",
                    "<viewRefreshMode>onRegion</viewRefreshMode>",
                    "</Link></NetworkLink>",
                ])
    parts.append("</Document></kml>")
    return "".join(parts).encode("utf-8")


def doc_kml(name: str, global_bounds) -> bytes:
    region = region_xml(
        global_bounds.left, global_bounds.bottom, global_bounds.right, global_bounds.top, 0, -1
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"<name>{escape(name)}</name><open>1</open><NetworkLink><name>{escape(name)}</name>"
        f"{region}<Link><href>0/0/0.kml</href><viewRefreshMode>onRegion</viewRefreshMode>"
        "</Link></NetworkLink></Document></kml>"
    ).encode("utf-8")


def rgba_png(tile: np.ndarray) -> bytes:
    rgba = np.zeros((TILE_HEIGHT, TILE_WIDTH, 4), dtype=np.uint8)
    selected = tile == CLASS_VALUE
    rgba[selected, 0] = RGBA[0]
    rgba[selected, 1] = RGBA[1]
    rgba[selected, 2] = RGBA[2]
    rgba[selected, 3] = RGBA[3]
    stream = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(stream, format="PNG", compress_level=6)
    return stream.getvalue()


def convert(source: Path, destination: Path, name: str | None = None) -> dict[str, object]:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    display_name = name or source.stem
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    with rasterio.open(source) as dataset:
        if dataset.count != 1:
            raise ValueError(f"expected one-band class2 TIFF: {source}")
        if dataset.width != TILE_WIDTH * 32 or dataset.height != TILE_HEIGHT * 32:
            raise ValueError(
                f"expected 7296x7040 for exact 6-level pyramid, got {dataset.width}x{dataset.height}"
            )
        if dataset.crs is None or not dataset.crs.is_geographic:
            raise ValueError(f"KMZ LatLonBox requires geographic CRS, got {dataset.crs}")
        values = np.unique(dataset.read(1)).tolist()
        if set(values) - {0, 2, 255}:
            raise ValueError(f"only-class2 TIFF has unexpected values: {values}")
        global_bounds = dataset.bounds

        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            write_bytes(archive, "doc.kml", doc_kml(display_name, global_bounds))
            for level in range(LEVELS):
                grid = 2 ** level
                archive.writestr(zip_info(f"{level}/", directory=True), b"")
                level_array = dataset.read(
                    1,
                    out_shape=(grid * TILE_HEIGHT, grid * TILE_WIDTH),
                    resampling=Resampling.nearest,
                )
                for x in range(grid):
                    archive.writestr(zip_info(f"{level}/{x}/", directory=True), b"")
                    for y_south in range(grid):
                        row_from_north = grid - 1 - y_south
                        tile = level_array[
                            row_from_north * TILE_HEIGHT:(row_from_north + 1) * TILE_HEIGHT,
                            x * TILE_WIDTH:(x + 1) * TILE_WIDTH,
                        ]
                        write_bytes(archive, f"{level}/{x}/{y_south}.png", rgba_png(tile))
                        write_bytes(
                            archive,
                            f"{level}/{x}/{y_south}.kml",
                            tile_kml(display_name, level, x, y_south, global_bounds),
                        )
    temporary.replace(destination)
    return validate(source, destination)


def convert_any_extent(source: Path, destination: Path, name: str | None = None) -> dict[str, object]:
    """Create a KMZ for every valid class-2 TIFF.

    Standard 7,296 x 7,040 tiles retain the deterministic six-level
    superoverlay. Boundary tiles are padded on the right/bottom with 255
    (transparent) pixels and then use the exact same six-level superoverlay.
    """
    source = source.resolve()
    with rasterio.open(source) as dataset:
        if dataset.count != 1:
            raise ValueError(f"expected one-band class2 TIFF: {source}")
        if dataset.crs is None or not dataset.crs.is_geographic:
            raise ValueError(f"KMZ LatLonBox requires geographic CRS, got {dataset.crs}")
        if (dataset.width, dataset.height) == (TILE_WIDTH * 32, TILE_HEIGHT * 32):
            return convert(source, destination, name)
        original_size = [dataset.width, dataset.height]
        if dataset.width > TILE_WIDTH * 32 or dataset.height > TILE_HEIGHT * 32:
            raise ValueError(f"boundary TIFF exceeds standard tile extent: {source}")
        source_values = dataset.read(1)
        if set(np.unique(source_values).tolist()) - {0, 2, 255}:
            raise ValueError(f"only-class2 TIFF has unexpected values: {source}")
        temporary_source = source.with_name(source.name + f".padded.{os.getpid()}.tif")
        profile = dataset.profile.copy()
        profile.update({
            "width": TILE_WIDTH * 32, "height": TILE_HEIGHT * 32,
            "nodata": 255, "BIGTIFF": "IF_SAFER",
        })
        with rasterio.open(temporary_source, "w", **profile) as padded:
            padded.write(np.full((TILE_HEIGHT * 32, TILE_WIDTH * 32), 255, dtype=np.uint8), 1)
            padded.write(source_values, 1, window=Window(0, 0, dataset.width, dataset.height))
    try:
        qa = convert(temporary_source, destination, name)
        qa["source_tif"] = str(source)
        qa["source_sha256"] = sha256(source)
        qa["padding"] = {"original_size": original_size,
                          "padded_size": [TILE_WIDTH * 32, TILE_HEIGHT * 32],
                          "fill_value": 255}
        return qa
    finally:
        temporary_source.unlink(missing_ok=True)


def validate(source: Path, kmz: Path) -> dict[str, object]:
    with zipfile.ZipFile(kmz) as archive:
        names = archive.namelist()
        png_names = [name for name in names if name.endswith(".png")]
        tile_kml_names = [name for name in names if name != "doc.kml" and name.endswith(".kml")]
        directory_names = [name for name in names if name.endswith("/")]
        expected_level_counts = {str(level): (2 ** level) ** 2 for level in range(LEVELS)}
        actual_level_counts = {
            str(level): sum(name.startswith(f"{level}/") and name.endswith(".png") for name in png_names)
            for level in range(LEVELS)
        }
        sample_names = ["0/0/0.png", "5/0/0.png", "5/31/31.png"]
        samples = {}
        for sample in sample_names:
            with Image.open(io.BytesIO(archive.read(sample))) as image:
                rgba = np.asarray(image.convert("RGBA"))
                alpha_values = np.unique(rgba[:, :, 3]).tolist()
                visible = rgba[:, :, 3] > 0
                visible_colors = np.unique(rgba[visible], axis=0).tolist() if visible.any() else []
                samples[sample] = {
                    "size": list(image.size),
                    "alpha_values": alpha_values,
                    "visible_rgba_values": visible_colors,
                }
        qa = {
            "source_tif": str(source),
            "source_sha256": sha256(source),
            "kmz": str(kmz),
            "kmz_sha256": sha256(kmz),
            "entry_count": len(names),
            "directory_entry_count": len(directory_names),
            "png_count": len(png_names),
            "tile_kml_count": len(tile_kml_names),
            "doc_kml_count": names.count("doc.kml"),
            "levels": LEVELS,
            "expected_png_per_level": expected_level_counts,
            "actual_png_per_level": actual_level_counts,
            "tile_size": [TILE_WIDTH, TILE_HEIGHT],
            "y_numbering": "south_to_north",
            "resampling": "nearest",
            "class_value_visible": CLASS_VALUE,
            "other_values": "transparent",
            "visible_rgba": list(RGBA),
            "samples": samples,
        }
        allowed_sample_rgba = {tuple(RGBA)}
        samples_pass = all(
            item["size"] == [TILE_WIDTH, TILE_HEIGHT]
            and set(item["alpha_values"]) <= {0, RGBA[3]}
            and {tuple(value) for value in item["visible_rgba_values"]} <= allowed_sample_rgba
            for item in samples.values()
        )
        qa["qa_pass"] = bool(
            qa["entry_count"] == 2800
            and qa["directory_entry_count"] == 69
            and qa["png_count"] == 1365
            and qa["tile_kml_count"] == 1365
            and qa["doc_kml_count"] == 1
            and actual_level_counts == expected_level_counts
            and samples_pass
        )
        if not qa["qa_pass"]:
            raise RuntimeError(qa)
        return qa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = validate(args.source, args.destination) if args.validate_only else convert(
        args.source, args.destination, args.name
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
