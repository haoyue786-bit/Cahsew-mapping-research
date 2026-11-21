"""Use GEHistoricalImagery to query and retrieve complete, exact-date Google imagery."""

from datetime import date
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image
import rasterio
from rasterio import Affine
from rasterio.enums import MaskFlags
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from shapely.geometry import box, shape

from _runtime.common import digest, read, write


BACKEND = "GEHistoricalImagery"
SIZE, GRID = 1024, 128


def binary_path():
    value = os.environ.get("CASHEW_GEHI_BIN")
    return Path(value).expanduser().resolve() if value else None


def invoke(folder, filename, bounds, zoom, verb, *options):
    binary = binary_path()
    if not binary or not binary.is_file():
        raise RuntimeError("Set CASHEW_GEHI_BIN to the installed GEHistoricalImagery executable")
    folder.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bounds
    command = [str(binary), verb, "--provider=TM", f"--zoom={zoom}",
               f"--lower-left={south},{west}", f"--upper-right={north},{east}",
               "--parallel=4", "--no-cache", *options]
    proxy = os.environ.get("CASHEW_GEHI_PROXY", os.environ.get("CASHEW_IMAGERY_PROXY", ""))
    env = dict(os.environ, HTTPS_PROXY=proxy, HTTP_PROXY=proxy, ALL_PROXY=proxy,
               PROJ_DATA=str(binary.parent / "gdal"))
    env.pop("PROJ_LIB", None)
    env.pop("GDAL_DATA", None)
    log = folder / filename
    with log.open("w") as stream:
        subprocess.run(command, cwd=folder, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=150)
    return log


def catalog(grid, root, oldest, newest=None, *, all_zooms=False):
    bounds = grid["bounds_wgs84"]
    footprint = box(*bounds)
    ceiling = min(newest, date.today()) if newest else date.today()
    frames = []
    failures = []
    for zoom in (18, 17):
        out = (root / f"availability_z{zoom}.geojson").resolve()
        try:
            log = invoke(root, f"availability_z{zoom}.log", bounds, zoom, "availability", "-q",
                         "--complete", f"--output={out}", "--min-date=" + oldest.strftime("%Y/%m/%d"))
        except (OSError, subprocess.SubprocessError) as error:
            if not all_zooms:
                raise
            failures.append(error)
            write(root / f"availability_z{zoom}_unavailable.json", {"reason": str(error)})
            continue
        if not out.exists():
            if "No dated imagery available" in log.read_text():
                continue
            error = RuntimeError("GEHistoricalImagery did not return date coverage; see " + str(log))
            if not all_zooms:
                raise error
            failures.append(error)
            write(root / f"availability_z{zoom}_unavailable.json", {"reason": str(error)})
            continue
        days = set()
        for row in read(out)["features"]:
            properties = row["properties"]
            if (properties.get("provider") == "TM" and properties.get("zoom_level") == zoom
                    and properties.get("iscomplete") == 1 and shape(row["geometry"]).covers(footprint)):
                day = date.fromisoformat(properties["image_date"])
                if oldest <= day <= ceiling:
                    days.add(day.isoformat())
        frames.extend(dict(provider="Google", google_backend=BACKEND, zoom=zoom,
                         acquisition_dates=[day], acquisition_range_verified=True, source_date=[day],
                         acquisition_date_scope="complete_footprint_at_requested_zoom", date_evidence=str(out),
                         frame_georeferencing_verified=True, native128_valid_fraction=1,
                         folder=str(root / f"{day}_z{zoom}")) for day in sorted(days, reverse=True))
        if frames and not all_zooms:
            return frames
    if not frames and failures:
        raise failures[-1]
    return frames


def download(folder, bounds, zoom, day):
    target = (folder / "source_rgb.tif").resolve()
    log = invoke(folder, "download.log", bounds, zoom, "download", "--exact-date",
                 "--date=" + day.replace("-", "/"), f"--output={target}",
                 "--co=PHOTOMETRIC=RGB", "--co=COMPRESS=DEFLATE")
    output = log.read_text()
    counts = re.search(r"([\d,]+) out of ([\d,]+) downloaded", output)
    if not counts or "substituting" in output.lower() or not target.is_file():
        raise RuntimeError("Exact-date tile download could not be verified at this zoom; see " + str(log))
    delivered, requested = (int(value.replace(",", "")) for value in counts.groups())
    if not 0 < delivered <= requested:
        raise ValueError("Invalid tile download counts in " + str(log))
    if delivered < requested:
        with rasterio.open(target) as raster:
            if MaskFlags.all_valid in raster.mask_flag_enums[0]:
                raise ValueError("Incomplete tile download must expose a raster validity mask")
    write(folder / "TILE_DOWNLOAD.json", dict(image_date=day, zoom=zoom, tiles_delivered=delivered,
                                              tiles_requested=requested, image=str(target),
                                              sha256=digest(target), log=str(log)))
    return target


def align(source_image, crs, transform):
    rgb = np.zeros((3, SIZE, SIZE), dtype=np.uint8)
    valid = np.zeros((SIZE, SIZE), dtype=np.uint8)
    with rasterio.open(source_image) as src:
        if not src.crs or src.count < 3:
            raise ValueError("Google raster did not contain georeferenced RGB")
        geo = dict(src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=crs)
        reproject(src.read((1, 2, 3)), rgb, **geo, resampling=Resampling.bilinear)
        reproject(src.dataset_mask(), valid, **geo, resampling=Resampling.nearest)
        source_resolution = src.res
    return np.moveaxis(rgb, 0, -1), valid > 0, source_resolution


def fetch_frame(grid, candidate):
    folder = Path(candidate["folder"])
    source = download(folder, grid["bounds_wgs84"], candidate["zoom"], candidate["acquisition_dates"][0])
    transform = Affine(*grid["native_transform"]) * Affine.scale(GRID / SIZE)
    image, valid, resolution = align(source, grid["native_crs"], transform)
    step = SIZE // GRID
    coverage = valid.reshape(GRID, step, GRID, step).all(axis=(1, 3))
    png, mask = folder / "registered_rgb.png", folder / "valid_native128.png"
    Image.fromarray(image).save(png)
    Image.fromarray(coverage.astype(np.uint8) * 255).save(mask)
    return dict(candidate, image_path=str(png), sha256=digest(png), image_crs=grid["native_crs"],
                image_transform=list(transform)[:6], bounds_wgs84=grid["bounds_wgs84"],
                native128_valid_mask_path=str(mask), native128_valid_mask_sha256=digest(mask),
                native128_valid_fraction=float(coverage.mean()), source_resolution_degrees=list(resolution),
                source_raster=str(source), tile_download_receipt=str(folder / "TILE_DOWNLOAD.json"))


def landscape(grid, frame, scale, output):
    native = Affine(*grid["native_transform"])
    offset = 64 * (1 - scale)
    transform = native * Affine.translation(offset, offset) * Affine.scale(GRID * scale / SIZE)
    bounds = transform_bounds(grid["native_crs"], "EPSG:4326", *array_bounds(SIZE, SIZE, transform))
    zoom = frame["zoom"] - (1 if scale == 3 else 2)
    source = download(output.parent, bounds, zoom, frame["acquisition_dates"][0])
    rgb, valid, _ = align(source, grid["native_crs"], transform)
    if not valid.all():
        raise ValueError("The same capture date did not cover the wider area completely")
    Image.fromarray(rgb).save(output)
    return bounds
