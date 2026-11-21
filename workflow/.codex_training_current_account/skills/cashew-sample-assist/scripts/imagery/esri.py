"""Fetch Esri current and locally changed Wayback imagery on the AEF footprint."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from hashlib import sha256
from io import BytesIO
from math import ceil, floor
from pathlib import Path

import numpy as np
from PIL import Image
import os
import requests
from rasterio import Affine
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

from _runtime.common import digest, write
from imagery_policy import date_rejection


CURRENT = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer"
CURRENT_TILES = CURRENT.replace("services.arcgisonline.com", "server.arcgisonline.com")
WAYBACK = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer"
CATALOG = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
TILE_SIZE, ZOOM, SIZE = 256, 17, 1024
HALF_WORLD = 20037508.342789244


def get(url, params=None):
    proxy = os.environ.get('CASHEW_ESRI_PROXY') or os.environ.get('CASHEW_IMAGERY_PROXY')
    with requests.Session() as session:
        session.trust_env = False
        proxies = {'http': proxy, 'https': proxy} if proxy else None
        response = session.get(url, params=params, timeout=(7, 20), proxies=proxies)
        response.raise_for_status()
        return response


def json_data(url, params=None):
    response = get(url, params)
    value = response.json()
    if value.get("error"):
        raise RuntimeError(str(value["error"]))
    return value


def tile_layout(crs, transform, size, zoom):
    west, south, east, north = transform_bounds(crs, "EPSG:3857", *array_bounds(size, size, transform))
    width = 2 * HALF_WORLD / 2**zoom
    cols = range(floor((west + HALF_WORLD) / width), ceil((east + HALF_WORLD) / width))
    rows = range(floor((HALF_WORLD - north) / width), ceil((HALF_WORLD - south) / width))
    mosaic_transform = Affine(width / TILE_SIZE, 0, -HALF_WORLD + cols.start * width,
                              0, -width / TILE_SIZE, HALF_WORLD - rows.start * width)
    return rows, cols, mosaic_transform


def acquire_date(attributes):
    timestamp = attributes.get("SRC_DATE2")
    if isinstance(timestamp, (int, float)) and 0 < timestamp < 5e12:
        return datetime.fromtimestamp(timestamp / 1000, timezone.utc).date().isoformat()
    raw = str(attributes.get("SRC_DATE") or "")
    if len(raw) == 8 and raw.isdigit():
        return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8])).isoformat()
    return None


def acquisition_coverage(data, transform, zoom):
    dates = {}
    highest = np.full((128, 128), -np.inf)
    known = np.ones(highest.shape, dtype=bool)
    for feature in data.get("features", []):
        attrs = feature.get("attributes", {})
        rings = (feature.get("geometry") or {}).get("rings")
        if not rings or not (attrs.get("MinMapLevel") or 0) <= zoom <= (attrs.get("MaxMapLevel") or 99):
            continue
        footprint = {"type": "Polygon", "coordinates": rings}
        mask = rasterize([(footprint, 1)], out_shape=highest.shape, transform=transform,
                         all_touched=True).astype(bool)
        priority = attrs.get("DrawOrder") or 0
        better = mask & (priority > highest)
        tied = mask & (priority == highest)
        acquired = acquire_date(attrs)
        known[better] = acquired is not None
        if not acquired:
            known[tied] = False
        highest[better] = priority
        dates.setdefault(acquired, []).append((mask, priority))
    dates_visible = sorted(day for day, regions in dates.items()
                           if day and any(np.any(mask & (highest == priority)) for mask, priority in regions))
    complete = bool((known & np.isfinite(highest)).all() and not data.get("exceededTransferLimit"))
    return dates_visible, complete


def frame_descriptor(grid, folder, release, tile_url, metadata_url, zoom=ZOOM, *, query_data=None):
    transform = Affine(*grid["native_transform"])
    bbox = array_bounds(128, 128, transform)
    wkid = grid["native_crs"].split(":")[-1]
    query = dict(f="json", where="1=1", outFields="*", returnGeometry="true",
                 geometry=",".join(map(str, bbox)), geometryType="esriGeometryEnvelope",
                 spatialRel="esriSpatialRelIntersects", inSR=wkid, outSR=wkid)
    metadata = (query_data or json_data)(metadata_url, query)
    write(folder / "source_footprints.json", metadata)
    dates, complete = acquisition_coverage(metadata, transform, zoom)
    return dict(provider="Esri", release=release, acquisition_dates=dates,
                acquisition_range_verified=complete, date_evidence=str(folder / "source_footprints.json"),
                source_date=dates, frame_georeferencing_verified=True, native128_valid_fraction=1,
                tile_url=tile_url, metadata_url=metadata_url, folder=str(folder))


def current(grid, root):
    return frame_descriptor(grid, root / "current", "current", CURRENT_TILES + "/tile/{level}/{row}/{col}",
                            CURRENT + "/4/query")


def archive(grid, root, oldest, *, query_data=None, parallel_metadata=False):
    query_data = query_data or json_data
    config = query_data(CATALOG)
    ordered = sorted(config, key=lambda key: config[key]["itemTitle"], reverse=True)
    index = {key: idx for idx, key in enumerate(ordered)}
    rows, cols, _ = tile_layout(grid["native_crs"], Affine(*grid["native_transform"]), 128, ZOOM)

    def changes(position):
        row, col = position
        result = []
        offset = 0
        while offset < len(ordered):
            key = ordered[offset]
            if config[key]["itemTitle"][-11:-1] < oldest.isoformat():
                break
            response = query_data(f"{WAYBACK}/tilemap/{key}/{ZOOM}/{row}/{col}")
            if not response["data"][0]:
                break
            selected = str((response.get("select") or [None])[0] or key)
            offset = max(offset, index.get(selected, offset)) + 1
            if selected in config:
                result.append(selected)
        return result

    with ThreadPoolExecutor(max_workers=8) as pool:
        selected = {key for values in pool.map(changes, ((r, c) for r in rows for c in cols)) for key in values}
    def descriptor(key):
        item = config[key]
        folder = root / f"wayback_{key}"
        try:
            return frame_descriptor(grid, folder, key, item["itemURL"],
                                    item["metadataLayerUrl"] + f"/{23-ZOOM}/query", query_data=query_data)
        except (ValueError, RuntimeError, requests.RequestException) as error:
            write(folder / "unavailable.json", {"reason": str(error)})
            return None

    keys = sorted(selected, key=index.__getitem__)
    if parallel_metadata:
        with ThreadPoolExecutor(max_workers=4) as pool:
            yield from (frame for frame in pool.map(descriptor, keys) if frame is not None)
    else:
        yield from (frame for frame in map(descriptor, keys) if frame is not None)


def mosaic(crs, transform, size, zoom, tile_url):
    rows, cols, src_transform = tile_layout(crs, transform, size, zoom)

    def fetch(position):
        row, col = position
        response = get(tile_url.format(level=zoom, row=row, col=col))
        with Image.open(BytesIO(response.content)) as image:
            return row, col, image.convert("RGBA")

    source = Image.new("RGBA", (len(cols) * TILE_SIZE, len(rows) * TILE_SIZE))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row, col, image in pool.map(fetch, ((r, c) for r in rows for c in cols)):
            source.paste(image, ((col - cols.start) * TILE_SIZE, (row - rows.start) * TILE_SIZE))
    result = np.zeros((4, size, size), dtype=np.uint8)
    reproject(np.moveaxis(np.asarray(source), -1, 0), result, src_crs="EPSG:3857",
              src_transform=src_transform, dst_crs=crs, dst_transform=transform,
              resampling=Resampling.bilinear)
    return np.moveaxis(result, 0, -1)


def fetch_frame(grid, descriptor):
    folder = Path(descriptor["folder"])
    transform = Affine(*grid["native_transform"]) * Affine.scale(128 / SIZE)
    image = mosaic(grid["native_crs"], transform, SIZE, ZOOM, descriptor["tile_url"])
    support = (image[:, :, 3] == 255).reshape(128, 8, 128, 8).all(axis=(1, 3))
    png, mask = folder / "registered_rgb.png", folder / "valid_native128.png"
    Image.fromarray(image[:, :, :3]).save(png)
    Image.fromarray(support.astype(np.uint8) * 255).save(mask)
    return dict(descriptor, image_path=str(png), sha256=digest(png), image_crs=grid["native_crs"],
                image_transform=list(transform)[:6], native128_valid_mask_path=str(mask),
                native128_valid_mask_sha256=digest(mask), native128_valid_fraction=float(support.mean()),
                rgb_pixel_sha256=sha256(image[:, :, :3].tobytes()).hexdigest(),
                bounds_wgs84=grid["bounds_wgs84"])


def landscape(grid, descriptor, scale, output):
    native = Affine(*grid["native_transform"])
    offset = 64 * (1 - scale)
    expanded = native * Affine.translation(offset, offset) * Affine.scale(scale)
    transform = expanded * Affine.scale(128 / SIZE)
    zoom = ZOOM - (1 if scale == 3 else 2)
    if 'label_period' in grid:
        release = descriptor['release']
        if release=='current':
            metadata_url = CURRENT + '/4/query'
        else:
            metadata_url = descriptor['metadata_url'].rsplit('/',2)[0] + f'/{23-zoom}/query'
        footprint = dict(grid, native_transform=list(expanded)[:6])
        evidence = frame_descriptor(footprint, output.parent, release, descriptor['tile_url'], metadata_url, zoom)
        reason = date_rejection(grid['label_period'],evidence)
        if reason:
            raise ValueError('The wider scene fails the label-period date window: ' + reason)
        write(output.parent/'landscape_dates.json',evidence['acquisition_dates'])
    image = mosaic(grid["native_crs"], transform, SIZE, zoom, descriptor["tile_url"])
    Image.fromarray(image[:, :, :3]).save(output)
    return transform_bounds(grid["native_crs"], "EPSG:4326", *array_bounds(SIZE, SIZE, transform))
