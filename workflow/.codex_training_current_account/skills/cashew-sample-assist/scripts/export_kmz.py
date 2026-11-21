#!/usr/bin/env python3
"""Package completed AEF labels and explicitly skipped tile footprints into one KMZ."""

import argparse
import io
from pathlib import Path
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree as ET
import zipfile

import numpy as np
from PIL import Image
import rasterio
from rasterio import Affine

from _runtime.common import digest, read
from annotation_geometry import COLORS, GRID
from imagery.coverage import read_support
from imagery_policy import review_rejection


KML = "http://www.opengis.net/kml/2.2"
ET.register_namespace("", KML)
CLASSES = {0: "Cropland and other", 1: "Mix tree, wetland and water", 2: "Cashew", 3: "Built-up", 10: "Uncertain"}


def tag(parent, name, value=None):
    result = ET.SubElement(parent, f"{{{KML}}}{name}")
    if value is not None:
        result.text = str(value)
    return result


def open_labels(receipt):
    grid = receipt["grid"]
    filename = Path(receipt["label_geotiff"])
    frame_path = Path(receipt["selection"])
    if digest(frame_path) != receipt["selection_sha256"] or digest(filename) != receipt["label_sha256"]:
        raise ValueError("Selection or label checksum differs from the final receipt")
    selected = read(frame_path)
    if selected["grid"] != grid or selected["status"] != "selected":
        raise ValueError("Selected source does not belong to this AEF grid")
    for audit_path in (selected["vision_audit"], receipt["vision_audit"]):
        if read(audit_path)["status"] != "verified":
            raise ValueError("Visual input verification is missing")
    frame = selected["frame"]
    if review_rejection(grid["label_period"], frame, selected["review"]):
        raise ValueError("Selected imagery fails the date or scope requirements")
    with rasterio.open(filename) as raster:
        if (raster.shape != (GRID, GRID) or raster.count != 1 or raster.crs.to_string() != grid["native_crs"]
                or raster.transform != Affine(*grid["native_transform"])):
            raise ValueError("Label GeoTIFF differs from the original AEF grid")
        labels = raster.read(1)
        bounds = tuple(raster.bounds)
    if np.any(labels[~read_support(frame)] != 10):
        raise ValueError("Pixels without RGB support must have class 10")
    with Image.open(filename.with_name("labels_native128.png")) as preview:
        if not np.array_equal(labels, np.asarray(preview)):
            raise ValueError("Native GeoTIFF and PNG label values differ")
    counts = {str(code): int((labels == code).sum()) for code in np.unique(labels)}
    if set(map(int, counts)) - set(CLASSES) or counts != receipt["class_pixel_counts"]:
        raise ValueError("Invalid codes or inconsistent label counts")
    return labels, bounds, frame


def overlay_png(labels, valid=None):
    values = np.zeros((GRID, GRID, 4), dtype=np.uint8)
    for code, color in COLORS.items():
        values[labels == code] = (*color, 130 if code == 10 else 215)
    if valid is not None:
        values[~valid, 3] = 0
    data = io.BytesIO()
    Image.fromarray(values).resize((512, 512), Image.Resampling.NEAREST).save(data, format="PNG")
    return data.getvalue()


def append_skipped(folder, receipt):
    grid = receipt["grid"]
    west, south, east, north = grid["bounds_wgs84"]
    mark = tag(folder, "Placemark")
    tag(mark, "name", grid["candidate_id"] + " · Unannotated")
    reason = receipt.get("reason_text", "No usable imagery")
    tag(mark, "description", f"{grid['label_period']} period · {reason}")
    line = tag(mark, "LineString")
    tag(line, "coordinates", " ".join(f"{x},{y},0" for x, y in
        [(west, south), (east, south), (east, north), (west, north), (west, south)]))


def append_labels(kmz, folder, receipt):
    grid = receipt["grid"]
    labels, bounds, frame = open_labels(receipt)
    if (grid["native_crs"] != "EPSG:4326" or grid["native_transform"][1] or grid["native_transform"][3]
            or grid["native_transform"][0] <= 0 or grid["native_transform"][4] >= 0):
        raise ValueError("KMZ GroundOverlay requires a north-up WGS84 AEF grid")
    overlay = tag(folder, "GroundOverlay")
    tag(overlay, "name", f"{grid['candidate_id']} · {grid['label_period']}")
    dates = ", ".join(frame["acquisition_dates"])
    tag(overlay, "description", f'{frame["provider"]} · Captured {dates}')
    path = f"overlays/{grid['candidate_id']}.png"
    tag(tag(overlay, "Icon"), "href", path)
    coordinates = tag(overlay, "LatLonBox")
    for name, value in zip(("west", "south", "east", "north"), bounds):
        tag(coordinates, name, repr(value))
    valid = None
    if receipt.get('resampling') == 'categorical_area_mode':
        with rasterio.open(receipt['label_geotiff']) as raster:
            valid = raster.dataset_mask() != 0
    kmz.writestr(path, overlay_png(labels, valid))


def export(receipts, output):
    root = ET.Element(f"{{{KML}}}kml")
    document = tag(root, "Document")
    tag(document, "name", output.stem)
    tag(document, "description", " · ".join(f"{code} {name}" for code, name in CLASSES.items()))
    folders = {}
    cases = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".kmz.partial", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as kmz:
            for receipt_path in receipts:
                receipt = read(receipt_path)
                grid = receipt["grid"]
                case, country = grid["candidate_id"], grid["country"]
                if case in cases:
                    raise ValueError("Repeated case in the combined KMZ: " + case)
                cases.add(case)
                if country not in folders:
                    folder = folders[country] = tag(document, "Folder")
                    tag(folder, "name", country)
                if receipt["status"] == "completed":
                    append_labels(kmz, folders[country], receipt)
                elif receipt["status"] == "skipped":
                    append_skipped(folders[country], receipt)
                else:
                    raise ValueError("Sample has no terminal output: " + case)
            kmz.writestr("doc.kml", ET.tostring(root, xml_declaration=True, encoding="UTF-8"))
        with zipfile.ZipFile(temporary) as kmz:
            if kmz.testzip() is not None:
                raise ValueError("KMZ archive verification failed")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("receipts", type=Path, nargs="+", help="One RECEIPT.json for each sample")
    args = parser.parse_args()
    export(args.receipts, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
