"""Measure Sentinel-2 coverage or export chosen annual AEF tiles from Earth Engine."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory


S2 = "COPERNICUS/S2_SR_HARMONIZED"
AEF = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
YEARS = (2018, 2019, 2024, 2025)
GRID = (Path(__file__).resolve().parent /
        "data/source/West_Africa2024_2025_s2_selection_cloudlt60.geojson.gz")
FROZEN_YEARS = Path(__file__).resolve().parent / "data/tile-years.csv"


def counts(ee, features):
    s2 = ee.ImageCollection(S2)

    def measure(feature):
        # Scene-level observation count on the same geographic footprint for each year.
        values = {}
        for year in YEARS:
            scenes = (s2.filterBounds(feature.geometry())
                      .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
            values[f"s2_count_{year}"] = scenes.size()
        return feature.set(values).setGeometry(None)

    for start in range(0, len(features), 25):
        batch = ee.FeatureCollection(features[start:start + 25])
        for feature in batch.map(measure).getInfo()["features"]:
            props = feature["properties"]
            row = {f"s2_count_{year}": props[f"s2_count_{year}"] for year in YEARS}
            early = 2018 if row["s2_count_2018"] > row["s2_count_2019"] else 2019
            late = 2024 if row["s2_count_2024"] > row["s2_count_2025"] else 2025
            yield dict(tile_id=props["tile_id"], **row, early_aef_year=early, late_aef_year=late)


def export(ee, features, choices, period, destination, country):
    import geemap
    import rasterio

    destination.mkdir(parents=True, exist_ok=True)
    collection = ee.ImageCollection(AEF)
    progress = destination / f"{country}_{period}_progress.txt"
    for tile_id, feature in features:
        year = choices[tile_id][f"{period}_aef_year"]
        target = destination / f"{country}{year}_{tile_id}.tif"
        if target.is_file():
            with rasterio.open(target) as raster:
                if raster.count != 64 or raster.crs.to_epsg() != 4326:
                    raise ValueError(f"Invalid existing raster: {target}")
            with progress.open("a", encoding="utf-8") as stream:
                stream.write(f"EXIST\t{year}\t{tile_id}\t{target.name}\n")
            continue
        # Fetch the selected annual embedding in the original 64-band CRS and grid convention.
        geometry = feature.geometry()
        annual = collection.filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic()
        with TemporaryDirectory(prefix=".aef-", dir=destination) as tmp:
            scratch = Path(tmp) / target.name
            geemap.download_ee_image(annual.clip(geometry), filename=str(scratch),
                                     region=geometry, scale=10, crs="EPSG:4326")
            with rasterio.open(scratch) as raster:
                if raster.count != 64 or raster.crs.to_epsg() != 4326:
                    raise ValueError(f"Unexpected AEF bands or CRS for {tile_id}")
            scratch.replace(target)
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(f"OK\t{year}\t{tile_id}\t{target.name}\n")
        print(target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Your authorized Google Earth Engine project")
    parser.add_argument("--grid", type=Path, default=GRID)
    parser.add_argument("--choices", type=Path, default=FROZEN_YEARS)
    parser.add_argument("--tile", action="append", default=[])
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--counts", type=Path)
    modes.add_argument("--download", type=Path)
    parser.add_argument("--period", choices=("early", "late"))
    parser.add_argument("--country", help="Inventory country name, for example Ghana")
    args = parser.parse_args()
    if args.download and (not args.period or not args.country or not args.tile):
        parser.error("--download requires --period, --country and at least one --tile")

    with args.choices.open(encoding="utf-8", newline="") as stream:
        choices = {row["tile_id"]: row for row in csv.DictReader(stream)}
    read = gzip.open if args.grid.suffix == ".gz" else open
    with read(args.grid, "rt", encoding="utf-8") as stream:
        grid = {f["properties"]["tile_id"]: f["geometry"] for f in json.load(stream)["features"]}
    requested = sorted(set(args.tile) if args.tile else set(choices))
    if set(requested) - set(grid) or args.download and set(requested) - set(choices):
        parser.error("Some requested tile IDs are absent from the grid or selected-year table")

    import ee
    ee.Initialize(project=args.project)
    features = [(tile, ee.Feature(ee.Geometry(grid[tile]), {"tile_id": tile})) for tile in requested]
    if args.counts:
        measured = list(counts(ee, [feature for _, feature in features]))
        with args.counts.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, list(measured[0]))
            writer.writeheader()
            writer.writerows(measured)
    else:
        export(ee, features, choices, args.period, args.download, args.country)


if __name__ == "__main__":
    main()
