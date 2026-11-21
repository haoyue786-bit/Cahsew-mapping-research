"""Summarize the offline reference corpus without publishing point-level data."""

import argparse
from collections import Counter
import csv
from pathlib import Path

import fiona


COUNTRIES = ("BEN", "BFA", "CIV", "GHA", "GIN", "GMB", "GNB", "MLI", "NGA", "SEN", "TGO")


def summarize(root):
    all_sites = set()
    all_classes = Counter()

    def record(country, sites, counts):
        return dict(country=country, unique_sites=len(sites), reference_records=sum(counts.values()),
                    cropland_other_open=counts[0], other_woody_wetland_water=counts[1],
                    cashew=counts[2], built_up=counts[3])

    for country in COUNTRIES:
        sites = set()
        counts = Counter()
        for year in (2019, 2025):
            matches = sorted((root / str(year) / country).glob(f"*_{year}_4class.shp"))
            if len(matches) != 1:
                raise ValueError(f"Expected one reference layer for {country} {year}")
            with fiona.open(matches[0]) as source:
                if source.crs.to_epsg() != 4326:
                    raise ValueError(f"Expected WGS84 reference points for {country} {year}")
                for row in source:
                    geometry = row["geometry"]
                    if geometry is None or geometry["type"] != "Point":
                        raise ValueError(f"Expected a point in the {country} {year} reference layer")
                    # The same reference location may have a record in both periods.
                    sites.add(tuple(round(value, 7) for value in geometry["coordinates"][:2]))
                    label = int(row["properties"]["class_id"])
                    if label not in (0, 1, 2, 3):
                        raise ValueError(f"Unrecognized reference class for {country} {year}")
                    counts[label] += 1
        all_sites.update(sites)
        all_classes.update(counts)
        yield record(country, sites, counts)
    yield record("ALL", all_sites, all_classes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path, help="Offline root containing the 2019 and 2025 layers")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = list(summarize(args.reference_root))
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
