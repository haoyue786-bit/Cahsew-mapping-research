"""Rebuild the manuscript tables and maps from the repository's frozen inputs."""

import argparse
from pathlib import Path

from build_manuscript_tables import csv_index, make_tables, map_years, read_json, write_table


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "methods/data/source"
GROUPING = ROOT / "methods/grouping"
PLAN = ROOT / "workflow/experiments/cashew_fada_formal2025_20261120_v1/cohort_2025.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    descriptors = read_json(GROUPING / "aef64-descriptors.json.gz")
    remaining = {row["record"]["tile_id"]:
                 dict(row["record"], image_year=str(row["record"]["image_year"]))
                 for row in descriptors}
    years, groups = make_tables(
        csv_index(DATA / "West_Africa2018_s2_count_cloudlt60.csv"),
        csv_index(DATA / "West_Africa2019_s2_count_cloudlt60.csv"),
        csv_index(DATA / "West_Africa2024_2025_s2_selection_cloudlt60.csv"),
        csv_index(GROUPING / "inventory.csv"),
        read_json(PLAN),
        remaining)
    args.output.mkdir(parents=True)
    write_table(args.output / "tile-years.csv", years)
    write_table(args.output / "group-membership.csv", groups)
    map_years(args.output, years,
              read_json(DATA / "West_Africa2024_2025_s2_selection_cloudlt60.geojson.gz"))


if __name__ == "__main__":
    main()
