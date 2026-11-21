"""Build the manuscript's year-selection table, group roster, and map."""

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path


def csv_index(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {row["tile_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Duplicate tile ID in {path}")
    return index


def read_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_tables(early_first, early_second, late, inventory, cohort, remaining):
    # Stable tile IDs match the two periods and collapse cross-country downloads.
    footprint = set(inventory)
    if set(early_first) != set(early_second) or set(early_first) != set(late):
        raise ValueError("The Sentinel-2 exports must describe the same tile grid")
    if not footprint <= set(late):
        raise ValueError("Some registered tiles lack a Sentinel-2 count record")

    members = {}
    for job in cohort["jobs"]:
        members[job["name"].replace("_y2025_", "_")] = set(job["members"])
        for tile_id, directory in job["members"].items():
            if f'{late[tile_id]["selected_year"]}_{tile_id}' not in Path(directory).name:
                raise ValueError(f"Late AEF inventory does not match the selected year for {tile_id}")
    formal_tiles = set().union(*members.values())
    if set(remaining) != footprint - formal_tiles:
        raise ValueError("The two sets of memberships must cover the registered footprint")
    for tile_id, row in remaining.items():
        if row["image_year"] != late[tile_id]["selected_year"]:
            raise ValueError(f"Late AEF inventory does not match the selected year for {tile_id}")
        members.setdefault(f'WA_group{int(row["group"]):03d}', set()).add(tile_id)

    # The scene-count rule compares whole calendar years within each period.
    tile_years = []
    for tile_id in sorted(footprint):
        c18 = int(early_first[tile_id]["s2_count_2018"])
        c19 = int(early_second[tile_id]["s2_count_2019"])
        c24 = int(late[tile_id]["s2_count_2024"])
        c25 = int(late[tile_id]["s2_count_2025"])
        chosen_late = 2024 if c24 > c25 else 2025
        if chosen_late != int(late[tile_id]["selected_year"]):
            raise ValueError(f"Late selection record disagrees with counts for {tile_id}")
        tile_years.append(dict(tile_id=tile_id, s2_count_2018=c18, s2_count_2019=c19,
                               s2_count_2024=c24, s2_count_2025=c25,
                               early_aef_year=2018 if c18 > c19 else 2019,
                               late_aef_year=chosen_late))
    group_rows = [dict(group_id=group, tile_id=tile_id)
                  for group, tiles in sorted(members.items()) for tile_id in sorted(tiles)]
    return tile_years, group_rows


def map_years(out, tile_years, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    from shapely.geometry import shape

    def polygon_parts(geometry):
        if geometry.geom_type == "Polygon":
            yield geometry.exterior.coords
        elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
            for part in geometry.geoms:
                yield from polygon_parts(part)

    polygons = {}
    for feature in grid["features"]:
        geometry = shape(feature["geometry"])
        polygons[feature["properties"]["tile_id"]] = list(polygon_parts(geometry))

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})
    colors = {2018: "#eab24a", 2019: "#287b77", 2024: "#eab24a", 2025: "#287b77"}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.3), sharex=True, sharey=True,
                             layout="constrained")
    for ax, field, years, label in zip(axes, ("early_aef_year", "late_aef_year"),
                                       ((2018, 2019), (2024, 2025)), ("a", "b")):
        coords, colors_by_polygon = [], []
        for row in tile_years:
            parts = polygons[row["tile_id"]]
            if not parts:
                raise ValueError(f"No polygon geometry for {row['tile_id']}")
            coords.extend(parts)
            colors_by_polygon.extend([colors[row[field]]] * len(parts))
        ax.add_collection(PolyCollection(coords, facecolors=colors_by_polygon,
                                         edgecolors="white", linewidth=0.32, zorder=2))
        totals = Counter(row[field] for row in tile_years)
        ax.legend(handles=[Patch(facecolor=colors[year], label=f"{year}  ({totals[year]} tiles)")
                           for year in years], loc="upper right", frameon=False)
        ax.set_title(f"({label})  {years[0]}–{years[1]}", loc="left", fontweight="bold")
        ax.set_ylabel("Latitude (°N)")
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.grid(color="#d9e1e8", linewidth=.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel("Longitude (°E)")
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"selected-aef-years.{suffix}", dpi=220, facecolor="white",
                    metadata={"Title": "AEF years selected for cashew mapping in West Africa"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("early-first", "early-second", "late", "grid", "inventory",
                 "cohort", "remaining", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    tile_years, group_rows = make_tables(
        csv_index(args.early_first), csv_index(args.early_second), csv_index(args.late),
        csv_index(args.inventory), read_json(args.cohort), csv_index(args.remaining))
    args.out.mkdir(parents=True, exist_ok=True)
    write_table(args.out / "tile-years.csv", tile_years)
    write_table(args.out / "group-membership.csv", group_rows)
    map_years(args.out, tile_years, read_json(args.grid))
    print(f"{len(tile_years)} tiles and {len({r['group_id'] for r in group_rows})} group IDs")
    for field in ("early_aef_year", "late_aef_year"):
        print(field, dict(sorted(Counter(row[field] for row in tile_years).items())))


if __name__ == "__main__":
    main()
