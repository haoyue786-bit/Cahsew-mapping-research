# Supplementary Tables and Figures

| Material | Contents |
| --- | --- |
| [Table S1. AEF year selection](tile-years.csv) | 676 tile IDs, Sentinel-2 scene counts for four candidate years, and the selected AEF year for each period. |
| [Table S2. Group membership](group-membership.csv) | Tile membership for all 90 groups. Join with Table S1 on tile ID to find the actual imagery years used in each group. |
| [Table S3. Reference sample counts](reference-point-counts.csv) | Counts for the full reference set by country and land-cover class, including distinct locations and interpretation records across the two periods. |
| [Figure S1. Selected AEF years](selected-aef-years.pdf) · [PNG preview](selected-aef-years.png) | The selected imagery year for each period across the same tile footprint. The PDF contains vector graphics. |

Tile IDs are strings. Preserve leading zeros when loading CSVs. The group membership table uses the composite key `group_id, tile_id`. Some formal processing groups share tiles. Count distinct tile IDs when calculating national or study-wide coverage.

**Figure S1.** Within each period and tile, count the Sentinel-2 scenes that intersect the tile and have less than 60% scene-level cloud cover. Use AEF from the year with more scenes. The selection assigns all 676 early-period tiles to 2019. For the late period, it assigns 58 tiles to 2024 and 618 tiles to 2025.

The table and figure generation script uses exported Sentinel-2 counts, the tile grid, and the frozen group roster. It checks period matching by tile ID, imagery years in the late-period inventory, and complete group coverage. [Generation script](../../tools/build_manuscript_tables.py) · [Year-selection method](../../docs/manuscript-supplement.md)

The reference point summary script aggregates 22 evaluation layers across 11 countries offline. It deduplicates locations using WGS84 coordinates rounded to seven decimal places and counts land-cover classes by interpretation record. Table S3 covers the full reference set. The evaluation for each map reports how many reference points the map covers. [Reference point summary script](../summarize_reference_points.py)
