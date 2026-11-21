# West African Cashew Mapping Supplement

## S1. Background and study area

Yin et al. mapped smallholder cashew plantations across Benin using time series of 2.4 m Planet Basemaps and 0.5 m aerial imagery. This study extends mapping to West Africa using consistent annual remote sensing features and transferable models. It maps cashew distribution at 10 m resolution for two periods and uses labeled data from Benin to train the source-domain model. [Yin et al. (2023)](https://doi.org/10.1016/j.rse.2023.113695)

The study covers Benin, Burkina Faso, Côte d’Ivoire, Gambia, Ghana, Guinea, Guinea-Bissau, Mali, Nigeria, Senegal, and Togo. The 11-country study footprint follows the mapping grid and tile inventory. Cashew cultivation in Liberia and Sierra Leone was too limited for inclusion.

## S2. Remote sensing data and year selection

The classifier and tile grouping use the **Satellite Embedding V1 annual product** from Google Earth Engine (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`). AlphaEarth Foundations (AEF) integrates multiple Earth observation sources into a 64-dimensional vector that represents land surface conditions at each 10 m pixel for a calendar year. Sentinel-2 scene counts guide the choice of imagery year for each tile. [AEF data catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) · [AlphaEarth Foundations paper](https://arxiv.org/abs/2507.22291)

1. Use a fixed geographic tile grid. Compare **2018 with 2019** for the early period and **2024 with 2025** for the late period.
2. For each tile and candidate calendar year, count Sentinel-2 surface reflectance scenes in `COPERNICUS/S2_SR_HARMONIZED` that intersect the tile and satisfy `CLOUDY_PIXEL_PERCENTAGE < 60`. The scene count is a proxy for observation availability. Usable observations at a given pixel depend on local cloud, shadow, and image coverage. [Sentinel-2 data catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
3. For each period, choose the year with the higher count. If the counts tie, choose the later year. Read and crop the AEF embedding for the chosen calendar year to the tile. Each tile uses one annual embedding per period.
4. Record the fixed tile ID and actual AEF year. The original year-selection grid contains 707 tiles. The table summarizes the 676 unique tiles included in the mapping groups.

| Period | Candidate years | Tiles assigned to the first year | Tiles assigned to the second year |
| --- | --- | ---: | ---: |
| Early | 2018 / 2019 | 0 (2018) | 676 (2019) |
| Late | 2024 / 2025 | 58 (2024) | 618 (2025) |

The table summarizes year selection based on observation counts. Check the tile ID and actual year of each input image against the inventory. The “2025 map” covers the late period, including tiles where the selected embedding is from 2024. [Tile year inventory](../methods/data/tile-years.csv)

## S3. Domain adaptation groups based on features and spatial adjacency

**Purpose.** West African landscapes vary in vegetation, rainfall, crop composition, and land use. Grouping geographically contiguous tiles with similar AEF features lets comparable landscapes share a domain adaptation model and pool training samples. Groups can cross national borders, as in Benin–Togo and Gambia–Senegal–Guinea-Bissau.

**Tile representation.** For each late-period tile, draw a spatially stratified sample and use valid pixels to compute the mean and standard deviation of each of the 64 embedding dimensions. Apply L2 normalization separately to the two vectors. Before concatenation, multiply the mean vector by √0.8 and the standard deviation vector by √0.2. The two components then contribute with weights of 80% and 20% to squared feature distance. The mean captures overall land surface characteristics, while the standard deviation captures within-tile heterogeneity.

**Spatial constraints.** Build an adjacency graph from tiles that touch. Use adjacency-constrained Ward clustering that accounts for embedding dissimilarity and the geographic distance between tile centers. At each step, Ward clustering merges adjacent clusters that produce the smallest increase in within-group dispersion. Compare partitions based on feature similarity within groups, spatial continuity, geographic extent, and sample size. The reproducible grouping code caps the maximum distance between tile centers within a group at 350 km and prioritizes groups with at least three tiles and 4,000 usable image patches. Compare cluster separation using the silhouette score. Assess stability using the adjusted Rand index across two stratified subsamples. Target-domain reference labels and evaluation scores are excluded from feature construction and grouping.

**Grouping result.** After deduplicating tiles downloaded for multiple countries, the study has **676 unique tiles in 90 groups**. Each group uses the same tile membership in both periods, giving **180 group-period mapping tasks**. The membership table associates tiles with groups. Coverage statistics count distinct tile IDs. [Group membership and maps](../methods/data/README.md)

## S4. Domain adaptation and map production

1. **Late period.** Initialize each group from the same Benin source-domain weights with a fresh optimizer state. The model performs four-class, pixel-level semantic segmentation. Labeled source-domain imagery provides supervision. Unlabeled target imagery supports domain alignment and monitoring on a fixed panel.
2. **Hard examples.** Every five completed epochs, select target areas for review based on competition among predicted classes and prediction uncertainty. Use high-resolution imagery with verified acquisition dates on or after 1 January 2024 for class interpretation and boundary delineation. Supplementary pixel labels contribute to four-class cross-entropy only after acceptance and alignment to the original 10 m grid. Record sample provenance and use.
3. **Early period.** Initialize from the frozen late-period checkpoint for the same group and start with a fresh optimizer state. Use the corresponding early-period AEF imagery. Disable automated annotation and exclude the late-period hard-example library.
4. **Tuning, stopping, and checkpoint selection.** The agent uses source-domain validation, unlabeled target imagery, model and gradient diagnostics, observed responses to interventions, and supplementary training labels already used by the group. Stop decisions consider the marginal reduction in comparable losses. After stopping, compare candidate checkpoints under the same deployment configuration and freeze the checkpoint for mapping.
5. **Map outputs.** Each group produces two versions, both in GeoTIFF and KMZ formats. The four-class argmax map assigns each pixel to the class with the highest predicted probability. The confidence-screened version reassigns pixels whose argmax class is cashew and whose cashew probability is below 0.5 to uncertain (code 10). The four classes are cropland and other open land (0), other woody vegetation, wetlands, or water (1), cashew (2), and built-up land (3).

## S5. Reference samples and map evaluation

**Sampling design.** Use grid-based spatially stratified random sampling for the initial reference points. Divide the study area into fine grid cells and treat each sampled cell as a spatial stratum. Select locations at random within each stratum to distribute the base sample across landscapes. Visually interpret historical high-resolution Google Earth imagery and assign each reference record to one of the four land-cover classes defined in S4. Then search within the same sampled cells for visually confirmed cashew sites to increase the sample size for that class. [Stehman and Foody (2019)](https://doi.org/10.1016/j.rse.2019.05.018)

**Table S3. Reference sample counts.** After deduplication by coordinates, the reference set contains **8,313 distinct locations** and **16,461 interpretation records**, including **2,291 cashew records**. A single location can have two interpretation records. The linked table includes counts for all four classes. [Reference sample counts](../methods/data/reference-point-counts.csv)

| Country | Distinct reference locations | All interpretation records | Cashew records |
| --- | ---: | ---: | ---: |
| Benin | 468 | 913 | 366 |
| Burkina Faso | 508 | 1,015 | 24 |
| Côte d’Ivoire | 1,528 | 3,039 | 1,070 |
| Gambia | 6 | 12 | 0 |
| Ghana | 689 | 1,347 | 153 |
| Guinea | 733 | 1,465 | 64 |
| Guinea-Bissau | 138 | 274 | 153 |
| Mali | 1,055 | 2,095 | 64 |
| Nigeria | 2,400 | 4,728 | 295 |
| Senegal | 585 | 1,170 | 76 |
| Togo | 203 | 403 | 26 |
| **Total** | **8,313** | **16,461** | **2,291** |

**Accuracy assessment.** Use target-domain reference points after the model has been frozen. Record reference coverage, checkpoint identity, and the rule used to convert probabilities into map classes so that F1, precision, and recall can be traced to each output. The provenance fields document collection or curation sources, with limited information for separating the initial random sample from the supplemental sample. Pooled metrics therefore describe performance on the combined reference set. Design-based estimates for the full study area require verifiable inclusion probabilities and corresponding weights. When a threshold is chosen to maximize F1 on these same points, report the score as a post hoc fit. [Probability sampling and map accuracy assessment](https://doi.org/10.1016/j.rse.2019.05.018)
