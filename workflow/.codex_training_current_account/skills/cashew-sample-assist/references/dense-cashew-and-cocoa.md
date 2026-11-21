# Dense Cashew and mixed orchards

## Choose a date

Select a sharp, well-registered scene inside the requested period. For open areas, clear boundaries are often sufficient. If connected green canopy hides the difference between Cashew and neighboring trees, compare a locally dry view in the same period. Compare matching locations, crown surfaces and parcel edges across dates. Check Google Earth and Esri capture dates separately. Move to a clearer scene when haze, deep shade or seams obscure the relevant detail. [Image selection](imagery-selection.md)

## Read crowns and land units

- At the full-image scale, locate canopy edges, tracks and farm clearings. Inspect the interior for repeated, similarly sized, rounded crown clusters. Dense Cashew can be dark green, with overlapping crowns and indistinct rows. Compare similarly lit crowns next to it and check whether boundaries and crown patterns recur on another date. A confirmed orchard can include ordinary inter-crown ground.
- In CIV and Ghana, inspect the outer belt, interior and transition separately. A Cashew margin can surround a darker or taller cocoa or mixed canopy. Compare color, crown relief and layers. A manual polygon supports its own scope. Extend the classification only to independently recognizable adjacent areas. Use 10 where identity remains ambiguous.
- In Senegal, examples include closed canopy and open edges with changing density. Compare crown roundness, gaps and organization at the same scale. Clearing, roads and new openings across dates may reflect actual land-cover change.

## Reviewed examples

Retrieve the marked area and RGB by ID. The scope shown in the review supports the classification. [Dense examples](http://127.0.0.1:18780/library?scope=dense)

| Reference | Visible evidence |
| --- | --- |
| `GOOGLE_DENSE_MANUAL_cashew_previous_0005` (CIV) | Esri capture dates 2023-11-21 and 2025-12-22. Similarly sized round crowns touch. The western clearing and straight eastern field edge locate the core. |
| `GOOGLE_DENSE_MANUAL_cashew_previous_0007` (CIV) | 2025-01-24 shows overlapping round crowns alongside irregular, darker riparian trees. The southern track and parcel angle align with 2020-01-20, when the crown contrast is weaker. |
| `GOOGLE_DENSE_MANUAL_cashew_previous_0019` (Ghana) | 2022-02-02 shows bright tracks and openings. 2024-12-28 better separates the round crowns from the coarser canopy to the southeast. The closed core matches across dates. |
| `GOOGLE_DENSE_MANUAL_cashew_previous_0028` (Ghana) | 2024-12-02 shows repeated tops in a dark, nearly closed canopy. 2020-04-13 shows the western opening more clearly, with shadow in the east. |
| `GOOGLE_DENSE_MANUAL_historical_b_cashew_0116` (Senegal) | 2022-11-29 shows the crown repetition and texture. The northern opening is clearer in 2024-01-28. Both scenes show connected crowns near the original observation. |
| `GOOGLE_DENSE_MANUAL_historical_b_cashew_0009` (Senegal) | 2025-03-01 shows round crowns, gaps and the southern track. 2022-11-29 shows the same parcel, with shadow along its edge. |
| `GOOGLE_DENSE_MANUAL_historical_b_cashew_0152` (Senegal) | 2022-02-13 shows the road and southern open area. In 2020-12-02 some crown tops are visible and the spacing is difficult to estimate in the most tightly packed canopy. |

## Northwestern CIV tile `0000000000000000012d`

Three locations fall in Cashew on the BNETD 2020 map. Google Earth views from 2025-01-10 and 2021-01-28 show connected canopy at the same locations. The 2025 view shows recurring round crowns more clearly. Compare paths and openings in both views. Follow visible boundaries and retain uncertainty near coarser crowns and mixed transitions. [BNETD dataset](https://developers.google.com/earth-engine/datasets/catalog/BNETD_land_cover_v1)

| Reference | Longitude, latitude | Visible cue |
| --- | --- | --- |
| `GOOGLE_CIV_0012D_NW_BNETD_WEST` | −2.710305, 7.991155 | Fine, fairly even connected crowns beside coarser and darker crowns to the east. |
| `GOOGLE_CIV_0012D_NW_BNETD_SOUTH` | −2.707523, 7.987245 | A track and open plots adjoin the closed canopy on both dates. |
| `GOOGLE_CIV_0012D_NW_BNETD_EAST` | −2.705895, 7.991695 | Repeated round crowns beside fields, with indistinct planting rows. |

The original Ghana PlanetMap has value 1 at all three centers and the remapped patch is mainly class 2 nearby. The local CIV TIFF has value 10 in those neighborhoods. Both versions originate from PlanetMap. On CIV tile `0015a`, the raw map agrees near manual example `0007`. Near `0005`, the patch is mostly 1 while the original center is 0. Keep the manual area and actual RGB evidence. [Northwestern checks](@WORKSPACE@/outputs/cashew_reference_library_google_20261118/dense_cashew_supplement/PLANET_NW_COMPARISON.json) · [CIV manual-location checks](@WORKSPACE@/outputs/cashew_reference_library_google_20261118/dense_cashew_supplement/PLANET_CIV_MANUAL_COMPARISON.json)

Ghana reference `GOOGLE_GHA_PLANET_00B0_CORE` is near −2.721940°, 7.946183°. The original PlanetMap value 1 and remapped class 2 agree at the core. The 2025-01-10 view shows rounded crowns and seams. The 2021-01-28 view makes the openings clearer. Check mixed boundaries against the local photos and manual examples.

## Compare crop maps

Record the source, map year, class and coverage when a map helps locate a lookalike. Check the same patch on the actual RGB. Use 10 where its visual identity stays unresolved.

- **Côte d’Ivoire.** The 10 m BNETD 2020 map codes cocoa as 9 and Cashew as 13. Keep both evidence records when the map and a manual polygon disagree. In the corrected pair, `0013` is Cashew with recurring blunt-rounded crown lobes within the marked scope. `0008` is Mix tree for visual comparison. [BNETD dataset](https://developers.google.com/earth-engine/datasets/catalog/BNETD_land_cover_v1) · [CIV review](http://127.0.0.1:18780/library?scope=civ-south)
- **Nigeria.** Forest Data Partnership cocoa probabilities are at 10 m for 2018, 2020, 2022 and 2024. Valid 2024 data were checked in small windows in Ondo, Cross River and Kogi. Use the original local `planetmap/planet_map/Nigeria.tif` for comparison. The cocoa layer uses AEF. Record that shared input in AEF experiments. [FDP dataset](https://developers.google.com/earth-engine/datasets/catalog/projects_forestdatapartnership_assets_cocoa_model_2026a) · [Coverage checks](../../../../outputs/cashew_reference_library_google_20261118/civ_south_supplement/COCOA_2026A_COVERAGE_PROBE.json)
- **Ghana.** ETH Zürich cocoa probability and binary maps cover Ghana and CIV. Compare the local Ghana PlanetMap and manual Cashew polygons. [ETH data](https://doi.org/10.3929/ethz-b-000654400)
- **Benin.** TechnoServe provides manual Cashew labels across about 120 km² of central Benin. The national study's distribution and density maps are available by request to the authors. [Open labels](https://source.coop/technoserve/cashews-benin) · [National study](https://www.sciencedirect.com/science/article/pii/S0034425723002468)
- **Guinea-Bissau.** ALFICC provides a readable 2021 10 m GeoTIFF spanning Guinea-Bissau and adjoining parts of Guinea. [ALFICC repository](https://github.com/mrp2106/ALFICC)
- **Togo and Senegal.** The existing local PlanetMaps can guide inspection. Open Togo cropland probabilities and Senegal land-cover vectors cover additional land uses. [Togo cropland](https://zenodo.org/records/3836629) · [Senegal land cover](https://geosenegal.gouv.sn/-donnees-vectorielles-d-occupation-du-sol-.html)

The operator corrected the marked areas of CIV `0002` and `0009` to Mix tree. Review the actual RGB and the corrected scope. Use class 1 for identifiable non-Cashew trees, including cocoa and rubber. [CIV Mix tree](http://127.0.0.1:18780/library?scope=civ-mixed) · [Country context](country-background.md)
