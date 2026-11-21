# Review the coordinate reference library

The parent review archive is `outputs/cashew_reference_library_v2_20261117`. Review the research copy's active references here. [Study compact library](../assets/coordinate-reference-library-compact/index.html)

The November 18 dual-source collection pairs Google Earth captures checked against the actual RGB with representative Esri examples. The review records distinguish model screening from operator approval. Keep useful variation across all classes and in orchard density, including visually similar negative examples. [Balanced library](http://127.0.0.1:18780/library?scope=balanced) · [Google additions](http://127.0.0.1:18780/library?scope=google)

The point-study library has 30 examples. It includes eleven Cashew, twelve Mix tree, four cropland and other, and three built-up examples. The additions span sparse and dense orchards, drainage woodland, scattered trees within fields and compounds on an orchard edge. Four matched TechnoServe examples show irregularly spaced orchards and neighbouring non-plantation woodland. These examples pair field-derived 2020 labels with verified 2020 RGB. Ordinary lookup excludes reserved evaluation origins. The study excludes reference footprints that overlap its independent query scenes. [Study compact library](../assets/coordinate-reference-library-compact/index.html) · [TechnoServe label documentation](https://data.source.coop/technoserve/cashews-benin/Documentation.pdf)

Prioritize gaps in geography and visual appearance. Finish when Cashew examples cover sparse, variable and dense patterns and other classes have useful representative coverage. Every eligible example can be retrieved on demand. Fixed-size sets are unnecessary. For the authorized reference curation pass, up to ten independent batches can run concurrently. Avoid duplicate submission of an active turn.

`library_catalog.json` records sources. Keep original labels, geometries and versions in `original_label`, `geometry` and `source_annotations`. Store new labels and polygons separately in `proposed_label` and `proposed_geometry`. Where the input label is binary Non-cashew, a more specific class from the RGB is a model proposal.

For new examples, use Sol max and X-AnyLabeling SAM2 when a mask is useful. Show manual class evidence with its exact scope. Confirm local homogeneity and a visible boundary before extending that area. If a lookalike or mixed boundary remains unclear, retain uncertainty. SAM2 uses the full registered 1024 image. Inspect source detail at its own resolution when needed. Query the authorized curation scope with `run_curation_queue.py --scope-manifest DIVERSITY_REVIEW_SCOPE.json --status`.

`merge_review_results.py` admits cases with verified visual delivery. `curated_references()` reads the current decisions in `human_reviews`, including removals and class corrections. The November 18 UTC approval receipt records 128 retained and five rejected cases. The initial Google selection had 41 approvals and one explicit rejection, `GOOGLE_REF_GHA_TRAIN_POS_03`. Exclude it from retrieval. [Evidence directory](@WORKSPACE@/outputs/cashew_reference_library_google_20261118)

The operator approved 14 southern CIV Cashew examples. The original manual areas of `0002` and `0009` were corrected to Mix tree and removed from the current Cashew-positive bank. The Mix tree proposal for `0003` was withdrawn. Use the actual RGB and the approved scopes to compare other woody patches. Retain uncertain or mixed edges as 10. [CIV Mix tree review](http://127.0.0.1:18780/library?scope=civ-mixed) · [Dense and mixed orchards](dense-cashew-and-cocoa.md)

The tile-trial library adds the operator-confirmed GHA B05 Mix tree lookalike. The green polygons identify representative interiors within the corrected woody areas. Compare the January 2022 and November 2024 views at the same locations. The annotation trial excludes this reference whenever the current tile intersects its footprint. [GHA B05 reference](http://127.0.0.1:18780/library/woody-background-review/#GOOGLE_GHA_V5_B05_MIXTREE_HUMAN)

GHA B04 is a pending Mix tree candidate. Its dated RGBs and five disputed areas are stored in `review_candidates` for visual review. The reviewed map assigns those areas class 10. Admit a candidate to the reference list only after its class and scope are confirmed. [GHA B04 candidate](http://127.0.0.1:18780/library/woody-background-review/#GOOGLE_GHA_V5_B04_MIXTREE_PENDING)

Reference imagery teaches appearance. Retain any known capture dates as provenance. Matching the target date window is required for the annotation image. A 10 m review grid without the original AEF affine is a preview. Use the actual AEF footprint before importing its labels into training.

The operator authorized a dedicated curation pass to examine manual samples and evaluation locations. Record any such location's development use and exclude it from independent evaluation claims. Training control and ordinary model inputs still exclude target evaluation labels and metrics.

Read `reference_prompt_notes.md` in the working directory for evidence behind the earlier class cues. Base updates on confirmed observations.
