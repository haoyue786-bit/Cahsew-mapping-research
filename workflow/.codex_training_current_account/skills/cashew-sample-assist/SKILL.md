---
name: cashew-sample-assist
description: Interpret Cashew and other land cover in dated Google Earth or Esri imagery with GPT-6.2 Sol max. Use for a marked point or a 128 × 128 AEF sample. Delineate tile labels with X-AnyLabeling SAM2 as needed and export on the original 10 m grid.
---

# Cashew sample annotation

For tile annotation, supply the original 128 × 128 AEF GeoTIFF, country code and label period. Start each sample in its own directory using the [run instructions](references/run-sample.md). Use English for prompts, tool descriptions, reference captions and model responses.

For a marked point, interpret the land unit at the centre of the mark using the [point guide](references/regional-point-evidence.md). Review difficult woody points with the [point evidence checks](references/woody-point-review.md). For polygon and raster annotation, apply the same local decision from several locations inside a visible land unit before tracing the boundary.

## Choose an image

Use the [local imagery preflight](references/imagery-selection.md) to acquire fresh Google Earth and Esri candidates. For a standalone 2019 annotation use captures from 2017 through 2020. For the 2025 label period, use 2025–2026 when at least five distinct usable acquisitions are available and include 2024 when there are fewer. Every used capture must be dated 2024-01-01 or later and must pass registration and coverage checks. The 2025 label period also accepts registered AEF inputs from the corresponding 2024 inventory. The program measures cloud and haze risk, local detail and rainfall season, then prepares the primary view and suitable alternate views.

Explain the four classes in the annotation turn. Class 0 covers cropland and other open land, class 1 other wooded or shrubby cover (including cocoa, rubber and open savanna), wetlands and water, class 2 Cashew orchards, and class 3 built-up land. A single usable dry-season image can establish Cashew where the local crowns are distinguishable. For a non-dry primary image, prefer a usable dry candidate before segmentation. With only non-dry imagery, confirm Cashew using another date of the same footprint that corroborates the local evidence. Keep supported other cover and assign unresolved areas class 10. Record “No usable imagery” when the eligible views fail.

## Label and export

Use `gpt-6.2-sol` at `max` reasoning effort with the standard service tier (`default`). Record the returned tier and check the thread and turn model bindings.

Start the standard annotation turn with the full registered primary RGB and the [four-class visual cues](references/visual-cues.md). Resolve specific doubts with on-demand native-detail crops or reviewed examples. A wider landscape, regional background and an eligible different-date image can also be requested. Use up to ten auxiliary calls per tile across selection attempts. The field detector is optional. Verify delivery of required visual evidence.

Sol submits a whole-tile class, clear direct polygons, or units requiring SAM2. Record local Cashew evidence for either geometry type. Freeze the working image and plan before executing masks. Reuse the full-image embedding and generate one mask per SAM2 unit. The host checks prompts and overlap, applies class 10 to ambiguous or conflicting cover, and writes the RGB and label PNGs plus the original-grid 10 m label GeoTIFF using an area vote.

The optional detector and DINOv2 trial use `--region-matching`. See the [trial instructions](references/run-sample.md) for this mode.

For a screened tile with two clear classes, use the [two-class background rule](references/code-selected-annotation.md). Sol checks the full scene and uses the detector's proposals to prompt SAM2 for one class. The program assigns the other class to the clear remainder and preserves localized uncertainty.

Use `--woody-review` for a tile where Cashew and other woody cover are especially hard to separate. The [independent review](references/woody-tile-review.md) checks the numbered units on the original RGB and any suitable alternate before SAM2. A disagreement is recorded as class 10. Use the compact lookalike references when an authorized reference bundle is installed when crown-scale texture is ambiguous.

| Stage | Scripts | Guidance |
| --- | --- | --- |
| Fresh imagery and quality screening | [Preflight](scripts/imagery_preflight.py), [quality and season](scripts/imagery_quality.py), [date policy](scripts/imagery_policy.py) | [Selection rules](references/imagery-selection.md) |
| Sol interpretation and examples | [Annotation](scripts/annotate.py), [reference lookup](scripts/reference_lookup.py) | [Annotation on screened imagery](references/code-selected-annotation.md), [visual cues](references/visual-cues.md), [compact references](references/compact-reference-20261116.md) |
| Land units and optional masks | [Geometry](scripts/annotation_geometry.py), [X-AnyLabeling SAM2](scripts/xanylabeling_sam2.py) | [Direct labels and SAM2](references/annotation-optional-sam.md) |
| Box-guided trial | [Field detector](scripts/propose_fields.py), [DINOv2 matching](scripts/similar_regions.py), [mask review](scripts/box_review.py) | [Representative selection](references/region-matching-prompt.md), [mask retention](references/keep-masks-prompt.md) |
| 10 m alignment and delivery | [AEF grid](scripts/imagery/grid.py) | [Alignment and outputs](references/dual-source-grid-alignment.md) |
| Model and image checks | [Transport](scripts/_runtime/transport.py), [skill binding](scripts/verify_explicit_skill_binding.py), [image delivery](scripts/_runtime/vision_audit.py) | [Run instructions](references/run-sample.md) |

## References and regional evidence

Retrieve a close positive or lookalike by class or ID when it can resolve a visual question. Match the visible form when relevant. The optional compact reference library retains reviewed scopes, dates and IDs. Apply the [curation rules](references/coordinate-reference-curation.md). Use only sample geometry and authorized visual evidence. Seal target evaluation labels, prior predictions and scores until the output is frozen.

The [country guide](references/country-background.md) covers 13 countries. The command `python3 scripts/country_context.py NGA` retrieves the Nigeria section. Follow the cited sources for local evidence and consult the [climate and cocoa guide](references/climate-cocoa-background.md) as needed. Regional evidence helps select a location or season for closer inspection. Use the actual local RGB to decide a unit's class.
