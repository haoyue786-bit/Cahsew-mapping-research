# AEF alignment and deliverables

Read the CRS, affine transform, footprint and 128 × 128 pixel grid from the original AEF sample. The label period is a separate input. Follow the [date and selection policy](imagery-selection.md) for each new annotation.

Google Earth screenshots use visible coordinate control points for registration, with independent check points to measure residual error. Retain the control points, residuals, provider and date evidence. Esri supplies georeferenced imagery and capture metadata. Record display spacing and source resolution separately. The registered RGB follows the original AEF footprint. Wider imagery provides geographic context for interpretation.

Save the exact RGB as `image_rgb.png`. Keep a color overlay, source labels (`labels1024.png` and `labels_source.tif`), and labels on the AEF grid (`labels_native128.png` and `labels_native128.tif`). Check that PNG and TIFF carry the same AEF label values. The GeoTIFF preserves the original AEF CRS, affine transform, shape and pixel centers. The receipt records output paths, provider and capture dates.

Use an area vote for each 10 m AEF cell. Source value 10 denotes uncertain cover, and 255 denotes missing imagery. The 10 m TIFF uses an internal validity mask to distinguish missing data from semantic uncertainty. Missing AEF cells have mask 0 and retain numeric value 10 for legacy readers. Semantic uncertainty has value 10 with a valid mask.
