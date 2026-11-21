# Fresh image selection

Read the location, CRS and affine transform from the original 128 × 128 AEF TIFF. Acquire fresh Google Earth and Esri candidates in a new sample directory. Approved appearance references are available during annotation.

| Label period | Actual capture dates |
| --- | --- |
| 2019 | 2017-01-01 through 2020-12-31, inclusive |
| 2025 | 2025–2026 first, with 2024 included if fewer than five distinct recent acquisitions qualify. Earliest actual capture is 2024-01-01. |

Record the label period, AEF file year, mosaic release and actual capture dates separately. For mosaics, verify that both ends of the actual capture range lie inside the window. Continue to the next scene if the actual dates are unverified or out of range.

Allow missing imagery on up to 5% of the original AEF grid. Fill those pixels black in the RGB and mask them in the delivered label. Look for another scene when more than 5% is missing.

1. Verify the dates and coverage from both providers, then group duplicate publications by provider and capture dates. Google historical catalogs can provide multiple zoom levels for the same date. Prefer the higher resolution and fall back to the lower one if available. Use verified Esri capture dates for each part of a mosaic.
2. Download the qualifying date set and register the RGB at 1024 × 1024 within the AEF footprint. Deduplicate registered pixels and support. If failures or duplicates leave fewer than five recent scenes, include the earlier 2025-period candidates. Record each attempted candidate and its outcome.
3. Rank scenes using cloud/haze indicator flags, lower-quartile local edge detail, contrast and capture date. The flags inform ranking conservatively. Assign seasons from the tile-center monthly rainfall normals. A capture month with up to 50 mm is dry, 80 mm and above is wet, and intermediate or mixed-date coverage is transitional. The program retains the seasonal metadata with the selection.
4. Show the selected primary scene to Sol first and provide auxiliary evidence on request. Use a locally cached dry candidate when the interpretation calls for dry imagery. Complete the image decision before requesting masks. Fall back to confirmed portions of a usable non-dry plan if dry candidates are exhausted. Same-location corroboration at another verified date is needed to accept Cashew on a non-dry scene. Record “No usable imagery” if the candidates are exhausted.

## Implementation

- [Local preflight](../scripts/imagery_preflight.py) writes the ranked `SEARCH.json` inventory and `SELECTION.json` for each available scene. [Annotation prompt](code-selected-annotation.md)
- [Image policy](../scripts/imagery_policy.py) enforces dates and coverage. [Quality and climate](../scripts/imagery_quality.py) use [clipped monthly WorldClim normals](../assets/west-africa-precipitation-normals.tif), a 1970–2000 baseline at five arc minutes. [Climate asset provenance](../assets/west-africa-precipitation-normals.json) · [WorldClim 2.1](https://www.worldclim.org/data/worldclim21.html)
- [Google Earth](../scripts/imagery/google.py) runs the [browser capture](../scripts/imagery/google_capture.py), reads the visible historical timeline and capture attribution and uses [map registration](../scripts/imagery/google_ui_registration.py). [Date parser](../scripts/imagery/google_date_evidence.py) handles dates, ranges and unknown-date markers. If the viewport includes unknown dates, a tighter registered view can establish evidence when it still covers the AEF footprint. Date evidence is recorded in `ACQUISITION_EVIDENCE.json`.
- [GEHistoricalImagery adapter](../scripts/imagery/gehi.py) checks dates covering the sample, acquires one exact date and zoom and verifies tile counts before registration. Coverage is measured from the acquired raster mask. Set `CASHEW_GEHI_BIN` to use the executable. The browser is available when a catalog request fails. [GEHistoricalImagery project](https://github.com/Mbucari/GEHistoricalImagery)
- [Esri](../scripts/imagery/esri.py) queries current World Imagery, Wayback archive versions, imagery levels and actual capture metadata.

The Google browser uses the host imagery proxy at `127.0.0.1:7850` with the deployed Playwright, Chromium and OCR. Esri uses the configured provider proxies. Acquisition, date and registration failures are logged under the sample directory. [Google historical imagery](https://developers.google.com/maps/documentation/earth/historical-imagery) · [Google image dates](https://support.google.com/earth/answer/6327779) · [Esri Wayback](https://github.com/Esri/wayback-core)
