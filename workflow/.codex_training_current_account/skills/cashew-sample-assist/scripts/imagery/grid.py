"""Read only geometry from the original AEF sample."""

from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


def from_aef(path, case_id, country, period):
    with rasterio.open(path) as source:
        if source.shape != (128, 128) or not source.crs:
            raise ValueError("Expected an AEF raster with shape 128×128 and a CRS")
        bounds = transform_bounds(source.crs, "EPSG:4326", *source.bounds)
        west, south, east, north = bounds
        return dict(candidate_id=case_id, country=country, label_period=period,
                    aef_path=str(Path(path).resolve()), bounds_wgs84=list(bounds),
                    center_wgs84=[(west + east) / 2, (south + north) / 2],
                    native_crs=source.crs.to_string(), native_transform=list(source.transform)[:6])
