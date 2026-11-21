"""Extract a detail view on the requested footprint from the original georeferenced imagery."""

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds


def registered_detail(path, bounds, crs, *, max_size):
    with rasterio.open(path) as source:
        if source.crs is None or source.count < 3:
            raise ValueError('Native detail requires a georeferenced RGB source')
        left, bottom, right, top = transform_bounds(crs, source.crs, *bounds)
        cols = (right-left) / abs(source.res[0])
        rows = (top-bottom) / abs(source.res[1])
        ratio = min(1, max_size / max(cols, rows))
        width, height = max(1, round(cols*ratio)), max(1, round(rows*ratio))
        transform = from_bounds(*bounds,width,height)
        pixels = np.zeros((3,height,width), dtype=np.uint8)
        reproject(rasterio.band(source,[1,2,3]), pixels, dst_crs=crs,
                  dst_transform=transform, resampling=Resampling.bilinear)
    return Image.fromarray(np.moveaxis(pixels,0,-1))
