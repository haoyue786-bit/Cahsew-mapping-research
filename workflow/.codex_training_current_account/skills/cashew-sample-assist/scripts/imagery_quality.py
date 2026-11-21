"""Measure registered RGB quality and look up local rainfall season."""

from datetime import date
from hashlib import sha256

import cv2
import numpy as np
from PIL import Image
import rasterio

from _runtime.common import SKILL_ROOT
from imagery.coverage import read_support

BLOCK = 64
DRY_MM = 50
WET_MM = 80
NORMALS = SKILL_ROOT / 'assets/west-africa-precipitation-normals.tif'

def blocks(image):
    height, width = image.shape
    return image.reshape(height // BLOCK, BLOCK, width // BLOCK, BLOCK).transpose(0, 2, 1, 3).reshape(-1, BLOCK**2)


def score_quality(rgb, valid):
    """Score visible RGB defects and patch-level detail at the registered display scale."""
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    brightness = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    chroma = brightness - minimum
    average = cv2.boxFilter(luma, -1, (15, 15))
    variance = cv2.boxFilter(luma**2, -1, (15, 15)) - average**2
    white = (luma >= 200) & (chroma <= 24) & (variance <= 12**2)
    interior = blocks(valid).mean(axis=1) >= .99
    if not interior.any() or not valid.any():
        return dict(state='no_support')

    patch_luma = blocks(luma)[interior]
    contrast = np.percentile(patch_luma, 90, axis=1) - np.percentile(patch_luma, 10, axis=1)
    laplacian = cv2.Laplacian(luma, cv2.CV_32F)
    detail = np.sqrt(np.mean(blocks(laplacian)[interior]**2, axis=1))
    dark_channel = cv2.erode(minimum, np.ones((15, 15), np.uint8))
    haze = (np.median(blocks(dark_channel)[interior], axis=1) >= 90) & (contrast <= 24)

    red, green, blue = np.moveaxis(rgb.astype(np.int16), 2, 0)
    green_share = blocks((green >= red + 3) & (green >= blue + 5))[interior].mean(axis=1)
    vegetation = green_share >= .35
    details = detail[vegetation] if vegetation.sum() >= 8 else detail
    contrasts = contrast[vegetation] if vegetation.sum() >= 8 else contrast
    cloud_fraction = float(white[valid].mean())
    haze_fraction = float(haze.mean())
    detail_p25 = float(np.percentile(details, 25))
    contrast_p25 = float(np.percentile(contrasts, 25))
    flags = []
    if cloud_fraction >= .10:
        flags.append('bright_smooth_fraction')
    if haze_fraction >= .35:
        flags.append('raised_dark_channel_and_low_contrast')
    if detail_p25 < 4 and contrast_p25 < 28:
        flags.append('weak_edges_and_low_contrast')
    return dict(state='scored', usable=not flags, flags=flags,
                bright_smooth_fraction=cloud_fraction, haze_proxy_fraction=haze_fraction,
                detail_p25=detail_p25, contrast_p25=contrast_p25,
                detail_median=float(np.median(details)), green_blocks=int(vegetation.sum()),
                valid_fraction=float(valid.mean()))


def describe_season(dates, normals):
    rain = [normals.get(date.fromisoformat(day).month) for day in dates]
    if not rain or any(value is None for value in rain):
        return 'unknown', rain
    if all(value <= DRY_MM for value in rain):
        return 'dry', rain
    if all(value >= WET_MM for value in rain):
        return 'wet', rain
    return 'transition_or_mixed', rain


def rainfall_normals(lon, lat):
    with rasterio.open(NORMALS) as dataset:
        values = next(dataset.sample([(lon, lat)], masked=True))
    return {month: None if np.ma.is_masked(value) else int(value) for month, value in enumerate(values, 1)}


def frame_quality(frame):
    with Image.open(frame['image_path']) as image:
        if image.size != (1024, 1024):
            raise ValueError('Quality scoring requires registered 1024 by 1024 RGB')
        rgb = np.asarray(image.convert('RGB'))
    native = read_support(frame)
    valid = native.repeat(rgb.shape[0] // native.shape[0], axis=0).repeat(rgb.shape[1] // native.shape[1], axis=1)
    return dict(score_quality(rgb, valid), rgb_pixel_sha256=sha256(rgb.tobytes()).hexdigest())


def quality_order(candidate):
    quality = candidate['quality']
    return (quality['state'] == 'scored', not quality.get('flags'), quality.get('detail_p25', 0),
            quality.get('contrast_p25', 0), max(candidate['frame']['acquisition_dates']))
