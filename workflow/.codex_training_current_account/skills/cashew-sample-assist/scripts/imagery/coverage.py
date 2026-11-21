"""Use the registered AEF support mask to hide missing imagery and label it 10."""

from pathlib import Path

import numpy as np
from PIL import Image

from _runtime.common import digest


def read_support(frame):
    path = Path(frame["native128_valid_mask_path"])
    with Image.open(path) as image:
        valid = np.asarray(image.convert("L")) > 0
    if (valid.shape != (128, 128) or digest(path) != frame["native128_valid_mask_sha256"]
            or float(valid.mean()) != frame["native128_valid_fraction"]):
        raise ValueError("Registered coverage did not match the saved native 128×128 mask")
    return valid


def black_mask_frame(frame):
    if frame["native128_valid_fraction"] == 1:
        return frame
    valid = read_support(frame)
    original = Path(frame["image_path"])
    if digest(original) != frame["sha256"]:
        raise ValueError("Registered source RGB changed before masking")
    with Image.open(original) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    if rgb.shape != (1024, 1024, 3):
        raise ValueError("Expected 1024×1024 RGB aligned to the native AEF sample")
    rgb[~valid.repeat(8, axis=0).repeat(8, axis=1)] = 0
    path = Path(frame["folder"]) / "rgb_missing_black.png"
    Image.fromarray(rgb).save(path)
    return dict(frame, image_path=str(path), sha256=digest(path), unmasked_image_path=str(original),
                native128_missing_pixels=int((~valid).sum()))
