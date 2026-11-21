"""Pair period-specific AlphaEarth images with year-independent human masks."""
import re
from pathlib import Path

_IMAGE = re.compile(r'AlphaEarth_(2018|2019|2024|2025)_([A-Za-z0-9][A-Za-z0-9_-]*)_(\d+)_(\d+)\.tif')


def parse_image_name(image_name):
    """Return acquisition year and stable tile/pixel suffix; reject paths."""
    if not isinstance(image_name, str) or Path(image_name).name != image_name:
        raise ValueError('supplementary image name must be a plain filename')
    match = _IMAGE.fullmatch(image_name)
    if match is None:
        raise ValueError('unexpected supplementary image name: ' + image_name)
    return int(match.group(1)), '_'.join(match.group(i) for i in (2, 3, 4))


def resolve_label_path(label_root, image_name):
    """Use the shared geometry label; never substitute a different image."""
    _, suffix = parse_image_name(image_name)
    return Path(label_root) / ('label_processed_' + suffix + '.tif')
