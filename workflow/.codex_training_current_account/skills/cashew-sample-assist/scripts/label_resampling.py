"""Categorical area voting from a registered source mask to its AEF footprint."""
import numpy as np

UNLABELLED = 255
CODES = (0, 1, 2, 3, 10, UNLABELLED)


def _axis_areas(values, size, axis):
    """Integrate piecewise constant values between equally spaced target edges."""
    values = np.moveaxis(values, axis, 0)
    n = values.shape[0]
    edges, remainders = np.divmod(np.arange(size + 1) * n, size)
    prefix = np.empty((n + 1,) + values.shape[1:], dtype=np.int64)
    prefix[0] = 0
    np.cumsum(values, axis=0, dtype=np.int64, out=prefix[1:])
    weights = remainders.reshape((-1,) + (1,) * (values.ndim - 1))
    integrals = prefix[edges] * size + values[np.minimum(edges, n - 1)] * weights
    return np.moveaxis(np.diff(integrals, axis=0), 0, axis)


def area_mode(source, shape):
    """Return the class with the most area. Resolve ties spatially, including class 10."""
    h, w = source.shape
    rows, cols = shape
    areas = []
    for code in CODES:
        pixels = source == code
        if h % rows == 0 and w % cols == 0:
            area = pixels.reshape(rows, h // rows, cols, w // cols).sum(axis=(1, 3))
        else:
            area = _axis_areas(_axis_areas(pixels, cols, 1), rows, 0)
        areas.append(area)
    areas = np.stack(areas)
    winners = areas == areas.max(axis=0)
    classes = np.asarray(CODES, dtype=np.uint8)
    result = classes[areas.argmax(axis=0)]
    ys, xs = np.where(winners.sum(axis=0) > 1)
    if len(ys):
        center_y, center_x = (ys + .5) * h / rows, (xs + .5) * w / cols
        middle = source[center_y.astype(int), center_x.astype(int)]
        center_wins = winners[classes.searchsorted(middle), ys, xs]
        result[ys[center_wins], xs[center_wins]] = middle[center_wins]
        for k in np.flatnonzero(~center_wins):
            y, x = ys[k], xs[k]
            y0, x0 = y * h // rows, x * w // cols
            y1, x1 = ((y + 1) * h + rows - 1) // rows, ((x + 1) * w + cols - 1) // cols
            block = source[y0:y1, x0:x1]
            # Nearest contributing pixel of a tied class. Prefer bottom/right on equal distance.
            iy, ix = np.where(np.isin(block, classes[winners[:, y, x]]))
            d2 = (iy + y0 + .5 - center_y[k]) ** 2 + (ix + x0 + .5 - center_x[k]) ** 2
            index = np.flatnonzero(d2 == d2.min())[-1]
            result[y, x] = block[iy[index], ix[index]]
    return result
