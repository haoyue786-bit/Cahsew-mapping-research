"""Compose confirmed masks, classified background and uncertainty on the source image."""

from time import monotonic

import numpy as np
from PIL import Image

from _runtime.common import write
from annotation_geometry import GRID, SIZE, class_counts, polygon_mask, sam_unit_folder, woody_background_mask
from label_resampling import UNLABELLED
from xanylabeling_sam2 import segment


def read_mask(folder, unit):
    """Check the only returned mask against the prompt before writing its class."""
    with Image.open(folder / 'mask0.png') as image:
        mask = np.asarray(image) != 0
    if mask.shape != (SIZE, SIZE):
        raise ValueError('Expected a full-scene SAM2 mask')
    if unit['box'] is not None:
        x0, y0, x1, y1 = unit['box']
        clipped = np.zeros_like(mask)
        clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        mask = clipped
    if (not mask.any() or not all(mask[y, x] for x, y in unit['positive_points'])
            or any(mask[y, x] for x, y in unit['negative_points'])):
        return None
    return mask


def materialize(plan, context, evidence):
    start = monotonic()
    units = []
    for i, unit in enumerate(plan['regions'], 1):
        box = unit['box']
        if box is not None:
            xa, ya, xb, yb = box
            x0, x1 = sorted((xa, xb))
            y0, y1 = sorted((ya, yb))
            # Ensure a boundary box contains its interior prompts even after integer rounding.
            box = [x0, y0, x1, y1]
            if unit['positive_points']:
                xs, ys = zip(*unit['positive_points'])
                box = [min(x0, min(xs)), min(y0, min(ys)), max(x1, max(xs) + 1), max(y1, max(ys) + 1)]
        reason = unit.get('cashew_evidence') or 'Sol confirmed this unit on the source RGB'
        units.append(dict(unit, box=box, unit_id=f'land_{i:02}', reason=reason,
                          evidence_image_ids=evidence))
    records = segment({'units': units}, context, render_review=False)['units'] if units else []
    sam_seconds = monotonic() - start
    choices, masks = [], []
    for unit in records:
        folder = sam_unit_folder(context, unit['unit_id']) / str(unit['revision'])
        mask = read_mask(folder, unit)
        if mask is not None:
            masks.append((unit['class_id'], mask))
        choices.append(dict(unit_id=unit['unit_id'], class_id=unit['class_id'], revision=unit['revision'],
                            candidate_index=0 if mask is not None else None,
                            status='accepted' if mask is not None else 'unresolved', box=unit['box']))
    return save_composition(plan, context, evidence, choices, masks), sam_seconds


def save_composition(plan, context, evidence, choices, masks, *, matching=None):
    """Write masks and any confirmed two-class remainder before applying uncertainty."""
    review = context['imagery_uncertainty']
    fill = plan['whole_class']
    if fill is not None and (set(review.get('visible_classes', [])) - {fill}
                             or fill != 2 and review.get('reported_presence') == 'present'):
        fill = None
    labels = np.full((SIZE, SIZE), 10 if fill is None else fill, dtype=np.uint8)
    assigned_sam = np.full_like(labels, 10)
    assigned = np.full_like(labels, 10)
    conflict = np.zeros_like(labels, dtype=bool)
    for code, mask in masks:
        conflict |= mask & (assigned != 10) & (assigned != code)
        assigned[mask] = code
        assigned_sam[mask] = code
    direct_areas = plan.get('direct_regions', [])
    for region in direct_areas:
        code = region['class_id']
        mask = polygon_mask(region['polygon'])
        conflict |= mask & (assigned != 10) & (assigned != code)
        assigned[mask] = code
    if masks or direct_areas:
        labels[:] = assigned
        labels[conflict] = 10
    background = plan.get('background')
    pair = set(plan.get('visible_classes', []))
    review_classes = set(review.get('visible_classes', []))
    if review.get('reported_presence') == 'present':
        review_classes.add(2)
    if background is not None and masks and not review_classes - pair:
        residual = (assigned == 10) & ~conflict & ~woody_background_mask(review.get('woody_background', []), [])
        unresolved_boxes = plan.get('withheld_foreground_boxes', []) + [
            unit['box'] for unit in choices if unit.get('status') == 'unresolved']
        for xa, ya, xb, yb in unresolved_boxes:
            x0, x1 = sorted((xa, xb))
            y0, y1 = sorted((ya, yb))
            residual[y0:y1, x0:x1] = False
        labels[residual] = background['class_id']
    valid = context['valid_mask_native128'].repeat(SIZE // GRID, 0).repeat(SIZE // GRID, 1)
    if fill is not None:
        labels[woody_background_mask(review.get('woody_background', []), [])] = 10
    for area in review['polygons'] + plan.get('uncertain_polygons', []):
        labels[polygon_mask(area)] = 10
    labels[~valid] = UNLABELLED
    checks = []
    for item in review['candidates']:
        x, y = map(int, item['point_xy'])
        checks.append(dict(item, source_class=int(labels[y, x]), supported_by_sam=bool(
            labels[y, x] == 2 and assigned_sam[y, x] == 2)))
    for item in review['search_windows']:
        x0, y0, x1, y1 = map(int, item['box_xyxy'])
        checks.append(dict(item, source_class_counts=class_counts(labels[y0:y1, x0:x1])))
    counts = class_counts(labels)
    if background is not None and str(background['class_id']) in counts:
        mode = 'two_classes_sam2_complement'
    elif any(int(code) < 4 for code in counts) and fill is not None:
        mode = 'whole_sample'
    elif direct_areas:
        mode = 'sam_and_direct' if choices else 'direct_polygons'
    else:
        mode = 'sam' if choices else 'unknown'
    folder = context['case_dir'] / 'compositions' / '1'
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels).save(folder / 'labels1024.png')
    annotation_id = f"{context['case']}_annotation1"
    composition = dict(
        annotation_id=annotation_id, annotation_method=mode, source_class_counts=counts,
        surviving_sam_source_counts=class_counts(labels[(assigned_sam < 4) & (labels < 4)]),
        cashew_review_checks=checks, required_image_ids=evidence, accepted_units=choices,
        direct_areas=([{'class_id': fill, 'polygon': None}] if fill is not None else direct_areas),
        whole_class_requested=plan['whole_class'],
    )
    if background is not None:
        composition['background'] = background
    if matching is not None:
        composition['region_matching'] = matching
    write(folder / 'COMPOSITION.json', composition)
    write(folder.parent / 'LATEST.json', {'folder': str(folder)})
    return annotation_id
