"""Geometry and display conventions shared by direct labeling and SAM2."""

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from label_resampling import UNLABELLED


SIZE = 1024
GRID = 128
MAX_SEGMENT_UNITS = 20
COLORS = {0: (231, 185, 56), 1: (48, 145, 66), 2: (239, 51, 90), 3: (45, 147, 239), 10: (160, 160, 160)}


def record_image(path, image_id, role, **metadata):
    return dict(path=str(path), image_id=image_id, role=role, **metadata)


def cover_image(labels):
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for class_id, color in COLORS.items():
        rgb[labels == class_id] = color
    return Image.fromarray(rgb)


def polygon_mask(vertices):
    polygon = Polygon(vertices)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError('Use a valid polygon in complete-scene coordinates')
    mask = Image.new('1', (SIZE, SIZE))
    ImageDraw.Draw(mask).polygon([tuple(point) for point in vertices], fill=1)
    return np.asarray(mask)


def same_image_review(reviews, grid, frame):
    """Operator annotations apply only to the exact reviewed sample and RGB."""
    return next((review for review in reviews if
                 (review['case_id'], review['label_period'], review['frame_sha256']) ==
                 (grid['candidate_id'], grid['label_period'], frame['sha256'])), {})


def imagery_uncertainty(decision, operator_review=None):
    """Keep the selected RGB's observations separate from areas locked to class 10."""
    polygons = decision.get('uncertain_polygons')
    presence = decision.get('cashew_presence')  # Older selections also recorded land-cover observations.
    legacy_note = (decision.get('explanation', '') if polygons is None and presence == 'uncertain' else '')
    candidates = [dict(item, id=f'review_{i}', source='image_review')
                  for i, item in enumerate(decision.get('cashew_candidates', []), 1)]
    operator_review = operator_review or {}
    candidates.extend(dict(item, source='human_point') for item in operator_review.get('points', []))
    woody = decision.get('woody_background', [])
    return dict(polygons=polygons or [], possible_cashew=presence in ('present', 'uncertain') or bool(candidates) or woody != [],
                reported_presence=presence, explanation=decision.get('explanation', ''),
                visible_classes=decision.get('visible_classes', []), candidates=candidates,
                woody_background=woody,
                search_windows=operator_review.get('search_windows', []),
                legacy_note=legacy_note, legacy_needs_outline=bool(legacy_note))


def apply_uncertainty(context, labels, polygons):
    review = context.get('imagery_uncertainty', {})
    legacy = context['activity'].get('legacy_review_polygons')
    new_outline = review.get('legacy_needs_outline') and legacy is None
    if new_outline:
        if not polygons:
            raise ValueError('Outline the unresolved units named in the selected image review first')
    all_polygons = review.get('polygons', []) + (legacy or []) + polygons
    if all_polygons:
        mask = np.logical_or.reduce([polygon_mask(area) for area in all_polygons])
        labels[mask] = 10
    return new_outline


def class_counts(labels):
    return {str(int(c)): int(n) for c, n in zip(*np.unique(labels, return_counts=True))}


def woody_background_mask(review_scope, annotation_scope):
    """Empty lists have no effect; null means the visually identified residual is woody throughout."""
    if review_scope is None or annotation_scope is None:
        return np.ones((SIZE, SIZE), dtype=bool)
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    for polygon in review_scope + annotation_scope:
        mask |= polygon_mask(polygon)
    return mask


def confirmed_other_wood(mask, woody, previous, points):
    """Limit an other-tree label inside suspect woodland to explicitly supported units."""
    overlap = mask & woody
    if not overlap.any():
        return mask
    height, width = mask.shape
    inside = [(int(y), int(x)) for x, y in points if 0 <= x < width and 0 <= y < height
              and overlap[int(y), int(x)]]
    if not inside:
        return mask & ~woody
    cashew_overlap = overlap & (previous == 2)
    if cashew_overlap.any() and not any(cashew_overlap[y, x] for y, x in inside):
        return mask & ~cashew_overlap
    return mask


def review_cashew_coverage(context, labels, sam_labels, mixed_scene, complement, exceptions, woody=None):
    review = context['imagery_uncertainty']
    candidates = review.get('candidates', [])
    windows = review.get('search_windows', [])
    explanations = {row['check_id']: row for row in exceptions}
    expected_ids = {row['id'] for row in candidates + windows}
    if review.get('reported_presence') == 'present' and not candidates:
        expected_ids.add('review_scene')
    if len(explanations) != len(exceptions) or set(explanations) - expected_ids:
        raise ValueError('Use each supplied review ID at most once for an exception')
    for row in exceptions:
        require_evidence(row, context['registered_frames'], context['case'] + '_scene')

    records, issues = [], []
    for item in candidates:
        identity = item['id']
        x, y = item['point_xy']
        sx, sy = int(x), int(y)
        code = int(labels[sy, sx])
        records.append(dict(item, source_class=code, supported_by_sam=bool(sam_labels[sy, sx] == 2)))
        location = f"{identity} at RGB ({x},{y})"
        if code == 2:
            if identity in explanations:
                issues.append(f'{location}: the exception contradicts the exported class 2; '
                              'correct the class or remove the exception after reviewing the cell')
            if mixed_scene and sam_labels[sy, sx] != 2 and complement != 2:
                issues.append(f'{location}: inspect and use a Cashew SAM mask that covers this interior')
        elif code == UNLABELLED:
            valid = context.get('valid_mask_native128')
            if valid is None or valid[sy * GRID // labels.shape[0], sx * GRID // labels.shape[1]]:
                issues.append(f'{location}: assign the visible stand a class, or outline it as uncertain '
                              'if its identity is unresolved on the RGB')
        elif code != 10 and item['source'] == 'human_point':
            issues.append(f'{location}: human-confirmed Cashew point now lies in class {code}; inspect the '
                          'unit with SAM, or mark the source area 10 if its identity is unclear and explain')
        elif (code == 10 and identity not in explanations and woody is not None and woody[sy, sx]
              and item['source'] == 'image_review' and item.get('assessment') == 'plausible'):
            records[-1]['uncertain_reason'] = 'unresolved_woody_background'
        elif identity not in explanations:
            issues.append(f'{location}: final class {code}; segment a supported interior, or give local '
                          'evidence in review_exceptions for an alternative class or uncertainty')
        elif (code == 10) != (explanations[identity]['outcome'] == 'uncertain'):
            issues.append(f'{location}: the exception outcome and source class {code} disagree')

    for window in windows:
        x0, y0, x1, y1 = window['box_xyxy']
        values = labels[int(np.ceil(y0-.5)):int(np.ceil(y1-.5)), int(np.ceil(x0-.5)):int(np.ceil(x1-.5))]
        counts = class_counts(values)
        records.append(dict(window, source_class_counts=counts))
        if '2' not in counts and window['id'] not in explanations:
            issues.append(f"{window['id']}: the reviewed search window has zero Cashew source pixels; inspect its "
                          'interiors and give a local explanation if the window remains unresolved or differs')
        elif ('2' not in counts and explanations[window['id']]['outcome'] == 'uncertain' and '10' not in counts):
            issues.append(f"{window['id']}: delineate the unresolved area as class 10 on the source RGB")
    if 'review_scene' in expected_ids and not (labels == 2).any() and 'review_scene' not in explanations:
        issues.append('review_scene: the image review saw Cashew; locate that stand or explain the local disagreement')
    if issues:
        raise ValueError(' | '.join(issues))
    return records


def require_evidence(item, registry, scene_id=None):
    ids = item['evidence_image_ids']
    if not item['reason'].strip() or not ids or any(identity not in registry for identity in ids):
        raise ValueError('Give the visible reason and cite delivered image IDs for the unit')
    if scene_id is not None and scene_id not in ids:
        raise ValueError('Cite the image ID of the complete query RGB for this decision')


def sam_unit_folder(context, unit_id):
    return context['case_dir'] / 'sam' / unit_id


def next_number(folder, prefix=''):
    values = [int(path.name[len(prefix):]) for path in folder.glob(prefix + '*')
              if path.name[len(prefix):].isdigit()]
    return max(values, default=0) + 1
