"""Display SAM2 proposals and commit only masks retained in the Sol review."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from _runtime.common import write
from annotation_geometry import COLORS, GRID, SIZE, record_image
from foreground_masks import save_composition
from similar_regions import match_detected_regions, read_mask


def mask_panel(context, records, name, classes=None):
    with Image.open(context['scene']) as rgb:
        scene = rgb.convert('RGB')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
    positions = []
    for item in records:
        identity = item['id']
        color = COLORS[classes[identity]] if classes is not None else (255, 216, 88)
        shape = Image.fromarray(read_mask(item['mask_path']).astype(np.uint8) * 255)
        interior = shape.filter(ImageFilter.MinFilter(5))
        contour = np.asarray(shape) > np.asarray(interior)
        tint = Image.fromarray(np.where(contour, 255, np.asarray(shape) // 6).astype(np.uint8))
        scene.paste(color, (0, 0, SIZE, SIZE), tint)
        x0, y0, x1, y1 = item['box']
        positions.append((identity, (min(SIZE - 52, max(0, (x0 + x1) // 2)), (y0 + y1) // 2)))
    draw = ImageDraw.Draw(scene)
    for identity, position in positions:
        draw.text(position, identity, font=font, fill='white', stroke_width=2, stroke_fill='black')
    path = context['case_dir'] / (name + '.png')
    scene.save(path)
    return record_image(path, context['case'] + '_' + name, name)


def suggest_groups(context, plan, *, match=True):
    masks = {item['id']: Path(item['mask_path']) for item in context['box_proposals']['masks']}
    representatives = {unit['mask_id']: unit['class_id'] for unit in plan['regions']}
    valid = context['valid_mask_native128'].repeat(SIZE // GRID, 0).repeat(SIZE // GRID, 1)
    anchors = {code: read_mask(masks[identity]) for identity, code in representatives.items()}
    candidates = {identity: path for identity, path in masks.items() if identity not in representatives}
    permit_woody = plan['allow_woody_similarity'] and context['source_confirmed_local_dry_season']
    if match and candidates and len(anchors) >= 2:
        report = match_detected_regions(context, anchors, candidates, valid, permit_woody=permit_woody)
    else:
        report = dict(status='nothing_to_match', matches=[])
    additions = {item['id']: item['class_id'] for item in report['matches'] if item['class_id'] is not None}
    classes = dict(representatives, **additions)
    groups = {f'G{code}': [identity for identity, value in classes.items() if value == code]
              for code in sorted(set(classes.values()))}
    return dict(groups=groups, classes=classes, representatives=list(representatives), additions=list(additions),
                status=report['status'], seconds=report.get('seconds', 0))


def resolve_keep(result, groups, classes):
    """Expand group and mask IDs while preserving the proposed class of each mask."""
    selected = set()
    for identity in result['keep']:
        if identity in groups:
            selected.update(groups[identity])
        elif identity in classes:
            selected.add(identity)
        else:
            raise ValueError('Unknown group or mask to retain: ' + identity)
    omitted = set(result['omit'])
    if not omitted.issubset(selected):
        raise ValueError('Omitted masks must be part of the retained selection')
    return [identity for identity in classes if identity in selected - omitted]


def commit_review(context, plan, evidence, suggestions, kept):
    masks = {item['id']: item['mask_path'] for item in context['box_proposals']['masks']}
    classes = suggestions['classes']
    choices = [dict(unit_id=identity, class_id=classes[identity], candidate_index=0, status='accepted')
               for identity in kept]
    selected = [(classes[identity], read_mask(masks[identity])) for identity in kept]
    ledger = dict(suggestions, kept=kept, keep_review_needed=bool(suggestions['additions']),
                  kept_suggested_count=len(set(kept) & set(suggestions['additions'])))
    path = context['case_dir'] / 'REGION_REVIEW.json'
    write(path, ledger)
    return save_composition(plan, context, evidence, choices, selected, matching=dict(ledger, review_path=str(path)))
