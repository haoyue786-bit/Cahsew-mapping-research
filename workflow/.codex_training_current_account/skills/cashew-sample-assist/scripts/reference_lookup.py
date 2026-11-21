#!/usr/bin/env python3
"""Retrieve reviewed appearance references around a fixed target footprint."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from _runtime.common import SKILL_ROOT, read
from annotation_geometry import record_image
from imagery_quality import describe_season, rainfall_normals


LIBRARY = Path(os.environ.get('CASHEW_REFERENCE_ROOT',
                              SKILL_ROOT.parents[2] / 'outputs/cashew_reference_library_v2_20261117'))
ASSETS = Path(os.environ.get('CASHEW_REFERENCE_ASSETS', SKILL_ROOT / 'assets'))
REFERENCE_ASSETS = ASSETS / 'coordinate-reference-library'
BALANCED_ASSETS = ASSETS / 'coordinate-reference-library-balanced'
COMPACT_ASSETS = ASSETS / 'coordinate-reference-library-compact'
APPEARANCE = ('sparse', 'variable', 'dense', 'edge', 'rows', 'riparian', 'open_field', 'built')


def intersects(first, second):
    return first[0] < second[2] and first[2] > second[0] and first[1] < second[3] and first[3] > second[1]


def evaluation_origin(item):
    provenance = item.get('provenance', {})
    label = provenance.get('source_original_label') or provenance.get('original_label') or {}
    scope = str(label.get('label_scope') or '').lower()
    return (provenance.get('source_kind') in ('canonical_current_validation_points', 'validation_location')
            or label.get('point_id') is not None or ('point' in scope and 'polygon' not in scope))


def reference_rank(item, context, center_lon, center_lat, lon_scale):
    lon, lat = item['coordinates']
    distance = ((lon - center_lon) * lon_scale)**2 + (lat - center_lat)**2
    preferred_provider = context.get('frame', {}).get('provider')
    provider = item.get('provider', '')
    if provider.startswith('Google'):
        provider = 'Google'
    same_provider = bool(preferred_provider and preferred_provider == provider)
    same_season = False
    expected = context.get('source_season')
    dates = item.get('provenance', {}).get('dates') or []
    if expected in ('dry', 'wet') and dates and all(
            isinstance(day, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', day) for day in dates):
        try:
            season, _ = describe_season(dates, rainfall_normals(lon, lat))
            same_season = season == expected
        except ValueError:
            pass
    # Season and source order similarly located examples while keeping nearby references first.
    appearance = context.get('reference_appearance')
    appearance_mismatch = appearance is not None and appearance not in item.get('appearance_tags', [])
    return (appearance_mismatch, int(math.sqrt(distance)), not same_season, not same_provider, distance)


def lookup_reference(arguments, context, *, confirmed_overlap_id=None):
    # Check the live review overlay so a later rejection takes effect immediately.
    if str(LIBRARY) not in sys.path:
        sys.path.insert(0, str(LIBRARY))

    west, south, east, north = context['bounds']
    center_lon, center_lat = (west + east) / 2, (south + north) / 2
    lon_scale = math.cos(math.radians(center_lat))
    live = None
    requested = arguments['reference_id']
    compact = COMPACT_ASSETS/'index.json'
    if compact.is_file():
        selection = read(compact)
        candidates = [(item,COMPACT_ASSETS) for item in selection['references']]
        admitted_ids = set(selection['eligible_reference_ids'])
    else:
        root = BALANCED_ASSETS if (BALANCED_ASSETS/'index.json').is_file() else REFERENCE_ASSETS
        if not (root/'index.json').is_file():
            return {'success': True, 'references': [], 'images': [],
                    'note': 'No authorized appearance-reference bundle is installed'}
        candidates = [(item,root) for item in read(root/'index.json')['references']]
        admitted_ids = None

    def matches(item):
        return requested is None or requested in (item['reference_id'],item.get('provenance',{}).get('revises_reference_id'))

    if requested is not None and not any(matches(item) for item,_ in candidates):
        candidates = [(item,root) for root in (BALANCED_ASSETS,REFERENCE_ASSETS)
                      if (root/'index.json').is_file()
                      for item in read(root/'index.json')['references'] if matches(item)]
    match_context = dict(context, reference_appearance=arguments.get('appearance'))
    candidates.sort(key=lambda row: reference_rank(row[0], match_context, center_lon, center_lat, lon_scale))
    references, images = [], []
    for item, assets in candidates:
        identity = item['reference_id']
        if not matches(item) or evaluation_origin(item) or (admitted_ids is not None and identity not in admitted_ids):
            continue
        if arguments['class_id'] is not None and item['class_id'] != arguments['class_id']:
            continue
        provider = item.get('provider', 'Esri')
        approval = item['approval_status']
        provenance = item.get('provenance', {})
        filename = hashlib.sha256(identity.encode()).hexdigest() + '.json'
        review_path = LIBRARY / 'human_reviews' / filename
        human_review = read(review_path) if review_path.is_file() else {}
        action = human_review.get('action')
        if action in ('reject','flag','change_class') or action=='approve' and human_review.get('class_id')!=item['class_id']:
            continue
        footprint = item.get('bounds_wgs84')
        if action=='approve' or (approval == 'human_approved'
                                and provenance.get('source_kind') == 'operator_confirmed_annotation_correction'):
            approval='human_approved'
        elif provenance.get('curation_run') or provider=='Google Earth':
            approval='model_reviewed_reference'
        elif provenance.get('source_kind')!='source_training_grid':
            if live is None:
                from merge_review_results import curated_references
                live = {r['id']:r for r in curated_references(LIBRARY,class_id=arguments['class_id'])
                        if r['approval_status']=='human_approved'}
            record = live.get(identity)
            if not record or item['class_id'] != record['effective_class_id']:
                continue
            footprint = footprint or next((image['bounds_wgs84'] for image in record['images']
                              if image.get('role') == 'rgb' and image.get('bounds_wgs84')), None)
        lon, lat = item['coordinates']
        overlaps = (west <= lon <= east and south <= lat <= north
                    or footprint is not None and intersects(footprint, context['bounds']))
        excluded = any((b[0] <= lon <= b[2] and b[1] <= lat <= b[3])
                       or (footprint is not None and intersects(footprint, b))
                       for b in context.get('excluded_reference_bounds', ()))
        if excluded:
            continue
        use_confirmed_overlap = identity == confirmed_overlap_id and approval == 'human_approved'
        if overlaps and not use_confirmed_overlap:
            continue
        scope = item['scope']
        references.append(dict(reference_id=identity, class_id=item['class_id'], coordinates=item['coordinates'],
                               scope=scope['kind'], label_scope=scope['description'],
                               appearance_note=item['appearance_note'], approval_status=approval,
                               provider=provider, cashew_density=item.get('cashew_density'),
                               appearance_focus=item.get('appearance_focus'),
                               source_dates=item.get('provenance', {}).get('dates'),
                               date_comparison_note=item.get('model_layer', {}).get('date_comparison_note')))
        for image in item['images']:
            path = assets / image['path']
            images.append(record_image(path, f"{identity}_{image['role']}", 'reviewed_appearance_reference',
                                       provider=provider,
                                       reference_id=identity, class_id=item['class_id'],
                                       label_scope=scope['description'], display_note=image['caption'],
                                       sha256=image['sha256']))
        if len(references) >= arguments['limit']:
            break
    return {'success': True, 'references': references, 'images': images}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bounds', nargs=4, type=float, metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
                        required=True, help='The current target bounds in WGS84')
    parser.add_argument('--class-id', type=int, choices=range(4))
    parser.add_argument('--reference-id')
    parser.add_argument('--limit', type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    output = lookup_reference(dict(class_id=args.class_id, reference_id=args.reference_id, limit=args.limit),
                              dict(bounds=args.bounds))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
