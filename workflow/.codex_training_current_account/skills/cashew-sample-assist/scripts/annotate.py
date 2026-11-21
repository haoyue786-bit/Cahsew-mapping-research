#!/usr/bin/env python3
"""Review a georeferenced sample and label it directly or with optional X-SAM2."""

import fcntl
import hashlib
import shutil
import subprocess
from pathlib import Path
from time import monotonic, time
from types import SimpleNamespace

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window, bounds as window_bounds

from _runtime import transport
from _runtime.common import EFFORT, MODEL, SKILL_ROOT, digest, read, write
from _runtime.vision_audit import image_delivery_report, verify_images
from annotation_geometry import (GRID, MAX_SEGMENT_UNITS, SIZE, apply_uncertainty, class_counts, confirmed_other_wood, cover_image,
                                 imagery_uncertainty, next_number, polygon_mask,
                                 record_image, require_evidence, review_cashew_coverage,
                                 sam_unit_folder, same_image_review, woody_background_mask)
from box_review import commit_review, mask_panel, resolve_keep, suggest_groups
from country_context import administrative_area, lookup_country_context
from foreground_masks import materialize
from imagery import esri, gehi
from imagery.coverage import read_support
from imagery.detail import registered_detail
from imagery_policy import date_window, distinct_capture_periods, frame_rejection, review_rejection
from label_resampling import UNLABELLED, area_mode
from reference_lookup import APPEARANCE, lookup_reference
from similar_regions import DINO_REPO, WEIGHTS, local_model_runtime
from xanylabeling_sam2 import segment, segment_boxes


def obj(**fields):
    return {'type': 'object', 'additionalProperties': False, 'required': list(fields), 'properties': fields}


def array(items, maximum, minimum=0):
    return {'type': 'array', 'minItems': minimum, 'maxItems': maximum, 'items': items}


def nullable(schema):
    return {'anyOf': [schema, {'type': 'null'}]}


STRING = {'type': 'string'}
REASON = {'type': 'string', 'minLength': 1}
CLASS = {'type': 'integer', 'enum': [0, 1, 2, 3]}
UNIT_ID = {'type': 'string', 'pattern': '^[A-Za-z0-9_-]{1,40}$'}
COORD = {'type': 'number', 'minimum': 0, 'maximum': SIZE}
POINT = array(COORD, 2, 2)
POLYGON = array(POINT, 64, 3)
BOX = array(COORD, 4, 4)
FRAMES = array(STRING, 16, 1)
MAX_AUXILIARY_CHECKS = 10
DIRECT_FIELDS = dict(unit_id=UNIT_ID, class_id=CLASS, polygon=nullable(POLYGON), reason=REASON, evidence_image_ids=FRAMES)
OTHER_WOOD_POINT = array({'type': 'integer', 'minimum': 0, 'maximum': SIZE - 1}, 2, 2)
DIRECT = {'anyOf': [obj(**DIRECT_FIELDS), obj(**dict(DIRECT_FIELDS, class_id={'type': 'integer', 'enum': [1]},
                                                     confirmed_noncashew_xy=OTHER_WOOD_POINT))]}
SAM_PROMPT = obj(unit_id=UNIT_ID, class_id=CLASS, positive_points=array(POINT, 12, 1),
                 negative_points=array(POINT, 12), box=nullable(BOX), reason=REASON, evidence_image_ids=FRAMES)
SAM_CHOICE = obj(unit_id=UNIT_ID, class_id=CLASS, revision={'type': 'integer', 'minimum': 1},
                 candidate_index={'type': 'integer', 'enum': [0, 1, 2]}, reason=REASON)
REVIEW_EXCEPTION = obj(check_id=STRING, outcome={'type': 'string', 'enum': ['uncertain', 'identified_other']},
                       reason=REASON, evidence_image_ids=FRAMES)
COMPOSITION_FIELDS = dict(direct_areas=array(DIRECT, 48), accepted_units=array(SAM_CHOICE, 48),
                          priority_order=array(UNIT_ID, 48), uncertain_polygons=array(POLYGON, 24),
                          woody_background=nullable(array(POLYGON, 24)),
                          review_exceptions={'type': 'array', 'items': REVIEW_EXCEPTION})
FINAL_SCHEMA = obj(annotation_id=nullable(STRING),
                   request_earlier_image={'type': 'boolean'}, needs_dry_season={'type': 'boolean'})
DRY_SCENE_SCHEMA = obj(annotation_id=STRING)
PIXEL = {'type': 'integer', 'minimum': 0, 'maximum': SIZE - 1}
EDGE = {'type': 'integer', 'minimum': 0, 'maximum': SIZE}
REGION = obj(class_id=CLASS, positive_points=array(array(PIXEL, 2, 2), 4, 1),
             negative_points=array(array(PIXEL, 2, 2), 4), box=nullable(array(EDGE, 4, 4)),
             cashew_evidence=nullable(STRING))
PLAN_FIELDS = dict(whole_class=nullable(CLASS), whole_cashew_evidence=nullable(STRING),
                   regions=array(REGION, MAX_SEGMENT_UNITS),
                   evidence_image_ids={'type': 'array', 'items': STRING})
DRY_PLAN_SCHEMA = obj(**PLAN_FIELDS)
PLAN_SCHEMA = obj(**PLAN_FIELDS, request_earlier_image={'type': 'boolean'}, needs_dry_season={'type': 'boolean'})
DIRECT_REGION = obj(class_id=CLASS, polygon=POLYGON, cashew_evidence=nullable(STRING))
BOX_REGION = obj(class_id=CLASS, positive_points=array(array(PIXEL, 2, 2), 0),
                 negative_points=array(array(PIXEL, 2, 2), 0),
                 box=array(EDGE, 4, 4), cashew_evidence=nullable(STRING))
PREFLIGHT_PLAN_SCHEMA = obj(**dict(PLAN_SCHEMA['properties'],
    regions=array({'anyOf': [REGION, BOX_REGION]}, MAX_SEGMENT_UNITS),
    direct_regions=array(DIRECT_REGION, MAX_SEGMENT_UNITS),
    visible_classes=array(CLASS, 4),
    background=nullable(obj(class_id=CLASS, visible_cue=REASON)),
    uncertain_polygons=array(POLYGON, 24)))
REPRESENTATIVE_CLASSES = dict(cropland=0, mix_tree=1, cashew=2, built=3)
MATCH_PLAN_FIELDS = {k: v for k, v in PLAN_FIELDS.items() if k != 'regions'}
MATCH_PLAN_FIELDS.update(representatives=obj(**{name: nullable(STRING) for name in REPRESENTATIVE_CLASSES}),
                         cashew_evidence=nullable(STRING),
                         allow_woody_similarity={'type': 'boolean'})
DRY_MATCH_PLAN_SCHEMA = obj(**MATCH_PLAN_FIELDS)
MATCH_PLAN_SCHEMA = obj(**MATCH_PLAN_FIELDS, request_earlier_image={'type': 'boolean'}, needs_dry_season={'type': 'boolean'})
KEEP_SCHEMA = obj(keep={'type': 'array', 'items': STRING}, omit={'type': 'array', 'items': STRING})
REFERENCE_QUERY = obj(class_id=nullable(CLASS), reference_id=nullable(STRING),
                      limit={'type': 'integer', 'minimum': 1, 'maximum': 4})
REFERENCE_QUERY['properties']['comparison_class_id'] = nullable(CLASS)
REFERENCE_QUERY['properties']['appearance'] = nullable({'type': 'string', 'enum': list(APPEARANCE)})
COMPARISON_TOOL = dict(name='view_comparison',
                       description='Inspect an available registered RGB of the same footprint from another verified date.',
                       inputSchema=obj())
TOOLS = [
    dict(name='view_image_scale', description='Inspect the raw scene, original source detail, or a wider '
         'geographic view from the selected source.',
         inputSchema=obj(mode={'type': 'string', 'enum': ['scene', 'detail', 'landscape']},
                         bbox=nullable(BOX), scale=nullable({'type': 'integer', 'enum': [3, 5]}))),
    dict(name='view_details', description='Inspect one to four native-detail crops. Supply boxes in the full 1024 RGB coordinate system.',
         inputSchema=obj(boxes=array(BOX, 4, 1))),
    dict(name='lookup_reference', description='Retrieve appearance examples and their review status by class or exact ID.',
         inputSchema=REFERENCE_QUERY),
    dict(name='lookup_country_context', description='Read the current country’s regional Cashew and lookalike '
         'background, with source dates and map links. Use for a specific uncertain crop identity.', inputSchema=obj()),
    dict(name='propose_land_boxes', description='For three or more visible classes, inspect numbered field proposals on the full RGB. Supply your own points for stands the detector misses.', inputSchema=obj()),
    dict(name='segment_land_units', description=f'X-AnyLabeling SAM2 on the full RGB for up to {MAX_SEGMENT_UNITS} land units per call. Use for a mixed scene and for supported Cashew interiors within it.',
         inputSchema=obj(units=array(SAM_PROMPT, MAX_SEGMENT_UNITS, 1))),
    dict(name='submit_annotation', description='Submit source-resolution geometry and selected SAM masks. '
         'woody_background is [] for none, coarse polygons for suspect woodland, or null for a woody remainder. '
         'A direct Mix tree unit in suspected woodland needs confirmed_noncashew_xy at positive local evidence. '
         'Use polygon=null for a homogeneous tile, or for the visually confirmed remaining class below SAM in a two-class tile. Order units low to high priority. Explain review points left uncertain or reassigned using review_exceptions.',
         inputSchema=obj(**COMPOSITION_FIELDS)),
]

IMAGE_BRIDGE = r"""For dynamic tools through functions.exec use this bridge for tool images:
const r=await tools.TOOL(ARGS);
const s=typeof r==="string"?r:JSON.stringify(r);
const p=/data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g;
const urls=s.match(p)||[]; text(s.replace(p,"[image rendered]"));
for(let i=0; i<urls.length; i+=2) {
  for(const u of urls.slice(i,i+2)) image(u);
  if(i+2<urls.length) await yield_control();
}
Replace TOOL and ARGS with a declared tool and its arguments. Finish any yielded cell with functions.wait
to see all pages before selecting a mask. Each page uses the same cached tool result."""


def propose_land_boxes(context):
    folder = context['case_dir'] / 'detector'
    path = folder / 'PROPOSALS.json'
    if not path.is_file():
        python, env = local_model_runtime()
        env['YOLO_CONFIG_DIR'] = str(folder / 'config')
        command = [python, str(SKILL_ROOT / 'scripts/propose_fields.py'), str(context['scene']), str(folder)]
        folder.mkdir(exist_ok=True)
        try:
            with (folder / 'detector.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as error:
            write(folder / 'unavailable.json', {'reason': str(error)})
            return {'success': True, 'boxes': [], 'detector_unavailable': True,
                    'next_step': 'Choose interiors on the supplied RGB and call segment_land_units.'}
    proposals = read(path)
    if proposals['rgb_sha256'] != digest(context['scene']):
        raise ValueError('The proposals must use the selected RGB')
    boxes = proposals['boxes']
    images = ([record_image(folder / 'numbered_boxes.png', context['case'] + '_boxes', 'numbered_region_proposals')]
              if boxes else [])
    return {'success': True, 'boxes': boxes, 'images': images}


def compose(arguments, context):
    registry = context['registered_frames']
    review = context['imagery_uncertainty']
    woody = woody_background_mask(review.get('woody_background', []), arguments.get('woody_background', []))
    direct = {item['unit_id']: item for item in arguments['direct_areas']}
    sam = {item['unit_id']: item for item in arguments['accepted_units']}
    order = arguments['priority_order']
    if len(order) != len(set(order)) or set(order) != set(direct) | set(sam):
        raise ValueError('Put each accepted unit once in priority_order')
    if len(order) != len(arguments['direct_areas']) + len(arguments['accepted_units']):
        raise ValueError('Each direct or SAM unit must have a unique ID')
    whole = [unit for unit in direct.values() if unit['polygon'] is None]
    complement = None
    if whole and len(order) > 1:
        if (len(direct) != 1 or order[0] != whole[0]['unit_id'] or
                len({unit['class_id'] for unit in sam.values()}) != 1 or
                next(iter(sam.values()))['class_id'] == whole[0]['class_id']):
            raise ValueError('A full-footprint background is allowed below SAM only for two distinct classes')
        complement = whole[0]['class_id']
    if whole and woody.all():
        raise ValueError('The residual woody background is uncertain. Omit the full-footprint class and submit '
                         'only identifiable individual units; the remaining visible area is automatically 10.')

    required_images = {context['case'] + '_scene'}
    labels = np.full((SIZE, SIZE), UNLABELLED, dtype=np.uint8)
    labels[woody] = 10
    sam_labels = np.full_like(labels, UNLABELLED)
    clipped = {}
    for unit_id in order:
        if unit_id in direct:
            unit = direct[unit_id]
            require_evidence(unit, registry, context['case'] + '_scene' if whole else None)
            required_images.update(unit['evidence_image_ids'])
            if unit['polygon'] is None:
                labels[~woody] = unit['class_id']
                continue
            mask = polygon_mask(unit['polygon'])
            points = [unit['confirmed_noncashew_xy']] if 'confirmed_noncashew_xy' in unit else []
            evidence = unit['evidence_image_ids']
        else:
            unit = sam[unit_id]
            base = sam_unit_folder(context, unit_id)
            if unit['revision'] != next_number(base) - 1:
                raise ValueError('Select the latest reviewed revision of the SAM unit')
            folder = base / str(unit['revision'])
            info = read(folder / 'receipt.json')
            if info['class_id'] != unit['class_id']:
                raise ValueError('Use the class that was inspected in this SAM overlay')
            if info['candidate_image_id'] not in registry:
                raise ValueError('The selected SAM candidate sheet must be delivered before composition')
            required_images.update(info['evidence_image_ids'])
            required_images.add(info['candidate_image_id'])
            mask = np.asarray(Image.open(folder / f"mask{unit['candidate_index']}.png")) > 0
            points = info['positive_points']
            evidence = info['evidence_image_ids']
        if unit['class_id'] == 1 and (mask & woody).any():
            if context['case'] + '_scene' not in evidence or mask.all():
                points = []
            limited = confirmed_other_wood(mask, woody, labels, points)
            removed = int((mask & ~limited).sum())
            if removed:
                clipped[unit_id] = removed
            mask = limited
        labels[mask] = unit['class_id']
        if unit_id in sam:
            sam_labels[mask] = unit['class_id']
    new_legacy_outline = apply_uncertainty(context, labels, arguments['uncertain_polygons'])
    if context.get('valid_mask_native128') is not None:
        scale = SIZE // GRID
        visible = context['valid_mask_native128'].repeat(scale, axis=0).repeat(scale, axis=1)
        labels[~visible] = UNLABELLED
    counts = class_counts(labels)
    if set(counts) == {str(UNLABELLED)}:
        raise ValueError('Assign identifiable source areas, or outline the visible area of unresolved identity')
    known = {int(code) for code in counts if int(code) < 4}
    classes = set(review.get('visible_classes', [])) | known
    if review.get('reported_presence') == 'present' or any(c['source'] == 'human_point' for c in review.get('candidates', [])):
        classes.add(2)
    mixed = len(classes) >= 2
    if complement is not None and len(classes) > 2:
        raise ValueError('The scene contains three or more classes. Delineate their supported regions individually.')
    if len(classes) >= 3:
        box_id = context['case'] + '_boxes'
        if box_id in registry:
            required_images.add(box_id)
        elif not (context['case_dir'] / 'detector/unavailable.json').is_file():
            raise ValueError('Inspect propose_land_boxes first, then prompt SAM on appropriate visible interiors')
    sam_labels[sam_labels != labels] = UNLABELLED
    sam_counts = class_counts(sam_labels)
    sam_counts.pop(str(UNLABELLED), None)
    if mixed and not sam_counts and (known or not context['activity']['sam_requests']):
        raise ValueError('Mixed scene: use segment_land_units, inspect the candidates, and accept geometry '
                         'that survives on the source RGB.')
    if mixed and '2' in counts and complement != 2 and '2' not in sam_counts:
        raise ValueError('Inspect and accept actual SAM geometry for supported Cashew in this mixed scene')
    exceptions = arguments.get('review_exceptions', [])
    checks = review_cashew_coverage(context, labels, sam_labels, mixed, complement, exceptions, woody)
    for item in exceptions:
        required_images.update(item['evidence_image_ids'])
    root = context['case_dir'] / 'compositions'
    index = next_number(root)
    folder = root / str(index)
    folder.mkdir(parents=True)
    mode = ('two_classes_sam2_complement' if complement is not None else 'whole_sample' if whole else 'direct_and_sam' if direct and sam else 'sam'
            if sam else 'direct_polygons' if direct else 'unknown')
    annotation_id = f"{context['case']}_annotation{index}"
    Image.fromarray(labels).save(folder / 'labels1024.png')
    composition = dict(arguments, composition_index=index, annotation_method=mode, annotation_id=annotation_id,
                       mixed_scene=mixed, surviving_sam_source_counts=sam_counts, cashew_review_checks=checks,
                       suspected_woody_pixels=int(woody.sum()), other_tree_clipped_pixels_by_unit=clipped,
                       required_image_ids=sorted(required_images), source_class_counts=counts)
    write(folder / 'COMPOSITION.json', composition)
    write(root / 'LATEST.json', {'folder': str(folder)})
    if new_legacy_outline:
        context['activity']['legacy_review_polygons'] = list(arguments['uncertain_polygons'])
    receipt = {'success': True, 'annotation_id': annotation_id, 'annotation_method': mode,
               'cashew_in_source': '2' in counts, 'other_tree_clipped_pixels_by_unit': clipped}
    if context.get('run_dir'):
        try:
            missing = image_delivery_report(Path(context['run_dir']), registry, composition['required_image_ids'])['missing_image_ids']
        except FileNotFoundError:
            # The finished-turn audit will read the persisted rollout before export.
            missing = []
        if missing:
            receipt.update(images=[registry[i] for i in missing[:4]], missing_evidence_image_ids=missing,
                           evidence_action='Inspect these source or candidate images before returning the saved '
                                           'annotation ID. Resubmit geometry only if they change your decision.')
    return receipt


def view(arguments, context):
    mode = arguments['mode']
    if mode == 'scene':
        return {'success': True, 'images': [record_image(context['scene'], context['case']+'_scene', 'full_scene_RGB')]}
    cache = context['activity'].setdefault('views', {})
    bbox = None
    if mode == 'landscape':
        key = (mode, arguments['scale'] or 3)
    else:
        if arguments['bbox'] is None:
            raise ValueError('Choose a detail box in complete 1024 coordinates')
        left, top, right, bottom = map(int, arguments['bbox'])
        if right <= left or bottom <= top:
            raise ValueError('Use an ordered XYXY detail box')
        left = min(left, SIZE - 64)
        top = min(top, SIZE - 64)
        bbox = (left, top, max(right, left + 64), max(bottom, top + 64))
        key = (mode, bbox)
    if key in cache:
        return cache[key]
    base = context['case_dir'] / 'views'
    number = next_number(base, 'view')
    folder = base / f'view{number}'
    folder.mkdir(parents=True)
    identity = f"{context['case']}_view{number}"
    if mode == 'landscape':
        frame = context['frame']
        view_dates = frame['acquisition_dates']
        if frame['provider'] == 'Esri':
            path = folder / 'landscape.png'
            bounds = esri.landscape(context['grid'], frame, arguments['scale'] or 3, path)
            if 'label_period' in context['grid']:
                view_dates = read(folder/'landscape_dates.json')
        elif frame.get('google_backend') == gehi.BACKEND:
            path = folder / 'landscape.png'
            bounds = gehi.landscape(context['grid'], frame, arguments['scale'] or 3, path)
        else:
            bounds = None
            path = Path(frame['provider_context_path'])
        result = {'success': True, 'images': [record_image(path, identity, 'raw_landscape',
                    bounds_wgs84=bounds, source_date=view_dates, display_note='Landscape context for interpreting the terrain. '
                    'Only the registered full scene is the label and SAM coordinate system.')]}
        cache[key] = result
        return result

    path = folder / 'detail.png'
    if context.get('native_image_path'):
        x0, y0, x1, y1 = bbox
        frame = context['frame']
        extent = window_bounds(Window(x0,y0,x1-x0,y1-y0), rasterio.Affine(*frame['image_transform']))
        detail = registered_detail(context['native_image_path'],extent,frame['image_crs'],max_size=SIZE)
    else:
        with Image.open(context['scene']) as original:
            detail = original.convert('RGB').crop(bbox)
    detail.save(path)
    result = {'success': True, 'full1024_bbox': bbox,
            'images': [record_image(path, identity, 'raw_detail', display_note='Extent in complete scene pixels')]}
    cache[key] = result
    return result


def view_details(arguments, context):
    views = []
    images = []
    for box in arguments['boxes']:
        result = view(dict(mode='detail', bbox=box, scale=None), context)
        image, = result['images']
        views.append(dict(image_id=image['image_id'], full1024_bbox=result['full1024_bbox']))
        images.append(image)
    return dict(success=True, views=views, images=images)


def dispatch_tool(name, arguments, context):
    if name == 'propose_land_boxes':
        return propose_land_boxes(context)
    if name == 'view_comparison':
        return {'success': True, 'images': [context['comparison_image']]}
    if name == 'lookup_country_context':
        country = context['grid']['country']
        return dict(lookup_country_context(country), local_administrative_area=administrative_area(country, context['bounds']))
    return {'view_image_scale': view, 'view_details': view_details, 'lookup_reference': lookup_reference,
            'segment_land_units': segment, 'submit_annotation': compose}[name](arguments, context)


def keep_dry_scene(context):
    return context['source_confirmed_local_dry_season']


def dispatch_plan_tool(name, arguments, context):
    if 'review_suggestions' in context:
        return {'success': False, 'reason': 'Use the supplied group review to return keep and omit IDs.', 'images': []}
    activity = context['activity']
    if activity.get('auxiliary_checks', 0) >= MAX_AUXILIARY_CHECKS:
        return {'success': False, 'reason': 'Use the images already seen and return your confirmed interiors.', 'images': []}
    activity['auxiliary_checks'] = activity.get('auxiliary_checks', 0) + 1
    if name == 'propose_land_boxes':
        result = propose_land_boxes(context)
        activity['detector_checked'] = True
        return result
    if name == 'lookup_reference':
        arguments = dict(arguments, limit=1)
        comparison = arguments.get('comparison_class_id')
        if comparison is not None and comparison != arguments['class_id']:
            first = lookup_reference(arguments, context)
            second = lookup_reference(dict(class_id=comparison, reference_id=None, limit=1,
                                           appearance=arguments.get('appearance')), context)
            return {'success': True, 'references': first['references'] + second['references'],
                    'images': first['images'] + second['images']}
    return dispatch_tool(name, arguments, context)


def annotation_plan(result):
    if 'representatives' not in result:
        return result
    regions = [dict(mask_id=identity, class_id=REPRESENTATIVE_CLASSES[name],
                    cashew_evidence=result['cashew_evidence'] if name == 'cashew' else None)
               for name, identity in result['representatives'].items() if identity is not None]
    return dict(result, regions=regions)


def validate_plan(context, result):
    if 'keep' in result:
        choices = context['review_suggestions']
        try:
            kept = resolve_keep(result, choices['groups'], choices['classes'])
        except ValueError as error:
            return {'eligible': False, 'reason': str(error)}
        return {'eligible': True, 'kept': kept}
    result = annotation_plan(result)
    if 'representatives' in result:
        selected = [row['mask_id'] for row in result['regions']]
        ids = {item['id'] for item in context['box_proposals']['masks']}
        if not set(selected).issubset(ids) or len(set(selected)) != len(selected):
            return {'eligible': False, 'reason': 'Each class must use a different supplied mask ID'}
    if result['whole_class'] is not None and (result['regions'] or result.get('direct_regions')):
        return {'eligible': False, 'reason': 'A homogeneous tile must have no regional marks'}
    background = result.get('background')
    if background is not None:
        foreground = {region['class_id'] for region in result['regions']}
        categories = foreground | {background['class_id']}
        if (result['whole_class'] is not None or result.get('direct_regions') or len(foreground) != 1
                or len(categories) != 2 or set(result['visible_classes']) != categories
                or any(region['box'] is None for region in result['regions'])
                or not background['visible_cue'].strip()):
            return {'eligible': False, 'reason': 'Use one boxed foreground class and its distinct confirmed remainder'}
        if not context['activity'].get('detector_checked'):
            return {'eligible': False, 'reason': 'Inspect propose_land_boxes before selecting the foreground prompts'}
    try:
        for region in result.get('direct_regions', []):
            polygon_mask(region['polygon'])
        for polygon in result.get('uncertain_polygons', []):
            polygon_mask(polygon)
    except ValueError as error:
        return {'eligible': False, 'reason': str(error)}
    for unit in result['regions']:
        box = unit.get('box')
        if box is not None and (box[0] == box[2] or box[1] == box[3]):
            return {'eligible': False, 'reason': 'A SAM2 box must have positive area'}
    if set(result['evidence_image_ids']) - context['registered_frames'].keys():
        return {'eligible': False, 'reason': 'Cite only delivered image IDs'}
    if result.get('request_earlier_image') and (context.get('code_preflight') or (
            not keep_dry_scene(context) and not context.get('dry_search_exhausted') and not result.get('needs_dry_season'))):
        return {'eligible': True, 'disposition': 'rejected_image', 'reason': 'another_image_needed'}
    return {'eligible': True, 'disposition': 'segment'}


def cashew_plan_for_export(result, *, dry_scene, comparison_image_id=None):
    corroborated = comparison_image_id is not None and comparison_image_id in result['evidence_image_ids']
    allow_cashew = dry_scene or corroborated
    withheld = []
    withheld_boxes = []
    kept = {}
    for field in ('regions', 'direct_regions'):
        kept[field] = []
        for number, region in enumerate(result.get(field, []), 1):
            if region['class_id'] != 2 or (allow_cashew and (region.get('cashew_evidence') or '').strip()):
                kept[field].append(region)
            else:
                withheld.append(number if field == 'regions' else f'direct_{number}')
                if result.get('background'):
                    withheld_boxes.append(region['box'])
    whole = result['whole_class']
    if whole == 2 and not (allow_cashew and (result.get('whole_cashew_evidence') or '').strip()):
        whole = None
        withheld.append('whole_sample')
    if whole == 1 and result.get('needs_dry_season'):
        whole = None
    if 'background' in result:
        background = result['background']
        if background and background['class_id'] == 2 and not allow_cashew:
            withheld.append('background')
            background = None
        kept.update(background=background, withheld_foreground_boxes=withheld_boxes)
    return dict(result, whole_class=whole, **kept), withheld


def validate_final(context, result):
    missing_dry_season_view = result.get('needs_dry_season') and not context['source_confirmed_local_dry_season']
    if not keep_dry_scene(context) and (result.get('request_earlier_image') or missing_dry_season_view):
        return {'eligible': True, 'disposition': 'rejected_image', 'reason': 'another_image_needed'}
    latest = context['case_dir'] / 'compositions/LATEST.json'
    if not latest.exists():
        return {'eligible': False, 'reason': 'Submit the source annotation first'}
    folder = Path(read(latest)['folder'])
    composition = read(folder / 'COMPOSITION.json')
    if not composition.get('annotation_id') or result.get('annotation_id') != composition['annotation_id']:
        return {'eligible': False, 'reason': 'Return the ID of the last accepted source annotation'}
    counts = composition['source_class_counts']
    if '2' in counts:
        presence = 'present'
    elif '10' in counts or str(UNLABELLED) in counts:
        presence = 'uncertain'
    else:
        presence = 'absent'
    return {'eligible': True, 'disposition': 'publish', 'composition_folder': str(folder),
            'annotation_method': composition['annotation_method'], 'annotation_id': composition['annotation_id'],
            'cashew_presence': presence}


def publish(context, validation):
    source = Path(validation['composition_folder'])
    target = context['case_dir'] / 'result'
    target.mkdir(exist_ok=True)
    rgb = target / 'image_rgb.png'
    shutil.copy2(context['scene'], rgb)
    source_png = target / 'labels1024.png'
    shutil.copy2(source / 'labels1024.png', source_png)
    labels = np.asarray(Image.open(source_png))
    native = area_mode(labels, (GRID, GRID))
    valid = native != UNLABELLED
    if context.get('valid_mask_native128') is not None:
        valid &= context['valid_mask_native128']
    # Historical consumers use 10 to ignore missing data; the internal mask identifies those cells.
    encoded = np.where(valid, native, 10).astype(np.uint8)
    Image.fromarray(encoded).save(target / 'labels_native128.png')
    colored = cover_image(labels)
    colored.save(target / 'segmentation1024.png')
    overlay = Image.blend(Image.open(rgb).convert('RGB'), colored, .4)
    overlay.save(target / 'overlay1024.png')
    sheet = Image.new('RGB', (2 * SIZE, SIZE + 40), 'white')
    sheet.paste(overlay, (0, 40))
    sheet.paste(cover_image(native).resize((SIZE, SIZE), Image.Resampling.NEAREST), (SIZE, 40))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 14), 'Source annotation', fill='black')
    draw.text((SIZE + 12, 14), 'AEF area vote: gray = uncertain, black = no label', fill='black')
    sheet.save(target / 'review_actual_and_native.png')
    source_tif = target / 'labels_source.tif'
    height, width = labels.shape
    transform = context['native_transform'] * rasterio.Affine.scale(GRID / width, GRID / height)
    with rasterio.open(source_tif, 'w', driver='GTiff', width=width, height=height, count=1, dtype='uint8',
                       crs=context['crs'], transform=transform, compress='deflate', nodata=UNLABELLED) as dst:
        dst.write(labels, 1)
        dst.update_tags(unknown_class=10, label_basis='source_rgb')
    filename = target / 'labels_native128.tif'
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(filename, 'w', driver='GTiff', width=GRID, height=GRID, count=1, dtype='uint8',
                           crs=context['crs'], transform=context['native_transform'], compress='deflate') as dst:
            dst.write(encoded, 1)
            dst.write_mask(valid.astype(np.uint8) * 255)
            dst.update_tags(unknown_class=10, resampling='categorical_area_mode', annotation_method=validation['annotation_method'],
                            label_period=context['grid']['label_period'], provider=context['frame']['provider'],
                            acquisition_dates=','.join(context['frame']['acquisition_dates']))
    composition = read(source / 'COMPOSITION.json')
    checks = composition['cashew_review_checks']
    for item in checks:
        if 'point_xy' in item:
            x, y = item['point_xy']
            col, row = int(x * GRID // width), int(y * GRID // height)
            item.update(native_col_row=[col, row], native_class=int(encoded[row, col]),
                        native_valid=bool(valid[row, col]))
        elif 'box_xyxy' in item:
            x0, y0, x1, y1 = item['box_xyxy']
            xs, ys = (np.arange(GRID) + .5) * width / GRID, (np.arange(GRID) + .5) * height / GRID
            item['native_class_counts'] = class_counts(native[np.ix_((ys >= y0) & (ys < y1),
                                                                     (xs >= x0) & (xs < x1))])
    composition.update(native_class_counts=class_counts(encoded), resampling='categorical_area_mode',
                       semantic_uncertain_native_pixels=int(((native == 10) & valid).sum()),
                       unlabelled_native_pixels=int((~valid).sum()))
    write(target / 'COMPOSITION.json', composition)
    return dict(annotation_method=validation['annotation_method'], sam_requests=context['activity']['sam_requests'],
                class_pixel_counts=composition['native_class_counts'],
                surviving_sam_source_counts=composition['surviving_sam_source_counts'], cashew_review_checks=checks,
                region_matching=composition.get('region_matching'),
                source_label_png=str(source_png), source_label_geotiff=str(source_tif), source_label_sha256=digest(source_tif),
                resampling=composition['resampling'], presence_basis='source_rgb', native_has_cashew=bool((native == 2).any()),
                unlabelled_native_pixels=composition['unlabelled_native_pixels'],
                semantic_uncertain_native_pixels=composition['semantic_uncertain_native_pixels'],
                rgb_png=str(rgb), label_png=str(target/'labels_native128.png'),
                label_preview_png=str(target/'review_actual_and_native.png'),
                label_geotiff=str(filename), label_sha256=digest(filename),
                imagery_provider=context['frame']['provider'], acquisition_dates=context['frame']['acquisition_dates'])


def context_for(selection_path, case_dir, *, existing_scene=False):
    selection = read(selection_path)
    grid, frame, decision = selection['grid'], selection['frame'], selection['review']
    code_preflight = selection.get('selection_method') == 'code_preflight'
    if selection['status'] != 'selected' or (not code_preflight and read(selection['vision_audit'])['status'] != 'verified'):
        raise ValueError('A freshly reviewed source image with a verified visual delivery is required')
    reason = review_rejection(grid['label_period'], frame, decision)
    if reason:
        raise ValueError('Selected source image failed the current policy: ' + reason)
    with rasterio.open(grid['aef_path']) as aef:
        if (aef.shape != (GRID, GRID) or aef.crs.to_string() != grid['native_crs']
                or list(aef.transform)[:6] != grid['native_transform']):
            raise ValueError('Original AEF grid changed after imagery selection')
        crs, native = aef.crs, aef.transform
    image_path = Path(frame['image_path'])
    if (digest(image_path) != frame['sha256'] or frame['image_crs'] != grid['native_crs']
            or frame['image_transform'] != list(native * rasterio.Affine.scale(GRID / SIZE))[:6]):
        raise ValueError('Registered image checksum or footprint differs from the accepted selection')
    valid = read_support(frame)
    scene = case_dir / 'query_rgb.png'
    with Image.open(image_path) as image:
        if image.size != (SIZE, SIZE):
            raise ValueError('Expected a complete 1024×1024 source image aligned to the AEF tile')
        rgb = image.convert('RGB')
        if existing_scene:
            with Image.open(scene) as previous:
                if previous.size != rgb.size or previous.convert('RGB').tobytes() != rgb.tobytes():
                    raise ValueError('The saved query image differs from the selected RGB')
        else:
            case_dir.mkdir(parents=True, exist_ok=False)
            rgb.save(scene)
    case_key = hashlib.sha256(str(case_dir).encode()).hexdigest()[:20]
    return dict(case=grid['candidate_id'], case_dir=case_dir, grid=grid, frame=frame, scene=scene,
                crs=crs, native_transform=native, bounds=grid['bounds_wgs84'],
                valid_mask_native128=valid, imagery_uncertainty=imagery_uncertainty(decision,
                    same_image_review([selection['same_image_operator_review']], grid, frame)
                    if selection.get('same_image_operator_review') else {}),
                native_image_path=frame.get('source_raster') if valid.all() else None,
                source_confirmed_local_dry_season=decision['confirmed_local_dry_season'],
                code_preflight=code_preflight, source_season=selection.get('season'),
                acquisition_window=selection.get('acquisition_window'),
                backend_case='annot_' + case_key, activity={'sam_requests': 0})


def finalize_annotation(context, selection_path, result, registry, model, validation):
    case_dir = context['case_dir']
    usage = dict(auxiliary_checks_used=context['activity'].get('auxiliary_checks', 0),
                 service_tier=context.get('service_tier'))
    if validation['disposition'] == 'rejected_image':
        verify_images(case_dir, case_dir, registry, [context['case'] + '_scene'])
        receipt = dict(status='rejected_image', reason=validation['reason'],
                       needs_dry_season=result.get('needs_dry_season', False), selection=str(selection_path),
                       model=model, effort=EFFORT, **usage)
        write(case_dir / 'receipt.json', receipt)
        return receipt
    composition = read(Path(validation['composition_folder']) / 'COMPOSITION.json')
    audit = verify_images(case_dir, case_dir, registry, composition['required_image_ids'],
                          through_turn=validation['decision_turn_id'])
    receipt = publish(context, validation)
    return dict(receipt, status='completed', model=model, effort=EFFORT, annotation_id=validation['annotation_id'],
                decision_turn_id=validation['decision_turn_id'],
                grid=context['grid'], selection=str(selection_path), selection_sha256=digest(selection_path),
                vision_audit=str(audit), final_cashew_presence=validation['cashew_presence'], **usage)


def review_box_proposals(context, export_plan, session, registry, evidence, turn_id, *, search_for_dry):
    report = context['box_proposals']
    suggestions = suggest_groups(context, export_plan, match=not search_for_dry)
    kept = suggestions['representatives']
    review_seconds = 0
    if suggestions['additions']:
        panel = mask_panel(context, [m for m in report['masks'] if m['id'] in suggestions['classes']],
                           'matched_groups', suggestions['classes'])
        folder = context['case_dir'] / 'keep_review'
        metadata = dict(groups=suggestions['groups'], representatives=suggestions['representatives'])
        prompt = (SKILL_ROOT / 'references/keep-masks-prompt.md').read_text()
        transport.prepare_turn(session.thread, folder, prompt, [panel], metadata, KEEP_SCHEMA,
                               prior_images=registry.values())
        started = monotonic()
        _, registry = session.run_turn(folder, dict(context, review_suggestions=suggestions), timeout_seconds=900)
        review_seconds = round(monotonic() - started, 2)
        turn_id = read(folder / 'status.json')['turn_id']
        kept = read(folder / 'final_validation.json')['kept']
        evidence = sorted(set(evidence) | {panel['image_id']})
        verify_images(context['case_dir'], folder, registry, evidence, through_turn=turn_id)
    annotation_id = commit_review(context, export_plan, evidence, suggestions, kept)
    return annotation_id, registry, turn_id, review_seconds


def publish_local_plan(context, plan):
    """Publish a verified Sol plan and resume after completed SAM work if needed."""
    with (context['case_dir'] / '.local_publish.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Local publication is already running for this attempt') from error
        return _publish_local_plan(context, plan)


def _publish_local_plan(context, plan):
    case_dir = context['case_dir']
    selection = Path(plan['selection'])
    if digest(selection) != plan['selection_sha256']:
        raise ValueError('Selection changed after the Sol plan was frozen')
    registry = plan['registry']
    context['registered_frames'] = registry
    context['service_tier'] = plan['service_tier']
    context['activity']['auxiliary_checks'] = plan['auxiliary_checks']
    verify_images(case_dir, case_dir, registry, plan['evidence'], through_turn=plan['turn_id'])
    started = monotonic()
    progress = case_dir / 'PROGRESS.json'
    composed = case_dir / 'LOCAL_COMPOSITION.json'
    review_check = {}
    export_plan = plan['export_plan']
    if 'woody_review' in plan:
        from woody_tile_review import run as review_woodland
        write(progress, dict(stage='woody_review', updated_epoch=time(), sol_seconds=plan['sol_seconds']))
        export_plan, review_check = review_woodland(context, plan)
    if composed.is_file():
        done = read(composed)
        folder = Path(read(case_dir / 'compositions/LATEST.json')['folder'])
        if digest(folder / 'labels1024.png') != done['mask_sha256']:
            raise ValueError('Completed source mask changed after composition')
    else:
        write(progress, dict(stage='sam2_masks', updated_epoch=time(), sol_seconds=plan['sol_seconds'],
                             requested_units=len(export_plan['regions'])))
        annotation_id, sam_seconds = materialize(export_plan, context, plan['evidence'])
        folder = Path(read(case_dir / 'compositions/LATEST.json')['folder'])
        done = dict(annotation_id=annotation_id, sam_seconds=round(sam_seconds, 2),
                    sam_requests=context['activity']['sam_requests'],
                    reused_sam_masks=context['activity'].get('reused_sam_masks', 0),
                    mask_sha256=digest(folder / 'labels1024.png'))
        write(composed, done)
    context['activity']['sam_requests'] = done['sam_requests']
    validation = validate_final(context, {'annotation_id': done['annotation_id']})
    if not validation['eligible']:
        raise ValueError(validation['reason'])
    validation['decision_turn_id'] = plan['turn_id']
    write(case_dir / 'PUBLICATION_VALIDATION.json', validation)
    write(progress, dict(stage='export', updated_epoch=time(), sol_seconds=plan['sol_seconds'],
                         sam_seconds=done['sam_seconds']))
    receipt = finalize_annotation(context, selection, plan['result'], registry, plan['model'], validation)
    elapsed = round(plan['elapsed_before_local'] + monotonic() - started, 2)
    timing = dict(preparation_seconds=plan['preparation_seconds'], sol_seconds=plan['sol_seconds'],
                  keep_review_seconds=0, sam_seconds=done['sam_seconds'], elapsed_seconds=elapsed,
                  post_sam_model_calls=0, hosted_annotation_turns=plan['hosted_annotation_turns'])
    receipt.update(timing, reused_sam_masks=done['reused_sam_masks'], token_usage=plan['token_usage'],
                   dry_comparison_requested=plan['dry_comparison_requested'],
                   confirmed_local_dry_season=plan['fixed_scene'],
                   dry_search_exhausted=plan['dry_search_exhausted'],
                   cashew_evidence_path=str(case_dir / 'CASHEW_EVIDENCE.json'))
    if review_check:
        receipt['woody_review'] = {key: review_check[key] for key in ('status', 'seconds', 'reviewed', 'added', 'withheld')}
    write(case_dir / 'receipt.json', receipt)
    write(progress, dict(stage='completed', updated_epoch=time(), **timing))
    return receipt


def resume_annotation(case_dir, *, allow_deferred=False):
    plan = read(case_dir / 'LOCAL_PLAN.json')
    if plan['status'] == 'dry_deferred':
        if not allow_deferred:
            raise ValueError('The waiting plan was deferred for a different working image')
        plan = dict(plan, status='ready', dry_search_exhausted=True)
        write(case_dir / 'LOCAL_PLAN.json', plan)
    context = context_for(Path(plan['selection']), case_dir, existing_scene=True)
    return publish_local_plan(context, plan)


def run_annotation(selection_path, case_dir, model=MODEL, *, previous_auxiliary_checks=0,
                   dry_search_exhausted=False, comparison_selection=None, comparison_frame=None,
                   region_matching=False, defer_for_dry=False, woody_review=False):
    context = context_for(selection_path, case_dir)
    context['activity']['auxiliary_checks'] = previous_auxiliary_checks
    context['dry_search_exhausted'] = dry_search_exhausted
    frame, grid = context['frame'], context['grid']
    case = grid['candidate_id']
    references = SKILL_ROOT / 'references'
    prompt = '\n\n'.join((references / name).read_text()
                         for name in ('annotation-prompt.md', 'visual-cues.md'))
    fixed_scene = keep_dry_scene(context)
    evidence_tools = [item for item in TOOLS if item['name'] not in
                      ('submit_annotation', 'segment_land_units')]
    initial = record_image(context['scene'], case + '_scene', 'full_scene_RGB',
                           source_date=frame['acquisition_dates'], provider=frame['provider'],
                           bounds_wgs84=context['bounds'], actual_AEF_grid=[GRID, GRID])
    comparison = None
    reviewed_comparison = None
    if dry_search_exhausted and comparison_selection:
        reviewed_comparison = read(comparison_selection)
        comparison_frame = reviewed_comparison['frame']
    aligned_comparison = (comparison_frame and not frame_rejection(grid['label_period'], comparison_frame)
                          and comparison_frame['image_transform'] == frame['image_transform']
                          and comparison_frame['image_crs'] == frame['image_crs']
                          and distinct_capture_periods(frame, comparison_frame))
    if reviewed_comparison and (not aligned_comparison or reviewed_comparison['grid'] != grid
                                or not reviewed_comparison['review']['usable']
                                or read(reviewed_comparison['vision_audit'])['status'] != 'verified'):
        raise ValueError('Corroborating imagery must cover the same grid at a different verified date')
    if aligned_comparison:
        comparison = record_image(comparison_frame['image_path'], case + '_different_date',
                                  'same_footprint_different_date_RGB',
                                  source_date=comparison_frame['acquisition_dates'], provider=comparison_frame['provider'],
                                  bounds_wgs84=context['bounds'], sha256=comparison_frame['sha256'])
        context['comparison_image'] = comparison
        evidence_tools.append(COMPARISON_TOOL)
    schema = DRY_PLAN_SCHEMA if fixed_scene or dry_search_exhausted else PLAN_SCHEMA
    if context['code_preflight']:
        schema = PREFLIGHT_PLAN_SCHEMA
        prompt += '\n\n' + (references / 'code-selected-annotation.md').read_text()
    if woody_review:
        prompt += '\n\n' + (references / 'woody-tile-review.md').read_text()
    progress = case_dir / 'PROGRESS.json'
    started = monotonic()
    write(progress, dict(stage='preparing_visual_context', updated_epoch=time()))
    inputs = [initial]
    metadata = dict(case_id=case, country=grid['country'], label_period=grid['label_period'],
                    missing_native128_pixels=int((~context['valid_mask_native128']).sum()),
                    auxiliary_checks_remaining=MAX_AUXILIARY_CHECKS - previous_auxiliary_checks,
                    retain_qualified_dry_scene=fixed_scene,
                    dry_season_search_exhausted=dry_search_exhausted and not fixed_scene,
                    acquisition_window=context['acquisition_window'] or date_window(grid['label_period']), image_size=[SIZE, SIZE])
    if context['code_preflight']:
        metadata.update(primary_season=context['source_season'], dry_candidate_available=defer_for_dry)
    if region_matching:
        setup = dict(status='local_matcher_unavailable')
        if WEIGHTS.is_file() and DINO_REPO.is_dir():
            proposals = propose_land_boxes(context)
            setup = dict(status='detector_unavailable' if proposals.get('detector_unavailable') else 'no_boxes')
            if proposals['boxes']:
                report = segment_boxes(proposals['boxes'], context)
                setup = dict(status='enabled' if report['masks'] else 'no_masks', box_count=report['box_count'],
                             mask_count=report['mask_count'], sam_seconds=report['seconds'])
                if report['masks']:
                    context['box_proposals'] = report
                    inputs.append(mask_panel(context, report['masks'], 'numbered_masks'))
                    metadata['mask_ids'] = [m['id'] for m in report['masks']]
                    prompt = '\n\n'.join((references / name).read_text()
                                         for name in ('region-matching-prompt.md', 'visual-cues.md'))
                    schema = DRY_MATCH_PLAN_SCHEMA if fixed_scene or dry_search_exhausted else MATCH_PLAN_SCHEMA
                    evidence_tools = [item for item in evidence_tools if item['name'] != 'propose_land_boxes']
        write(case_dir / 'REGION_MATCHING_SETUP.json', setup)
    thread = transport.prepare_thread(IMAGE_BRIDGE, evidence_tools, model=model)
    review = context['imagery_uncertainty']
    if review['reported_presence'] is not None:
        metadata['imagery_review_uncertainty'] = {k: review[k] for k in
                ('candidates', 'search_windows', 'reported_presence', 'visible_classes', 'explanation')}
        metadata['suspect_woody_background'] = review['woody_background'] != []
    elif review['candidates'] or review['search_windows']:
        metadata['same_image_observations'] = dict(points=review['candidates'], search_windows=review['search_windows'])
    transport.prepare_turn(thread, case_dir, prompt, inputs, metadata, schema)
    initial_evidence = {record['image_id'] for record in inputs}
    tool_module = SimpleNamespace(dispatch_tool=dispatch_plan_tool, validate_final=validate_plan)
    session = transport.Session(case_dir, thread, tool_module)
    decision_dir = case_dir
    sol_started = monotonic()
    preparation_seconds = round(sol_started - started, 2)
    write(progress, dict(stage='sol_classification', updated_epoch=time(), preparation_seconds=preparation_seconds))
    try:
        session.start()
        context['service_tier'] = session.service_tier
        result, registry = session.run_turn(case_dir, context, timeout_seconds=3600)
        validation = read(case_dir / 'final_validation.json')
        if validation['disposition'] == 'rejected_image':
            return finalize_annotation(context, selection_path, result, registry, model, validation)
        evidence = sorted(initial_evidence.union(result['evidence_image_ids']))
        turn_id = read(case_dir / 'status.json')['turn_id']
        seen = image_delivery_report(case_dir, registry, evidence, through_turn=turn_id)
        if seen['missing_image_ids']:
            write(progress, dict(stage='source_evidence_redelivery', updated_epoch=time()))
            decision_dir = case_dir / 'source_evidence_redelivery'
            missing = [registry[key] for key in seen['missing_image_ids']]
            transport.prepare_turn(thread, decision_dir,
                                   'Confirm or adjust the same brief SAM prompt plan after inspecting the directly supplied evidence. '
                                   'Return only the plan. Previously submitted plan: ' + str(result),
                                   missing, dict(metadata, evidence_redelivery=True, auxiliary_checks_remaining=
                                                 MAX_AUXILIARY_CHECKS - context['activity']['auxiliary_checks']), schema,
                                   prior_images=[item for key, item in registry.items()
                                                 if key not in seen['missing_image_ids']])
            result, registry = session.run_turn(decision_dir, context, timeout_seconds=1200)
            validation = read(decision_dir / 'final_validation.json')
        if validation['disposition'] == 'rejected_image':
            return finalize_annotation(context, selection_path, result, registry, model, validation)
        evidence = sorted(initial_evidence.union(result['evidence_image_ids']))
        turn_id = read(decision_dir / 'status.json')['turn_id']
        verify_images(case_dir, case_dir, registry, evidence, through_turn=turn_id)
        sol_seconds = round(monotonic() - sol_started, 2)
        result = annotation_plan(result)
        units = result['regions'] + result.get('direct_regions', [])
        background = result.get('background') or {}
        proposed_cashew = (result['whole_class'] == 2 or background.get('class_id') == 2
                           or any(unit['class_id'] == 2 for unit in units))
        search_for_dry = not fixed_scene and not dry_search_exhausted and (proposed_cashew or result.get('needs_dry_season'))
        comparison_id = comparison['image_id'] if comparison else None
        export_plan, withheld = cashew_plan_for_export(
            result, dry_scene=fixed_scene,
            comparison_image_id=comparison_id if dry_search_exhausted else None)
        gate = dict(confirmed_local_dry_season=fixed_scene, dry_comparison_requested=bool(search_for_dry),
                    comparison_capture_dates=comparison['source_date'] if comparison_id in result['evidence_image_ids'] else [],
                    withheld_cashew_units=withheld, export_plan=export_plan)
        write(case_dir / 'CASHEW_EVIDENCE.json', gate)
        review_seconds = 0
        post_sam_model_calls = 0
        if 'box_proposals' in context:
            write(progress, dict(stage='matching_and_keep_review', updated_epoch=time(), sol_seconds=sol_seconds))
            annotation_id, registry, turn_id, review_seconds = review_box_proposals(
                context, export_plan, session, registry, evidence, turn_id, search_for_dry=search_for_dry)
            sam_seconds = context['box_proposals']['seconds']
            post_sam_model_calls = 1 + int(decision_dir != case_dir) + int(bool(review_seconds))
        else:
            deferred = bool(search_for_dry and defer_for_dry)
            usage_path = decision_dir / 'token_usage.json'
            usage = read(usage_path).get('tokenUsage', {}).get('total', {}) if usage_path.is_file() else {}
            local_plan = dict(status='dry_deferred' if deferred else 'ready', export_plan=export_plan,
                              result=result, registry=registry, evidence=evidence, turn_id=turn_id,
                              selection=str(selection_path), selection_sha256=digest(selection_path),
                              model=model, service_tier=session.service_tier,
                              auxiliary_checks=context['activity']['auxiliary_checks'],
                              token_usage=usage, hosted_annotation_turns=1 + int(decision_dir != case_dir),
                              dry_comparison_requested=bool(search_for_dry),
                              fixed_scene=fixed_scene, dry_search_exhausted=dry_search_exhausted,
                              preparation_seconds=preparation_seconds, sol_seconds=sol_seconds,
                              elapsed_before_local=round(monotonic() - started, 2))
            if woody_review:
                local_plan['woody_review'] = dict(comparison=comparison)
            write(case_dir / 'LOCAL_PLAN.json', local_plan)
            if deferred:
                write(progress, dict(stage='waiting_for_dry_candidate', updated_epoch=time(), sol_seconds=sol_seconds))
                return dict(status='needs_dry_image', selection=str(selection_path),
                            auxiliary_checks_used=local_plan['auxiliary_checks'])
    finally:
        session.close()
    if 'box_proposals' not in context:
        return publish_local_plan(context, local_plan)
    validation = validate_final(dict(context, registered_frames=registry), {'annotation_id': annotation_id})
    validation['decision_turn_id'] = turn_id
    write(case_dir / 'PUBLICATION_VALIDATION.json', validation)
    write(progress, dict(stage='export', updated_epoch=time(), sol_seconds=sol_seconds, sam_seconds=sam_seconds))
    receipt = finalize_annotation(context, selection_path, result, registry, model, validation)
    timing = dict(preparation_seconds=preparation_seconds, sol_seconds=round(sol_seconds + review_seconds, 2),
                  keep_review_seconds=review_seconds, sam_seconds=round(sam_seconds, 2),
                  elapsed_seconds=round(monotonic() - started, 2), post_sam_model_calls=post_sam_model_calls)
    write(progress, dict(stage='completed', updated_epoch=time(), **timing))
    return dict(receipt, **timing, dry_comparison_requested=bool(search_for_dry),
                confirmed_local_dry_season=fixed_scene, dry_search_exhausted=dry_search_exhausted,
                cashew_evidence_path=str(case_dir / 'CASHEW_EVIDENCE.json'))
