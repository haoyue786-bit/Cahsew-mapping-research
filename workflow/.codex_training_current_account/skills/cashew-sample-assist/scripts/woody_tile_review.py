"""Independently review woody land units once before requesting their final masks."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFont

from _runtime import transport
from _runtime.common import SKILL_ROOT, digest, read, write
from _runtime.vision_audit import verify_images
from annotation_geometry import MAX_SEGMENT_UNITS, SIZE, record_image
from reference_lookup import intersects

GUIDE = SKILL_ROOT / 'references/woody-tile-review.md'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
WOODY = (1, 2)


def plan_hash(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def bounds_for(unit):
    if unit.get('box') is not None:
        x1, y1, x2, y2 = unit['box']
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    if unit.get('polygon'):
        x, y = zip(*unit['polygon'])
        return [min(x), min(y), max(x), max(y)]
    x, y = zip(*unit['positive_points'])
    return [max(0, min(x)-48), max(0, min(y)-48), min(SIZE, max(x)+48), min(SIZE, max(y)+48)]


def overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def woody_units(plan):
    rows = {}
    if plan['whole_class'] in WOODY:
        rows['whole'] = ('whole_class', 0, plan['whole_class'], [0, 0, SIZE, SIZE], [])
    background = plan.get('background')
    if background and background['class_id'] in WOODY:
        rows['background'] = ('background', 0, background['class_id'], [0, 0, SIZE, SIZE], [])
    for key, prefix in (('regions', 'sam'), ('direct_regions', 'direct')):
        for index, unit in enumerate(plan.get(key, [])):
            if unit['class_id'] in WOODY:
                rows[f'{prefix}_{index+1:02d}'] = (key, index, unit['class_id'], bounds_for(unit),
                                                  unit.get('positive_points', []))
    return rows


def outline_view(context, selected, target):
    with Image.open(context['scene']) as original:
        rgb = original.convert('RGB')
    painter = ImageDraw.Draw(rgb)
    font = ImageFont.truetype(FONT, 19)
    for unit, (_, _, _, (x, y, x2, y2), marks) in selected.items():
        painter.rectangle((x, y, min(x2, SIZE-1), min(y2, SIZE-1)), outline='#ffffff', width=2)
        for mx, my in marks:
            painter.ellipse((mx-8, my-8, mx+8, my+8), outline='#111111', width=4)
            painter.ellipse((mx-8, my-8, mx+8, my+8), outline='white', width=2)
        label_x, label_y = min(x+3, SIZE-122), min(y+2, SIZE-27)
        painter.text((label_x, label_y), unit, font=font, fill='white', stroke_width=3, stroke_fill='#111111')
    rgb.save(target, format='PNG')


def merge_plan(plan, selected, decision, comparison_id):
    review = {row['id']: row['class_id'] for row in decision['units']}
    two_classes = plan.get('background') is not None
    dropped = {key for key, (_, _, code, _, _) in selected.items() if review[key] != code}
    processed = deepcopy(plan)
    if 'whole' in dropped:
        processed['whole_class'] = None
        processed['whole_cashew_evidence'] = None
    if 'background' in dropped:
        processed['background'] = None
    if two_classes:
        processed['withheld_foreground_boxes'] = plan.get('withheld_foreground_boxes', []) + [
            selected[key][3] for key in sorted(dropped) if selected[key][0] == 'regions']
    for key, prefix in (('regions','sam'), ('direct_regions','direct')):
        processed[key] = [item for i,item in enumerate(plan.get(key, []), 1) if f'{prefix}_{i:02d}' not in dropped]
    added = 0
    # The host accepts newly supported interiors only in previously unassigned ground.
    if plan['whole_class'] is None and not two_classes and comparison_id in decision['evidence_image_ids']:
        reserved = [bounds_for(row) for key in ('regions','direct_regions') for row in plan.get(key, [])]
        for item in decision['additions']:
            x, y, x2, y2 = item['box']
            if (len(processed['regions']) >= MAX_SEGMENT_UNITS or x2 <= x or y2 <= y
                    or any(overlaps(item['box'], b) for b in reserved)
                    or not all(x <= mx < x2 and y <= my < y2 for mx, my in item['positive_points'])
                    or not item['cashew_evidence'].strip()):
                continue
            reserved.append(item['box'])
            processed['regions'].append(item)
            added += 1
    return processed, sorted(dropped), added


def contrast_atlas(context):
    directory = SKILL_ROOT/'assets/coordinate-reference-library-compact'
    atlas = read(directory/'WOODY_CONTRASTS.json')
    excluded = [context['bounds'], *context.get('excluded_reference_bounds', ())]
    if any(intersects(anchor['bounds_wgs84'], box) for anchor in atlas['anchors'] for box in excluded):
        return None
    return record_image(directory/atlas['image'], 'woody_contrast_examples', 'Cashew and lookalike references',
                        sha256=atlas['sha256'])


def run(context, plan):
    from annotate import obj, array, REGION, STRING

    original = plan['export_plan']
    comparison = plan['woody_review'].get('comparison')
    selected = woody_units(original)
    if not selected and (not comparison or original['whole_class'] is not None or original.get('background') is not None):
        return original, dict(status='not_needed', seconds=0, reviewed=0, added=0, withheld=0)
    root = context['case_dir']
    target = root / 'woody_review'
    receipt = target / 'RECEIPT.json'
    signature = plan_hash(original)
    if receipt.is_file():
        saved = read(receipt)
        if saved['before_sha256'] != signature or read(Path(saved['vision_audit']))['status'] != 'verified':
            raise ValueError('The saved woodland review differs from its audited plan')
        return saved['final_plan'], saved

    before = monotonic()
    target.mkdir(parents=True, exist_ok=True)
    marked_path = target / 'numbered_woody_units.png'
    inputs = [record_image(context['scene'], 'working_RGB', 'original_working_RGB',
                           source_date=context['frame']['acquisition_dates'])]
    if selected:
        if not marked_path.is_file():
            outline_view(context, selected, marked_path)
        inputs.append(record_image(marked_path, 'numbered_woody_units', 'numbered_unit_footprints'))
    if comparison:
        inputs.append(comparison)
    atlas = contrast_atlas(context)
    if atlas:
        inputs.append(atlas)
    comparison_id = comparison['image_id'] if comparison else None
    labels = {'type':'integer','enum':[0,1,2,3,10]}
    units = obj(id={'type':'string','enum':list(selected) or ['none']}, class_id=labels, visible_cue=STRING)
    addition_fields = dict(REGION['properties'], class_id={'type':'integer','enum':list(WOODY)},
                           cashew_evidence={'type':'string','minLength':1})
    addition_fields['box'] = REGION['properties']['box']['anyOf'][0]
    can_add = comparison and original['whole_class'] is None and original.get('background') is None
    additions = array(obj(**addition_fields), 4 if can_add else 0)
    result_schema = obj(units=array(units,len(selected),len(selected)), additions=additions,
                        evidence_image_ids=array(STRING,len(inputs)))
    metadata = dict(case_id=context['case'], country=context['grid']['country'],
                    label_period=context['grid']['label_period'], coordinate_space='original 1024x1024',
                    units=[dict(id=k, box=v[3], interior_marks=v[4]) for k,v in selected.items()],
                    prior_unit_footprints=[bounds_for(row) for key in ('regions','direct_regions')
                                           for row in original.get(key, [])],
                    uncertain_polygons=original.get('uncertain_polygons', []),
                    alternate_image_id=comparison_id)
    prompt = GUIDE.read_text() + (
        '\n\nClassify each numbered unit independently and return every ID once. Use the existing four class IDs '
        'and class 10. Report the observable cue that supports each decision. '
        'For additions give a tight box outside the supplied prior footprints and a brief cue '
        'from the working image and alternate. Cite the alternate only when it corroborates those additions.'
    )

    def validate(state, result):
        ids = [row['id'] for row in result['units']]
        return dict(eligible=len(ids)==len(selected) and set(ids)==set(selected)
                    and set(result['evidence_image_ids']) <= state['registered_frames'].keys())

    state = target/'status.json'
    if not state.is_file():
        thread = transport.prepare_thread('Independently review land cover in the selected dated tile.', [])
        transport.prepare_turn(thread,target,prompt,inputs,metadata,result_schema)
        session = transport.Session(target,thread,SimpleNamespace(dispatch_tool=None,validate_final=validate))
        try:
            session.start()
            if session.service_tier != plan['service_tier']:
                raise ValueError('Woodland review must use the first decision service tier')
            session.run_turn(target,{},timeout_seconds=1200)
        finally:
            session.close()
    saved = read(state)
    if saved['status'] != 'completed':
        raise RuntimeError('The woodland review turn is unresolved')
    registry = read(target/'image_registry.json')
    output = target/'result.json'
    if digest(output) != saved['result_sha256']:
        raise ValueError('The saved woodland review output changed')
    decision = read(output)
    verify_images(target,target,registry,{row['image_id'] for row in inputs},through_turn=saved['turn_id'])
    final, dropped, added = merge_plan(original,selected,decision,comparison_id)
    elapsed = max(monotonic()-before,saved['completed_epoch']-saved['prepared_epoch'])
    summary = dict(status='verified', seconds=round(elapsed,2), reviewed=len(selected), added=added,
                   withheld=len(dropped), withheld_ids=dropped, before_sha256=signature, final_plan=final,
                   comparison_date=comparison['source_date'] if comparison else None,
                   vision_audit=str(target/'VISION_DELIVERY_AUDIT.json'))
    write(receipt,summary)
    return final,summary
