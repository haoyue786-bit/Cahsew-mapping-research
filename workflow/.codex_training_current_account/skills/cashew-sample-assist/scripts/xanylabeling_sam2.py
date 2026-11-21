"""Send an optional whole-image prompt to the actual X-AnyLabeling SAM2 service."""

import base64
import hashlib
import shutil
from time import monotonic

import numpy as np
import requests
from PIL import Image, ImageDraw

from _runtime.common import SKILL_ROOT, digest, read, write
from annotation_geometry import SIZE, cover_image, next_number, record_image, require_evidence, sam_unit_folder


LIBRARY = SKILL_ROOT.parents[2] / 'outputs/cashew_reference_library_v2_20261117'
BOX_BATCH_SIZE = 24


def segment_boxes(boxes, context):
    """Send each detector box once with the same full-scene image embedding."""
    started = monotonic()
    directory = context['case_dir'] / 'proposal_masks'
    directory.mkdir(exist_ok=True)
    encoded = base64.b64encode(context['scene'].read_bytes()).decode('ascii')
    with Image.open(context['scene']) as image:
        expected_rgb = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
    case_id = context['backend_case']
    records, batches = [], []
    with requests.Session() as client:
        client.trust_env = False
        for offset in range(0, len(boxes), BOX_BATCH_SIZE):
            units = boxes[offset:offset + BOX_BATCH_SIZE]
            batch_id = f'boxes_{len(batches) + 1:03}'
            marks = [dict(type='rectangle', data=row['box']) for row in units]
            base = LIBRARY / 'runtime/raw_sam2' / case_id / batch_id
            last = next_number(base) - 1
            raw = base / str(last)
            prior = (raw / 'PREDICTOR_RECEIPT.json').is_file() and all(
                (raw / f'batch{index}_mask0.png').is_file() for index in range(len(units)))
            if not prior:
                last += 1
                raw = base / str(last)
                params = dict(image_id=case_id, record_case_id=case_id, record_unit_id=batch_id,
                              record_revision=str(last), marks=marks, multimask_output=False,
                              show_masks=False, show_boxes=False)
                payload = dict(model='segment_anything_2', image='data:image/png;base64,' + encoded, params=params)
                context['activity']['sam_requests'] += 1
                response = client.post('http://127.0.0.1:19574/v1/predict', json=payload, timeout=180)
                response.raise_for_status()
                if not response.json().get('success'):
                    raise RuntimeError('X-AnyLabeling SAM2 box batch failed')
            audit_path = raw / 'PREDICTOR_RECEIPT.json'
            audit = read(audit_path)
            scores = np.asarray(audit['predicted_geometry_IoU']).ravel().tolist()
            if (audit['RGB_pixel_sha256'] != expected_rgb or audit['image_shape'] != [SIZE, SIZE, 3]
                    or audit['multimask_output'] or len(scores) != len(units)
                    or np.asarray(audit['box']).reshape(-1, 4).tolist() != [unit['box'] for unit in units]):
                raise ValueError('Expected one SAM2 mask for each supplied box on the full RGB')
            for index, (unit, score) in enumerate(zip(units, scores)):
                path = directory / (unit['id'] + '.png')
                source = raw / f'batch{index}_mask0.png'
                with Image.open(source) as mask:
                    if mask.size != (SIZE, SIZE):
                        raise ValueError('The SAM2 mask must use the source RGB grid')
                    if mask.getbbox() is None:
                        continue
                shutil.copy2(source, path)
                records.append(dict(unit, mask_path=str(path), geometry_iou=score))
            batches.append(dict(boxes=len(units), predictor_receipt=str(audit_path),
                                embedding_cache_hit=audit['embedding_cache_hit']))
    report = dict(box_count=len(boxes), mask_count=len(records), masks=records, batches=batches,
                  seconds=round(monotonic() - started, 2))
    write(directory / 'SAM_PROPOSALS.json', report)
    return report


def recover_mask(unit, base, expected_rgb):
    revision = next_number(base) - 1
    folder = base / str(revision)
    receipt_path = folder / 'receipt.json'
    mask = folder / 'mask0.png'
    if not receipt_path.is_file() or not mask.is_file() or not (folder / 'prompt.json').is_file():
        return None
    try:
        receipt = read(receipt_path)
        if (read(folder / 'prompt.json') == unit and receipt.get('RGB_pixel_sha256') == expected_rgb
                and receipt.get('mask_sha256') == digest(mask)):
            return receipt
    except (OSError, ValueError):
        return None
    return None


def save_single_mask(unit, folder, revision, source, scores, expected_rgb):
    path = folder / 'mask0.png'
    shutil.copy2(source, path)
    saved = dict(unit, revision=revision, multimask_output=False, predicted_geometry_iou=scores,
                 RGB_pixel_sha256=expected_rgb, mask_sha256=digest(path))
    write(folder / 'receipt.json', saved)
    return saved


def segment(arguments, context, *, render_review=True):
    client = requests.Session()
    client.trust_env = False
    encoded = base64.b64encode(context['scene'].read_bytes()).decode('ascii')
    expected_rgb = hashlib.sha256(np.asarray(Image.open(context['scene']).convert('RGB')).tobytes()).hexdigest()
    rgb = Image.open(context['scene']).convert('RGB') if render_review else None
    records, images, recovered = [], [], {}
    units = arguments['units']
    for unit in units:
        require_evidence(unit, context['registered_frames'])
        box = unit['box']
        if box is not None and (box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError('Use an ordered XYXY prompt box')
    if not render_review:
        for unit in units:
            previous = recover_mask(unit, sam_unit_folder(context, unit['unit_id']), expected_rgb)
            if previous:
                recovered[unit['unit_id']] = previous
                context['activity']['reused_sam_masks'] = context['activity'].get('reused_sam_masks', 0) + 1
    for unit in units:
        unit_id = unit['unit_id']
        if unit_id in recovered:
            records.append(recovered[unit_id])
            continue
        base = sam_unit_folder(context, unit_id)
        revision = next_number(base)
        folder = base / str(revision)
        folder.mkdir(parents=True, exist_ok=False)
        box = unit['box']
        marks = [{'type': 'point', 'data': point, 'label': label}
                 for label, kind in [(1, 'positive_points'), (0, 'negative_points')] for point in unit[kind]]
        if box is not None:
            marks.append({'type': 'rectangle', 'data': box})
        backend_case = context['backend_case']
        payload = {'model': 'segment_anything_2', 'image': 'data:image/png;base64,' + encoded,
                   'params': {'image_id': backend_case, 'record_case_id': backend_case,
                              'record_unit_id': unit_id, 'record_revision': str(revision),
                              'marks': marks, 'multimask_output': render_review,
                              'show_masks': render_review, 'show_boxes': False}}
        write(folder / 'prompt.json', unit)
        context['activity']['sam_requests'] += 1
        response = client.post('http://127.0.0.1:19574/v1/predict', json=payload, timeout=180)
        response.raise_for_status()
        if not response.json().get('success'):
            raise RuntimeError('X-AnyLabeling SAM2 did not return masks')

        raw = LIBRARY / 'runtime/raw_sam2' / backend_case / unit_id / str(revision)
        receipt = read(raw / 'PREDICTOR_RECEIPT.json')
        if receipt['RGB_pixel_sha256'] != expected_rgb or receipt['image_shape'] != [SIZE, SIZE, 3]:
            raise ValueError('SAM2 did not receive the complete query RGB')
        scores = np.asarray(receipt['predicted_geometry_IoU']).ravel().tolist()
        if not render_review:
            if receipt['multimask_output'] or len(scores) != 1:
                raise ValueError('Expected one SAM2 mask per land unit')
            records.append(save_single_mask(unit, folder, revision, raw / 'batch0_mask0.png', scores, expected_rgb))
            continue
        overlay = Image.new('RGB', (1536, 552), 'white')
        draw = ImageDraw.Draw(overlay)
        for candidate in range(3):
            path = folder / f'mask{candidate}.png'
            shutil.copy2(raw / f'batch0_mask{candidate}.png', path)
            mask = np.asarray(Image.open(path)) > 0
            if mask.shape != (SIZE, SIZE):
                raise ValueError('Expected a mask in the complete scene geometry')
            plane = np.full(mask.shape, 10, dtype=np.uint8)
            plane[mask] = unit['class_id']
            tinted = Image.composite(Image.blend(rgb, cover_image(plane), .4), rgb, Image.fromarray(mask))
            overlay.paste(tinted.resize((512, 512), Image.Resampling.LANCZOS), (512 * candidate, 40))
            draw.text((512 * candidate + 8, 10),
                      f'{unit_id} r{revision} mask {candidate}  geometry IoU {scores[candidate]:.3f}',
                      fill='black')
        path = folder / 'candidates.png'
        overlay.save(path)
        image_id = f"{context['case']}_{unit_id}_r{revision}_sam"
        saved = dict(unit, revision=revision, candidate_image_id=image_id, predicted_geometry_iou=scores)
        write(folder / 'receipt.json', saved)
        records.append(saved)
        images.append(record_image(path, image_id, 'SAM_geometry_overlay_candidates'))
    return {'success': True, 'units': records, 'images': images}
