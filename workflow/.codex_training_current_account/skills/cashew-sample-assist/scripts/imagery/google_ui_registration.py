"""Calibrate a real Google Earth screenshot from visible mouse coordinates.

The four corner controls fit a projective map. Five separately observed controls
test it before an image can become a SAM source. No target labels are read.
"""
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
from affine import Affine
from PIL import Image
from pyproj import Transformer


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def coordinates(text):
    # DMS values are read from the actual visible coordinate status, not URLs.
    text = text.replace('º', '°').replace('’', "'").replace('′', "'")
    text = text.replace('”', '"').replace('″', '"').replace('O', '0')
    # The small visible minute mark can be OCR'd as a double quote or star;
    # the numerical D/M/S values are left unchanged and spatial checks remain.
    matches = re.findall(r'(\d{1,3})\s*[°o]\s*(\d{1,2})\s*[\'’′"*°]?\s*'
                         r'(\d{1,2}(?:\.\d+)?)\s*["”″*]*\s*([NSEW])', text, re.I)
    parsed = {}
    for d, m, s, direction in matches:
        d, m, s = int(d), int(m), float(s)
        if m >= 60 or s >= 60:
            continue
        direction = direction.upper()
        value = (d + m / 60 + s / 3600) * (-1 if direction in 'SW' else 1)
        parsed['lat' if direction in 'NS' else 'lon'] = value
    if set(parsed) != {'lon', 'lat'}:
        raise ValueError('No complete visible latitude and longitude: ' + text)
    return [parsed['lon'], parsed['lat']]


def polynomial_basis(xy):
    x, y = np.asarray(xy).T
    return np.column_stack([np.ones(len(x)), x, y, x*x, x*y, y*y])


def capture_grid(page, out, width, height, read_ocr):
    out = Path(out)
    points = [(350, 350), (width - 350, 350),
              (350, height - 250), (width - 350, height - 250),
              (width // 2, height // 2), (width // 2, 350),
              (350, height // 2), (width - 350, height // 2),
              (width // 2, height - 250)]
    rows = []

    def observe(index, point):
        seen = {tuple(row['lon_lat']) for row in rows}
        # A map tooltip or delayed frame can leave the previous coordinate in
        # the footer. Move to a nearby visible point and read a new control.
        for attempt, (dx, dy) in enumerate(((0, 0), (0, 36), (50, 120))):
            x, y = point[0] + dx, point[1] + dy
            page.mouse.move(x, y)
            page.wait_for_timeout(550 if attempt == 0 else 950)
            suffix = '' if attempt == 0 else f'_retry{attempt}'
            path = out / f'coordinate_status_{index:02d}{suffix}.png'
            page.screenshot(path=str(path), clip={'x': 750, 'y': height - 55,
                                                 'width': width - 750, 'height': 55})
            text = ' '.join(row['text'] for row in read_ocr(path))
            try:
                lon_lat = coordinates(text)
            except ValueError:
                continue
            if tuple(lon_lat) not in seen:
                return {'screen_pixel': [x, y], 'lon_lat': lon_lat, 'visible_status_text': text,
                        'proof_path': str(path), 'proof_sha256': sha(path)}
        raise ValueError(f'Visible footer did not report a fresh control at {point}: {text}')

    for i, point in enumerate(points):
        rows.append(observe(i, point))
    source = np.array([r['screen_pixel'] for r in rows], dtype='float64')
    transformer = Transformer.from_crs(4326, 3857, always_xy=True)
    ground = np.array([transformer.transform(*r['lon_lat']) for r in rows])
    design = np.c_[source, np.ones(len(source))]
    # Fit only the first four controls; all subsequent points are independent.
    coeff = np.linalg.lstsq(design[:4], ground[:4], rcond=None)[0]
    affine_residual = np.linalg.norm(design @ coeff - ground, axis=1)
    # Earth's screen is projective. Fit only the same four corners, leaving
    # the five other observed coordinates unused until the accuracy check.
    origin = ground[:4].mean(axis=0)
    relative_h, _ = cv2.findHomography(source[:4], ground[:4] - origin, method=0)
    translation = np.array([[1., 0., origin[0]], [0., 1., origin[1]], [0., 0., 1.]])
    homography = translation @ relative_h
    projected = design @ homography.T
    projected = projected[:, :2] / projected[:, 2:]
    residual = np.linalg.norm(projected - ground, axis=1)
    pixel_scale = float(np.sqrt(abs(np.linalg.det(coeff[:2]))))
    supported = bool(pixel_scale > 0 and max(residual) <= 4.
                     and np.median(residual[4:]) <= 2.)
    # Screen samples are measured in CSS image edge coordinates.
    matrix = Affine(coeff[0, 0], coeff[1, 0], coeff[2, 0],
                    coeff[0, 1], coeff[1, 1], coeff[2, 1])
    receipt = {'method': 'Actual visible mouse-coordinate footer; four-control projective map and five held-out checks',
               'screen_to_map_verified': supported, 'source_crs': 'EPSG:3857',
               'screen_to_mercator_homography': homography.tolist(),
               'affine_comparison_transform': list(matrix)[:6],
               'affine_comparison_residuals_mercator_m': affine_residual.tolist(),
               'screen_coordinate_convention': 'CSS pixel edge coordinates; OpenCV sampling subtracts0.5',
               'residuals_mercator_m': residual.tolist(),
               'heldout_max_error_mercator_m': float(max(residual[4:])),
               'heldout_median_error_mercator_m': float(np.median(residual[4:])),
               'display_pixel_size_mercator_m': pixel_scale,
               'ground_imagery_geolocation_accuracy_unknown': True,
               'target_labels_read': False, 'control_points': rows}
    if not supported:
        # A local second-order map accounts for terrain-dependent curvature.
        # Its nine fit points and four newly observed checks remain separate.
        checks = [(575, 550), (1025, 550), (575, 950), (1025, 950)]
        for i, point in enumerate(checks, 9):
            rows.append(observe(i, point))
        source = np.array([r['screen_pixel'] for r in rows], dtype='float64')
        ground = np.array([transformer.transform(*r['lon_lat']) for r in rows])
        screen_origin = np.array([800., 750.])
        screen_scale = np.array([450., 400.])
        ground_origin = ground[:9].mean(axis=0)
        forward = np.linalg.lstsq(polynomial_basis((source[:9]-screen_origin)/screen_scale),
                                 ground[:9]-ground_origin, rcond=None)[0]
        inverse = np.linalg.lstsq(polynomial_basis((ground[:9]-ground_origin)/1000.),
                                 source[:9], rcond=None)[0]
        predicted = polynomial_basis((source-screen_origin)/screen_scale) @ forward + ground_origin
        forward_error = np.linalg.norm(predicted-ground, axis=1)
        inverse_predicted = polynomial_basis((ground-ground_origin)/1000.) @ inverse
        inverse_error = np.linalg.norm(inverse_predicted-source, axis=1)*pixel_scale
        check_error = np.maximum(forward_error[9:], inverse_error[9:])
        supported = bool(max(check_error) <= 4. and np.median(check_error) <= 2.)
        receipt.update(method='Visible coordinates; nine-point local quadratic map and four new held-out checks',
            screen_to_map_verified=supported, mapping_type='quadratic_bidirectional',
            projective_heldout_errors_mercator_m=residual[4:].tolist(),
            quadratic_screen_origin=screen_origin.tolist(), quadratic_screen_scale=screen_scale.tolist(),
            quadratic_ground_origin=ground_origin.tolist(), quadratic_ground_scale=1000.,
            quadratic_forward_coefficients=forward.tolist(), quadratic_inverse_coefficients=inverse.tolist(),
            fit_control_indices=list(range(9)), heldout_control_indices=list(range(9,13)),
            forward_residuals_mercator_m=forward_error.tolist(),
            inverse_sampling_errors_mercator_m=inverse_error.tolist(),
            heldout_max_error_mercator_m=float(max(check_error)),
            heldout_median_error_mercator_m=float(np.median(check_error)), control_points=rows)
    (out / 'coordinate_calibration.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    if not supported:
        raise ValueError('Visible coordinate calibration failed held-out checks')
    return receipt


def aligned_frame(req, metadata, image, calibration, out):
    candidate = req['candidate']
    out = Path(out)
    image = Path(image)
    rgb = np.array(Image.open(image).convert('RGB'))
    native_transform = Affine(*candidate['native_transform'])
    size = 1024
    target_transform = native_transform * Affine.scale(128 / size)
    if calibration.get('screen_to_mercator_homography'):
        homography = np.array(calibration['screen_to_mercator_homography'])
    else:
        # Preserve the first executed affine canary without pretending it used
        # the later projective method.
        affine = Affine(*calibration['screenshot_transform'])
        homography = np.array([[affine.a, affine.b, affine.c],
                               [affine.d, affine.e, affine.f], [0., 0., 1.]])
    rows, cols = np.indices((size, size), dtype='float64')
    world_x = target_transform.a * (cols + .5) + target_transform.b * (rows + .5) + target_transform.c
    world_y = target_transform.d * (cols + .5) + target_transform.e * (rows + .5) + target_transform.f
    transformer = Transformer.from_crs(candidate['native_crs'], 'EPSG:3857', always_xy=True)
    mercator_x, mercator_y = transformer.transform(world_x, world_y)
    world = np.stack([mercator_x.ravel(), mercator_y.ravel(), np.ones(size**2)])
    if calibration.get('mapping_type') == 'quadratic_bidirectional':
        ground = np.column_stack([mercator_x.ravel(), mercator_y.ravel()])
        normalized = (ground-np.array(calibration['quadratic_ground_origin']))/calibration['quadratic_ground_scale']
        mapped = polynomial_basis(normalized) @ np.array(calibration['quadratic_inverse_coefficients'])
        map_x = (mapped[:, 0]-.5).reshape(size,size).astype('float32')
        map_y = (mapped[:, 1]-.5).reshape(size,size).astype('float32')
    else:
        mapped = np.linalg.inv(homography) @ world
        map_x = (mapped[0] / mapped[2] - .5).reshape(size, size).astype('float32')
        map_y = (mapped[1] / mapped[2] - .5).reshape(size, size).astype('float32')
    aligned = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    visible = np.ones(rgb.shape[:2], dtype='uint8') * 255
    visible[:200, :] = 0
    visible[-100:, :] = 0
    visible[:, :90] = 0
    visible[-350:, -90:] = 0
    visible_on_grid = cv2.remap(visible, map_x, map_y, cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT)
    support = visible_on_grid.reshape(128, 8, 128, 8).min(axis=(1, 3))
    path = out / 'Google_registered_AOI1024.png'
    Image.fromarray(aligned).save(path)
    valid = out / 'Google_image_support128.png'
    Image.fromarray(support).save(valid)
    frame = {'candidate_id': candidate['candidate_id'],
        'frame_id': candidate['candidate_id'] + '_Google_UI_registered',
        'provider': 'Google', 'path': str(path), 'image_path': str(path),
        'sha256': sha(path), 'width': size, 'height': size,
        'image_crs': candidate['native_crs'], 'image_transform': list(target_transform)[:6],
        'native_crs': candidate['native_crs'], 'native_transform': candidate['native_transform'],
        'bounds_wgs84': candidate['bounds_wgs84'],
        'source_years': metadata['source_years'], 'source_date': metadata['source_date_raw'],
        'date_unknown_or_mixed': metadata['date_unknown_or_mixed'],
        'native128_valid_mask_path': str(valid), 'native128_valid_mask_sha256': sha(valid),
        'native128_valid_fraction': float((support > 0).mean()),
        'frame_georeferencing_verified': True,
        'grid_role': candidate.get('grid_role', 'actual_AEF_native128'),
        'same_exact_native128_AOI': candidate.get('grid_role', 'actual_AEF_native128') == 'actual_AEF_native128',
        'annotation_allowed': False, 'quality_review_required': True,
        'attribution': metadata['attribution_text_visible'],
        'registration_receipt_path': str(out / 'coordinate_calibration.json'),
        'registration_receipt_sha256': sha(out / 'coordinate_calibration.json'),
        'raw_capture_sha256': sha(image)}
    (out / 'frame_pending_quality_review.json').write_text(json.dumps(frame, ensure_ascii=False, indent=2) + '\n')
    return frame
