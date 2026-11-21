"""Normal public Google Earth navigation with visible date evidence and no label transform."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from urllib.parse import urlsplit

PROJECT = Path('@WORKSPACE@')
OLD = PROJECT / 'outputs/tgo_visual_negative_pilot_20261115'
SHARED = PROJECT / 'outputs/tgo_visual_annotation_v4_20261115'
sys.path[:0] = [str(SHARED / '_tools/python'), str(OLD / '_tools/python')]
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(OLD / '_tools/browsers')
from playwright.sync_api import sync_playwright
from PIL import Image
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from google_date_evidence import evidence, visible_dates


def newest_visible_marker(path):
    """Locate the rightmost available dot in a cropped light-theme timeline."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    red, green, blue = np.moveaxis(rgb[13:18], -1, 0)
    active = (red < 55) & (green < 132) & (blue >= 60) & (blue - red >= 43)
    columns = np.flatnonzero(active.sum(axis=0) >= 2)
    if not len(columns):
        return None
    last = np.split(columns, np.where(np.diff(columns) > 1)[0] + 1)[-1]
    return int(round(float(last.mean())))


def announcement_title_matches(text):
    """Google OCR may omit spaces in the known visible announcement title."""
    return 'newfromgoogleearth' in re.sub(r'\s+', '', text.lower())


def main(request_path):
    started = time.time()
    req = json.loads(Path(request_path).read_text())
    out = Path(req['output_dir']); out.mkdir(parents=True, exist_ok=True)
    width, height = 1600, 1400
    actions = []
    ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1,
                   det_limit_side_len=2000, det_limit_type='max', use_cls=False)

    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def record(action, **extra):
        actions.append({'action': action, 'elapsed_seconds': time.time() - started,
                        'utc': dt.datetime.now(dt.timezone.utc).isoformat(), **extra})
        write('browser_actions.json', actions)

    def screen(page, name, clip=None):
        path = out / name
        page.screenshot(path=str(path), **({'clip': clip} if clip else {}))
        return path

    def read_ocr(path):
        result, _ = ocr(str(path))
        return [{'box': row[0], 'text': row[1], 'confidence': float(row[2])} for row in (result or [])]

    def dismiss_tips(page, name):
        # Some map tips extend below the historical date strip and cover part
        # of the target RGB. Use the visible Dismiss action on those cards.
        path = screen(page, name+'.png', {'x': 0, 'y': 0, 'width': width, 'height': 470})
        dismissed = False
        for row in read_ocr(path):
            if row['text'].strip().lower() == 'dismiss':
                x, y = np.array(row['box']).mean(axis=0)
                page.mouse.click(float(x), float(y))
                record('dismiss_visible_map_tip', proof_image=str(path), click=[float(x), float(y)])
                dismissed = True
        return dismissed

    def challenge(page):
        body = page.locator('body').inner_text()
        if any(term in body.lower() for term in ['unusual traffic', 'verify you are human', 'captcha']):
            screen(page, 'visible_access_challenge.png')
            raise RuntimeError('Visible access challenge; no bypass attempted.')

    def wait_scene(page, name, timeout=45):
        deadline = time.time() + timeout
        previous = None
        while time.time() < deadline:
            challenge(page)
            path = screen(page, name)
            scene = np.asarray(Image.open(path).convert('RGB'))[230:height-100, :, :]
            deviation = float(scene.std())
            difference = None if previous is None else float(np.abs(scene.astype(float)-previous).mean())
            if deviation > 6 and difference is not None and difference < 1.5:
                record('scene_pixels_ready', std=deviation, successive_mean_difference=difference)
                return path
            previous = scene.astype(float)
            page.wait_for_timeout(800)
        raise RuntimeError('No stable textured public satellite view within content-readiness window')

    with sync_playwright() as p:
        proxy = os.environ.get('CASHEW_IMAGERY_PROXY')
        browser = p.chromium.launch(headless=True, proxy={'server': proxy} if proxy else None,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--use-gl=angle',
                  '--use-angle=swiftshader', '--enable-webgl'])
        page = browser.new_page(viewport={'width': width, 'height': height}, device_scale_factor=1, locale='en-US')
        network_failures = []
        def request_failed(request):
            u = urlsplit(request.url)
            path = u.path.split('/bpb=')[0]
            if 'token=' in path:
                path = path.split('&token=')[0]
            network_failures.append({'host': u.netloc, 'path': path, 'error': request.failure})
            write('network_failures.json', network_failures)
        page.on('requestfailed', request_failed)
        try:
            lon, lat = req['navigation']['center_wgs84']
            radius = float(req['navigation']['requested_radius_m'])
            camera = radius / (math.tan(math.radians(17.5)) * width / height) + 400
            url = f'https://earth.google.com/web/@{lat},{lon},0a,{camera:.2f}d,35y,0h,0t,0r'
            response = page.goto(url, wait_until='domcontentloaded', timeout=90000)
            record('open_official_viewer', url=url, http_status=response.status if response else None,
                   viewport=[width, height], camera_distance_parameter=camera)
            # Earth renders its application outside the main document's accessible
            # tree in this deployed viewer. Verify the real visible toolbar by OCR.
            deadline = time.time() + 90
            toolbar_ready = False
            while time.time() < deadline:
                top = screen(page, 'initial_toolbar.png', {'x': 0, 'y': 0, 'width': width, 'height': 230})
                rows = read_ocr(top)
                text = ' '.join(row['text'] for row in rows).lower()
                if 'search google earth' in text or ('file' in text and 'tools' in text and 'help' in text):
                    toolbar_ready = True
                    write('initial_toolbar_ocr.json', rows)
                    break
                page.wait_for_timeout(800)
            if not toolbar_ready:
                raise RuntimeError('Public Earth application toolbar did not become visible')
            challenge(page)
            announcement = screen(page, 'visible_announcement_check.png')
            announcement_rows = read_ocr(announcement)
            if any(announcement_title_matches(row['text']) for row in announcement_rows):
                # The actual screenshot shows this announcement's close cross at
                # this viewport-relative location. It is not an authentication UI.
                page.mouse.click(width/2+204, height/2-240)
                record('close_actual_visible_new_earth_announcement', click=[width/2+204,height/2-240],
                       proof_image=str(announcement))
                page.wait_for_timeout(800)
                after_close = screen(page, 'announcement_after_close.png')
                after_rows = read_ocr(after_close)
                write('announcement_after_close_ocr.json', after_rows)
                if any(announcement_title_matches(row['text']) for row in after_rows):
                    raise RuntimeError('Visible Google announcement remains after the ordinary close action')
                record('confirm_actual_announcement_closed', proof_image=str(after_close),
                       announcement_title_absent=True)
                for row in after_rows:
                    if row['text'].strip().lower() == 'dismiss':
                        xy = np.array(row['box']).mean(axis=0)
                        page.mouse.click(float(xy[0]), float(xy[1]))
                        record('dismiss_visible_catalog_tip', proof_image=str(after_close), click=xy.tolist())
                        page.wait_for_timeout(400)
            step = req.get('history_step')
            history = None
            if step is not None:
                dismiss_tips(page, 'before_history_tips')
                # Validate the actual history title and date after the click.
                # Hover tooltip text can be obscured by map labels.
                page.mouse.click(542, 58)
                record('click_historical_imagery_toolbar')
                page.wait_for_timeout(850)
                dismiss_tips(page, 'history_visible_tips')
                strip = {'x': 145, 'y': 90, 'width': 350, 'height': 65}

                def selected_date(name, previous=None):
                    deadline = time.time() + 15
                    while time.time() < deadline:
                        toolbar = screen(page, name+'.png', strip)
                        rows = read_ocr(toolbar)
                        label = ' '.join(row['text'] for row in rows)
                        dates = visible_dates(label)
                        if len(dates) == 1 and dates[0] != previous:
                            write(name+'_ocr.json', rows)
                            return dates[0]
                        page.wait_for_timeout(600)
                    return None

                chosen = selected_date('history_open')
                if chosen is None:
                    raise RuntimeError('Historical toolbar did not expose a readable date')
                # The initial selection may still be a global placeholder
                # (Jan 1, 2022). Find the actual available timeline markers
                # after rendering the local view and select the rightmost one.
                wait_scene(page, 'history_initial_view.png')
                dismiss_tips(page, 'history_loaded_tips')
                timeline = screen(page, 'history_available_timeline.png',
                                  {'x': 500, 'y': 105, 'width': 920, 'height': 40})
                marker = newest_visible_marker(timeline)
                if marker is None:
                    raise RuntimeError('The loaded local timeline has no readable available markers')
                page.mouse.click(500 + marker, 120)
                record('select_rightmost_available_timeline_marker', initial_selection=chosen,
                       click=[500 + marker, 120], proof_image=str(timeline))
                wait_scene(page, 'history_latest_view.png')
                newest = selected_date('history_latest')
                if newest is None or newest < chosen:
                    raise RuntimeError('Latest historical date was not verified after timeline navigation')
                chosen = newest
                for index in range(step):
                    if chosen < req['date_floor']:
                        break
                    page.mouse.click(300, 120)
                    previous = selected_date(f'history_previous_{index+1}', previous=chosen)
                    if previous is None or previous >= chosen:
                        write('history_end.json', {'last_verified_selected_date': chosen,
                                                   'reason': 'Previous did not expose an earlier date'})
                        return
                    record('visible_earlier_date', before=chosen, after=previous)
                    chosen = previous
                if chosen < req['date_floor']:
                    write('history_end.json', {'last_verified_selected_date': chosen, 'reason': 'Before date floor'})
                    return
                history = {'selected_date': chosen, 'history_step': step,
                           'basis': 'Visible date after backward timeline navigation'}
            raw = wait_scene(page, 'original_browser_view.png')
            if history is not None:
                rendered = selected_date('history_rendered')
                if rendered != chosen:
                    raise RuntimeError('Rendered history date changed after imagery readiness')
                record('verify_final_rendered_historical_date', **history)
            # The observed Google annual-survey card can cover the coordinate footer.
            # Close its visible X; never choose an answer or submit the survey.
            survey_proof = screen(page, 'survey_overlay_check.png')
            survey_text = ' '.join(r['text'] for r in read_ocr(survey_proof)).lower()
            if 'survey' in survey_text and ('roadmap' in survey_text or 'by continuing' in survey_text):
                page.mouse.click(width-52, height-300)
                record('close_observed_annual_survey_X', click=[width-52,height-300],
                       proof_image=str(survey_proof), survey_answered_or_submitted=False)
                page.wait_for_timeout(500)
                raw = wait_scene(page, 'original_browser_view.png')
            if dismiss_tips(page, 'tips_before_calibration'):
                raw = wait_scene(page, 'original_browser_view.png')
            from google_ui_registration import capture_grid, aligned_frame
            try:
                calibration = capture_grid(page, out, width, height, read_ocr)
            except ValueError as error:
                if not any(word in str(error).lower() for word in ['by continuing', 'survey']):
                    raise
                proof = screen(page, 'late_survey_overlay.png')
                text = ' '.join(r['text'] for r in read_ocr(proof)).lower()
                if 'survey' not in text or not ('roadmap' in text or 'by continuing' in text):
                    raise
                page.mouse.click(width-52, height-300)
                record('close_late_observed_survey_X', click=[width-52,height-300],
                       proof_image=str(proof), survey_answered_or_submitted=False)
                raw = wait_scene(page, 'original_browser_view.png')
                calibration = capture_grid(page, out, width, height, read_ocr)

            page.mouse.move(width//2, height//2)
            page.wait_for_timeout(500)
            footer = screen(page, 'date_and_scale_status.png', {'x': 0, 'y': height-70, 'width': width, 'height': 70})
            footer_ocr = read_ocr(footer)
            # Ordinary visible credits control, matching the existing public viewer workflow.
            page.mouse.click(180, height-13)
            page.wait_for_timeout(350)
            popup = screen(page, 'official_attribution_popup.png',
                           {'x': width/2-350, 'y': height/2-350, 'width': 700, 'height': 700})
            popup_ocr = read_ocr(popup)
            write('attribution_ocr.json', popup_ocr)
            write('footer_ocr.json', footer_ocr)
            text = '\n'.join(row['text'] for row in popup_ocr)
            footer_text = '\n'.join(row['text'] for row in footer_ocr)
            dates = evidence(text, footer_text, history['selected_date'] if history else None)
            write('ACQUISITION_EVIDENCE.json', dates)
            acquisition_dates = dates['acquisition_dates']
            range_verified = dates['acquisition_range_verified']
            acquisition_years = sorted({int(date[:4]) for date in acquisition_dates})
            image = out / 'context_rgb.jpg'
            full = Image.open(raw).convert('RGB'); full.save(image, quality=94, subsampling=0)
            model_image = out / 'context_model128.png'
            full.crop((0, 230, width, height-100)).resize((128, 128), Image.Resampling.BOX).save(model_image)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            frame = f'context_Google_{req["candidate_id"]}_{digest[:16]}'
            record('capture_actual_context_and_public_date', final_url=page.url,
                   acquisition_dates=acquisition_dates, historical_selection=history,
                   source_years=acquisition_years, image_sha256=digest)
            metadata = {'status': 'available', 'candidate_id': req['candidate_id'], 'provider': 'Google',
                'view': req.get('view', 'context'), 'frame_id': frame, 'annotation_allowed': False,
                'image_crs': None, 'image_transform': None, 'frame_georeferencing_verified': False,
                'requested_navigation': {k:v for k,v in req['navigation'].items() if k!='bbox_wgs84'},
                'center_distance_from_original_m': req['navigation'].get('center_distance_from_original_m', 0),
                'context_extent_precision': 'Real camera navigation; no calibrated image-to-ground affine',
                'actual_viewport': [width, height],
                'historical_selection': history, 'source_years': acquisition_years,
                'source_date_raw': acquisition_dates,
                'known_acquisition_date_endpoints': acquisition_dates,
                'acquisition_range_verified': range_verified,
                'acquisition_date_evidence': dates['acquisition_date_basis'],
                'date_precision': 'Visible acquisition credits when present; selected historical date recorded separately',
                'date_unknown_or_mixed': not range_verified or len(acquisition_dates)>1,
                'source_native_resolution_m': None,
                'displayed_scale_and_date_status': [row['text'] for row in footer_ocr],
                'attribution_text_visible': text, 'original_browser_screenshot': str(raw),
                'source': 'Official Google Earth satellite web viewer, actual public browser screenshot',
                'request_url': url, 'final_url': page.url, 'browser_action_log': str(out/'browser_actions.json'),
                'model_preview_128': str(model_image),
                'interpretation_rule': 'Unregistered wider/panned/closer view only; never transfer screenshot coordinates into native128 labels.'}
            registered = aligned_frame(req, metadata, image, calibration, out)
            registered.update(dates, history_step=step, date_evidence=str(out/'ACQUISITION_EVIDENCE.json'),
                              source_native_resolution_m=None, provider_context_path=str(raw))
            write('frame_pending_quality_review.json', registered)
            metadata['registered_frame_pending_quality_review'] = registered
            write('browser_result.json', {'success': True, 'metadata': metadata,
                'images': [{'path': str(image), 'mime_type': 'image/jpeg', 'image_id': frame,
                            'role': 'context', 'provider': 'Google', 'sha256': digest,
                            'image_crs': None, 'image_transform': None, 'source_years': acquisition_years}]})
            write('timing_receipt.json', {'elapsed_seconds': time.time()-started, 'success': True,
                'candidate_id': req['candidate_id'], 'Google_browser_only': True,
                'Planet_calls': 0, 'model_calls': 0, 'source_years': acquisition_years,
                'historical_selection': history, 'full_QA_dimensions': [width, height],
                'model_dimensions': [128, 128]})
        except Exception as exc:
            record('browser_failed', error=str(exc))
            screen(page, 'failure_page.png')
            (out/'visible_body_text.txt').write_text(page.locator('body').inner_text())
            write('network_failures.json', network_failures)
            write('browser_result.json', {'success': False, 'metadata': {'status': 'unavailable',
                'provider': 'Google', 'candidate_id': req['candidate_id'], 'view': 'context',
                'annotation_allowed': False, 'reason': str(exc)}, 'images': []})
            write('timing_receipt.json', {'elapsed_seconds': time.time()-started, 'success': False,
                'candidate_id': req['candidate_id'], 'Planet_calls': 0, 'model_calls': 0})
            (out/'exception.txt').write_text(traceback.format_exc())
        finally:
            browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request_path', type=Path)
    main(parser.parse_args().request_path)
