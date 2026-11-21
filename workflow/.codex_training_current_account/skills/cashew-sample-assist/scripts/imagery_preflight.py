"""Acquire and rank date-qualified imagery locally before the Sol annotation turn."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import lru_cache
from pathlib import Path
from time import monotonic, time

from _runtime.common import write
from annotation_geometry import same_image_review
from imagery import esri, gehi, google
from imagery.coverage import black_mask_frame
from imagery_policy import DATE_CEILING, DATE_FLOOR, date_window, dates_for, frame_rejection
from imagery_quality import describe_season, frame_quality, quality_order, rainfall_normals
from select_imagery import PROVIDER_ERRORS, google_history

RECENT_FROM = date(2025, 1, 1)
RECENT_TO = date(2026, 12, 31)
MIN_RECENT = 5


def recent(frame):
    dates = dates_for(frame)
    return bool(dates) and all(RECENT_FROM <= day <= RECENT_TO for day in dates)


def capture_key(descriptor):
    return descriptor['provider'], dates_for(descriptor)


def group_candidates(descriptors, period):
    groups = defaultdict(list)
    for frame in descriptors:
        if frame_rejection(period, frame) or period == 2025 and dates_for(frame)[-1] > RECENT_TO:
            continue
        bucket = groups[capture_key(frame)]
        if not any(item['folder'] == frame['folder'] for item in bucket):
            bucket.append(frame)
    return list(groups.values())


def google_catalog(grid, folder):
    period = grid['label_period']
    floor = DATE_FLOOR[period]
    ceiling = RECENT_TO if period == 2025 else DATE_CEILING[period]
    if gehi.binary_path():
        try:
            records = gehi.catalog(grid, folder / 'GEHistoricalImagery', floor, ceiling, all_zooms=True)
            if records:
                return records
        except PROVIDER_ERRORS as error:
            write(folder / 'catalog_unavailable.json', {'reason': str(error)})
    current = None
    try:
        current = google.capture(grid, folder, floor)
    except PROVIDER_ERRORS as error:
        write(folder / 'current_unavailable.json', {'reason': str(error)})
    return ([current] if current else []) + list(google_history(grid, folder, floor))


def esri_catalog(grid, folder, floor, query_data, *, include_current):
    frames = []
    if include_current:
        try:
            frames.append(esri.current(grid, folder))
        except PROVIDER_ERRORS as error:
            write(folder / 'current_unavailable.json', {'reason': str(error)})
    try:
        frames.extend(esri.archive(grid, folder, floor, query_data=query_data, parallel_metadata=True))
    except PROVIDER_ERRORS as error:
        write(folder / ('archive_unavailable_' + floor.isoformat() + '.json'), {'reason': str(error)})
    return frames


def acquire_candidate(grid, options, attempted, normals):
    trials = []
    for descriptor in options:
        if descriptor['folder'] in attempted:
            continue
        trace = dict(folder=descriptor['folder'], provider=descriptor['provider'], dates=descriptor['acquisition_dates'])
        started = monotonic()
        try:
            if descriptor['provider'] == 'Esri':
                frame = esri.fetch_frame(grid, descriptor)
            elif descriptor.get('google_backend') == gehi.BACKEND:
                frame = gehi.fetch_frame(grid, descriptor)
            else:
                frame = descriptor
            reason = frame_rejection(grid['label_period'], frame)
            if reason:
                trials.append(dict(trace, status='excluded', reason=reason, seconds=round(monotonic()-started, 3)))
                continue
            frame = black_mask_frame(frame)
            quality = frame_quality(frame)
            season, rainfall = describe_season(frame['acquisition_dates'], normals)
            trials.append(dict(trace, status='scored', quality=quality, season=season, seconds=round(monotonic()-started, 3)))
            return dict(frame=frame, quality=quality, season=season, rainfall_mm=rainfall), trials
        except PROVIDER_ERRORS as error:
            trials.append(dict(trace, status='unavailable', reason=str(error), seconds=round(monotonic()-started, 3)))
    return None, trials


def prepare_inventory(grid, root, scene_reviews=()):
    started = monotonic()
    period = grid['label_period']
    normals = rainfall_normals(*grid['center_wgs84'])
    trials, accepted = [], {}
    attempted, seen_pixels = set(), set()
    progress = root / 'PROGRESS.json'
    write(progress, dict(stage='querying_catalogs', label_period=period, updated_epoch=time()))

    @lru_cache(maxsize=None)
    def esri_cached_query(url, params):
        return esri.json_data(url, dict(params) if params else None)

    def esri_query(url, params=None):
        return esri_cached_query(url, tuple(sorted(params.items())) if params else ())

    def acquire_groups(groups):
        # Provider groups already hold ranked versions of the same capture.
        pending = [group for group in groups if capture_key(group[0]) not in accepted]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(acquire_candidate, grid, options, attempted.copy(), normals) for options in pending]
            for future in futures:
                item, tried = future.result()
                trials.extend(tried)
                attempted.update(row['folder'] for row in tried)
                if not item:
                    continue
                frame = item['frame']
                identity = (item['quality']['rgb_pixel_sha256'], frame['native128_valid_mask_sha256'])
                if identity in seen_pixels:
                    trials[-1].update(status='duplicate_rgb', reason='same registered pixel values and support')
                    continue
                seen_pixels.add(identity)
                accepted[capture_key(frame)] = item

    with ThreadPoolExecutor(max_workers=2) as pool:
        google_future = pool.submit(google_catalog, grid, root / 'Google')
        esri_future = pool.submit(esri_catalog, grid, root / 'Esri',
                                  RECENT_FROM if period == 2025 else DATE_FLOOR[period], esri_query,
                                  include_current=True)
        google_frames, esri_frames = google_future.result(), esri_future.result()
    groups = group_candidates(google_frames + esri_frames, period)
    recent_groups = [group for group in groups if recent(group[0])]
    write(progress, dict(stage='downloading_and_scoring', recent_metadata_count=len(recent_groups), updated_epoch=time()))
    catalog_seconds = monotonic() - started
    download_started = monotonic()
    if period == 2025 and len(recent_groups) >= MIN_RECENT:
        acquire_groups(recent_groups)
    recent_download_seconds = monotonic() - download_started
    older_needed = period == 2025 and len(accepted) < MIN_RECENT
    download_seconds = recent_download_seconds
    if older_needed:
        write(progress, dict(stage='checking_older_catalogs', recent_metadata_count=len(recent_groups),
                             downloaded_scenes=len(accepted), updated_epoch=time()))
        def older_esri():
            start = monotonic()
            frames = esri_catalog(grid, root / 'Esri', DATE_FLOOR[period], esri_query, include_current=False)
            return frames, monotonic() - start

        with ThreadPoolExecutor(max_workers=1) as pool:
            older_future = pool.submit(older_esri)
            download_started = monotonic()
            acquire_groups(groups)
            download_seconds += monotonic() - download_started
            extra_frames, older_catalog_seconds = older_future.result()
        esri_frames += extra_frames
        groups = group_candidates(google_frames + esri_frames, period)
        catalog_seconds += older_catalog_seconds
    download_started = monotonic()
    if period == 2019 or older_needed:
        acquire_groups(groups)
    download_seconds += monotonic() - download_started

    window = date_window(period)
    if period == 2025:
        window = dict(earliest=(DATE_FLOOR[period] if older_needed else RECENT_FROM).isoformat(),
                      latest=RECENT_TO.isoformat())
    manifest = dict(label_period=period, acquisition_window=window, minimum_recent=MIN_RECENT,
                    recent_metadata_count=len(recent_groups), older_years_included=older_needed,
                    ranked_candidates=[], trials=trials,
                    timing=dict(catalog_seconds=round(catalog_seconds, 3), download_and_score_seconds=round(download_seconds, 3),
                                imagery_preflight_seconds=round(monotonic() - started, 3)))
    for item in sorted(accepted.values(), key=quality_order, reverse=True):
        frame, quality, season = (item[k] for k in ('frame', 'quality', 'season'))
        path = Path(frame['folder']) / 'SELECTION.json'
        write(path, dict(grid=grid, frame=frame, status='selected', selection_method='code_preflight',
                         season=season, rainfall_mm=item['rainfall_mm'], quality=quality,
                         acquisition_window=window,
                         review=dict(usable=True, confirmed_local_dry_season=season == 'dry', needs_dry_season=False),
                         same_image_operator_review=same_image_review(scene_reviews, grid, frame)))
        manifest['ranked_candidates'].append(str(path))
    write(root / 'SEARCH.json', manifest)
    write(progress, dict(stage='completed', recent_metadata_count=len(recent_groups), older_years_included=older_needed,
                         downloaded_scenes=len(accepted), updated_epoch=time()))
    return manifest
