"""Acquire fresh Google and Esri imagery and review the exact AEF footprint."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import SubprocessError
import sys

import requests

from imagery import esri, gehi, google
from imagery.coverage import black_mask_frame
from imagery_policy import (DATE_FLOOR, DATE_CEILING, date_window, dates_for, distinct_capture_periods,
                            frame_rejection, review_rejection)
from _runtime import transport
from _runtime.common import EFFORT, MODEL, SKILL_ROOT, write
from _runtime.vision_audit import verify_images
from annotation_geometry import same_image_review


REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["usable", "needs_dry_season", "confirmed_local_dry_season", "explanation"],
    "properties": {
        "usable": {"type": "boolean"},
        "needs_dry_season": {"type": "boolean"},
        "confirmed_local_dry_season": {"type": "boolean"},
        "explanation": {"type": "string", "minLength": 1},
    },
}
PROVIDER_ERRORS = (ValueError, RuntimeError, OSError, requests.RequestException, SubprocessError)


def validate_final(context, result):
    return {"eligible": bool(result["explanation"].strip()) and context["image_id"] in context["registered_frames"]}


def capture_current(grid, root, oldest):
    def google_candidates():
        if gehi.binary_path():
            source = root / "Google" / "GEHistoricalImagery"
            try:
                if candidates := gehi.catalog(grid, source, oldest, DATE_CEILING[grid['label_period']]):
                    return candidates
            except PROVIDER_ERRORS as error:
                write(source / "catalog_unavailable.json", {"reason": str(error)})
        frame = google.capture(grid, root / "Google", oldest)
        return [frame] if frame else []

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            "Google": pool.submit(google_candidates),
            "Esri": pool.submit(esri.current, grid, root / "Esri"),
        }
        frames = []
        for provider, future in futures.items():
            try:
                result = future.result()
                frames.extend(result if provider == "Google" else [result])
            except PROVIDER_ERRORS as error:
                write(root / provider / "current_unavailable.json", {"reason": str(error)})
        return [frame for frame in frames if frame]


def esri_history(grid, root, oldest):
    try:
        frames = sorted(esri.archive(grid, root, oldest), key=dates_for, reverse=True)
    except PROVIDER_ERRORS as error:
        write(root / "archive_unavailable.json", {"reason": str(error)})
        return
    yield from frames


def google_history(grid, root, oldest):
    step = 0
    while True:
        try:
            frame = google.capture(grid, root, oldest, step)
        except PROVIDER_ERRORS as error:
            folder = root / f"history_{step:03d}"
            write(folder / "unavailable.json", {"reason": str(error)})
            if not (folder / "history_rendered_ocr.json").exists():
                return
        else:
            if frame is None:
                return
            yield frame
        step += 1


def ordered_history(grid, root, oldest):
    streams = [esri_history(grid, root / "Esri", oldest), google_history(grid, root / "Google", oldest)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(next, stream, None) for stream in streams]
        heads = [future.result() for future in pending]
    while any(frame is not None for frame in heads):
        index = max((i for i, frame in enumerate(heads) if frame is not None), key=lambda i: dates_for(heads[i]))
        yield heads[index]
        heads[index] = next(streams[index], None)


class Selector:
    def __init__(self, grid, root, scene_reviews=()):
        self.grid = grid
        self.scene_reviews = scene_reviews
        self.root = root
        self.period = grid["label_period"]
        self.needs_dry_season = False
        self.session = None
        self.reviews = 0
        self.trials = []
        self.current = []
        self.registered = {}

    def save_search(self):
        write(self.root / "SEARCH.json", {"label_period": self.period,
              "acquisition_window": date_window(self.period), "trials": self.trials})

    def record(self, frame, status, reason, **extra):
        self.trials.append(dict(provider=frame["provider"], folder=frame["folder"],
                                dates=frame["acquisition_dates"], status=status, reason=reason, **extra))
        self.save_search()

    def review(self, frame):
        self.reviews += 1
        case_id = self.grid["candidate_id"]
        directory = self.root / "review" / f"{self.reviews:03d}"
        image_id = f"{case_id}_source_{self.reviews:03d}"
        image = dict(path=frame["image_path"], image_id=image_id, role="current_candidate_complete_RGB",
                     provider=frame["provider"], source_date=frame["acquisition_dates"], sha256=frame["sha256"],
                     bounds_wgs84=self.grid["bounds_wgs84"])
        if self.session is None:
            description = "Review freshly acquired RGB and return only the requested image suitability decision."
            thread = transport.prepare_thread(description, [])
            folder = self.root / "review"
            folder.mkdir(parents=True, exist_ok=True)
            self.session = transport.Session(folder, thread, sys.modules[__name__])
            self.session.start()
        prompt = (SKILL_ROOT / "references/imagery-review-prompt.md").read_text()
        transport.prepare_turn(self.session.thread, directory, prompt, [image],
                               dict(case_id=case_id, country=self.grid["country"], label_period=self.period,
                                    requested_image_id=image_id, acquisition_window=date_window(self.period),
                                    already_needs_dry_season=self.needs_dry_season,
                                    missing_native128_pixels=frame.get("native128_missing_pixels", 0)),
                               REVIEW_SCHEMA)
        result, registry = self.session.run_turn(directory, {"image_id": image_id}, timeout_seconds=1200)
        audit = verify_images(self.session.directory, directory, registry, [image_id])
        return result, audit

    def acquire(self, frame):
        key = frame["folder"]
        if key in self.registered:
            return self.registered[key]
        if frame["provider"] == "Esri":
            frame = esri.fetch_frame(self.grid, frame)
        elif frame.get("google_backend") == gehi.BACKEND:
            frame = gehi.fetch_frame(self.grid, frame)
        if not frame_rejection(self.period, frame):
            frame = black_mask_frame(frame)
            self.registered[key] = frame
        return frame

    def comparison_for(self, primary, preferred=None):
        """Get one verified-date comparison without spending another model turn."""
        candidates = sorted(self.current, key=lambda f: f['provider'] == primary['provider'])
        seen = set()
        attempts = 0
        for candidate in ([preferred, *candidates] if preferred else candidates):
            key = candidate['folder']
            if key in seen or not distinct_capture_periods(primary, candidate) or frame_rejection(self.period, candidate):
                continue
            if attempts == 2:
                break
            seen.add(key)
            attempts += 1
            try:
                frame = self.acquire(candidate)
            except PROVIDER_ERRORS as error:
                self.record(candidate, "comparison_unavailable", str(error))
                continue
            reason = frame_rejection(self.period, frame)
            if reason:
                self.record(frame, "comparison_excluded", reason)
                continue
            if frame.get('rgb_pixel_sha256', frame['sha256']) == primary.get('rgb_pixel_sha256', primary['sha256']):
                self.record(frame, 'comparison_excluded', 'identical_rgb')
                continue
            self.record(frame, "comparison_ready", "Different-date context for visual interpretation")
            return frame
        return None

    def choices(self):
        oldest = DATE_FLOOR[self.period]
        seen = set()
        self.current = sorted(capture_current(self.grid, self.root, oldest), key=dates_for, reverse=True)

        def candidates():
            yield from self.current
            yield from ordered_history(self.grid, self.root, oldest)

        for frame in candidates():
            skipped = frame_rejection(self.period, frame)
            if skipped:
                issue = frame.get("acquisition_date_issue")
                self.record(frame, "excluded", skipped, **({"date_evidence_issue": issue} if issue else {}))
                continue
            try:
                frame = self.acquire(frame)
            except PROVIDER_ERRORS as error:
                self.record(frame, "unavailable", str(error))
                continue
            skipped = frame_rejection(self.period, frame)
            if skipped:
                self.record(frame, "excluded", skipped)
                continue
            identity = (frame.get("rgb_pixel_sha256", frame["sha256"]), tuple(dates_for(frame)))
            if identity in seen:
                self.record(frame, "duplicate_in_current_run", "same RGB and acquisition dates")
                continue
            seen.add(identity)
            decision, audit = self.review(frame)
            rejection = review_rejection(self.period, frame, decision)
            self.needs_dry_season |= decision["needs_dry_season"]
            if rejection:
                self.record(frame, "excluded", rejection, review=decision, audit=str(audit))
                continue
            self.record(frame, "selected", decision["explanation"], review=decision, audit=str(audit))
            selection = dict(status="selected", grid=self.grid, frame=frame, review=decision,
                             same_image_operator_review=same_image_review(self.scene_reviews, self.grid, frame),
                             dry_season_requested=self.needs_dry_season,
                             vision_audit=str(audit), model=MODEL, effort=EFFORT, service_tier=self.session.service_tier)
            receipt = Path(frame["folder"]) / "SELECTION.json"
            write(receipt, selection)
            yield receipt

    def close(self):
        if self.session:
            self.session.close()
