"""Actual acquisition-date and coverage requirements for each label period."""

from datetime import date


DATE_FLOOR = {2019: date(2017, 1, 1), 2025: date(2024, 1, 1)}
DATE_CEILING = {2019: date(2020, 12, 31), 2025: None}
MINIMUM_VISIBLE_FRACTION = 0.95


def dates_for(frame):
    dates = frame.get("acquisition_dates")
    if not dates or not frame.get("acquisition_range_verified"):
        return ()
    try:
        return tuple(sorted(date.fromisoformat(day) for day in dates))
    except (TypeError, ValueError):
        return ()


def distinct_capture_periods(first, second):
    a, b = dates_for(first), dates_for(second)
    return bool(a and b and (a[-1] < b[0] or b[-1] < a[0]))


def date_window(period):
    ceiling = DATE_CEILING[period]
    return {"earliest": DATE_FLOOR[period].isoformat(), "latest": ceiling.isoformat() if ceiling else None}


def date_rejection(period, frame):
    dates = dates_for(frame)
    if not dates:
        return "acquisition_date_unverified"
    if dates[-1] > date.today():
        return "acquisition_date_in_future"
    if dates[0] < DATE_FLOOR[period]:
        return "acquisition_before_period_floor"
    ceiling = DATE_CEILING[period]
    if ceiling and dates[-1] > ceiling:
        return "acquisition_after_period_ceiling"
    return None


def frame_rejection(period, frame):
    reason = date_rejection(period, frame)
    if reason:
        return reason
    if (not frame.get("frame_georeferencing_verified")
            or frame.get("native128_valid_fraction", 0) < MINIMUM_VISIBLE_FRACTION):
        return "incomplete_registered_footprint"
    return None


def review_rejection(period, frame, decision):
    reason = frame_rejection(period, frame)
    if reason:
        return reason
    if not decision["usable"]:
        return "image_quality_or_interpretability"
    return None
