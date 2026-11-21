"""Read acquisition dates from visible Google Earth attribution."""

from datetime import date, timedelta
import re


MONTHS = {name.lower(): number for number, name in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}
SLASH_DATE = r"\d{1,2}/\d{1,2}/(?:19|20)\d{2}"


def visible_dates(text):
    """Read complete dates, including ranges whose separator OCR dropped."""
    text = re.sub(rf"({SLASH_DATE})(?={SLASH_DATE})", r"\1 ", text)
    values = []
    for month, day, year in re.findall(
            r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[A-Za-z]*)"
            r"\s*(\d{1,2})\s*,?\s*((?:19|20)\d{2})\b", text, re.I):
        values.append((int(year), MONTHS[month[:3].lower()], int(day)))
    for month, day, year in re.findall(r"(?<!\d)(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})(?!\d)", text):
        values.append((int(year), int(month), int(day)))
    for year, month, day in re.findall(r"(?<!\d)((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})(?!\d)", text):
        values.append((int(year), int(month), int(day)))
    days = []
    for value in values:
        try:
            parsed = date(*value).isoformat()
        except ValueError:
            continue
        if parsed not in days:
            days.append(parsed)
    return days


def evidence(credit_text, footer_text, selected_date=None):
    # Artwork, tips and captions outside the acquisition field cannot supply a
    # date or make the actual dates unknown.
    field = re.search(r"imagery\s+from\s+the\s+dates?\s*[:：]?\s*(.*?)(?:\nterms of service|\ndone|$)",
                      credit_text, re.I | re.S)
    date_text = field.group(1) if field else ""
    credit_dates = visible_dates(date_text)
    dates = sorted(set(credit_dates + visible_dates(footer_text)))
    if not field or not credit_dates:
        reason = "no_readable_attribution_dates"
    elif re.search(r"\bunknown\b", date_text, re.I):
        reason = "viewport_contains_unknown_dates"
    elif re.search(r"\b(?:older|newer|unknown|before|after|or later)\b", date_text + "\n" + footer_text, re.I):
        reason = "open_acquisition_range"
    elif selected_date and date.fromisoformat(dates[-1]) > date.fromisoformat(selected_date) + timedelta(days=1):
        reason = "acquisition_newer_than_history_selection"
    else:
        reason = None
    return dict(acquisition_dates=dates, acquisition_range_verified=reason is None,
                acquisition_date_issue=reason, attribution_dates=credit_dates,
                acquisition_date_scope="visible_google_earth_viewport",
                acquisition_date_basis="Visible data-attribution field" if credit_dates else "Visible imagery footer only")
