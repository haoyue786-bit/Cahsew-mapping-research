"""Run the live, locally registered Google Earth browser adapter."""

import os
import subprocess
import sys
from pathlib import Path

from _runtime.common import read, write


def capture_view(grid, folder, floor, step, radius):
    request = folder / "request.json"
    write(request, dict(candidate_id=grid["candidate_id"], candidate=grid, output_dir=str(folder),
                        date_floor=floor.isoformat(), history_step=step, view="detail",
                        navigation=dict(center_wgs84=grid["center_wgs84"], requested_radius_m=radius)))
    command = [os.environ.get("CASHEW_BROWSER_PYTHON", sys.executable),
               str(Path(__file__).with_name("google_capture.py")), str(request)]
    with (folder / "capture.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=320 + 12 * (step or 0))
    if (folder / "history_end.json").exists():
        return None
    response = read(folder / "browser_result.json")
    if not response["success"]:
        raise RuntimeError(response["metadata"]["reason"])
    frame = read(folder / "frame_pending_quality_review.json")
    return dict(frame, folder=str(folder), source_date=frame["acquisition_dates"])


def capture(grid, root, floor, step=None):
    name = "current" if step is None else f"history_{step:03d}"
    folder = root / name
    frame = capture_view(grid, folder, floor, step, radius=1250)
    if not frame or frame["acquisition_date_issue"] != "viewport_contains_unknown_dates":
        return frame
    if max(frame["acquisition_dates"], default="") < floor.isoformat():
        return frame

    # Attribution refers to the visible viewport. Reacquire a smaller one
    # before rejecting a sample because the surrounding map has unknown dates.
    tighter_folder = folder / "tighter_view"
    try:
        tighter = capture_view(grid, tighter_folder, floor, step, radius=1000)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        write(tighter_folder / "unavailable.json", {"reason": str(error)})
        return frame
    use_tighter = (tighter and tighter["acquisition_range_verified"]
                   and tighter["native128_valid_fraction"] == 1
                   and min(tighter["acquisition_dates"]) >= floor.isoformat())
    write(folder / "TIGHTER_VIEW.json", {"selected": bool(use_tighter), "previous": frame["date_evidence"],
                                        "new": tighter["date_evidence"] if tighter else None})
    return tighter if use_tighter else frame
