"""Verify that required bytes entered the actual visual model input."""

import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .common import read, training_account_home, write


def sessions_root():
    return (training_account_home() / "sessions").resolve()


def find_rollout(start):
    thread = start["thread"]
    sessions = sessions_root()
    if thread.get("path"):
        folder = Path(thread["path"]).resolve().parent
        folder.relative_to(sessions)
    else:
        day = datetime.fromtimestamp(thread["createdAt"], timezone.utc).strftime("%Y/%m/%d")
        folder = sessions / day
    candidates = list(folder.glob(f"*{thread['id']}.jsonl"))
    if not candidates:
        raise FileNotFoundError("Actual model rollout is not available yet for " + thread["id"])
    if len(candidates) != 1:
        raise ValueError("Actual model rollout is ambiguous for " + thread["id"])
    rollout = candidates[0].resolve(strict=True)
    rollout.relative_to(sessions)
    return rollout


def image_delivery_report(session_dir, registry, required_ids, turn_id=None, through_turn=None):
    """Match exact image bytes in the actual model input. Optionally scope to one turn."""
    fingerprints = {}
    for identity, image in registry.items():
        fingerprints.setdefault(image["sha256"], set()).add(identity)
    rollout = find_rollout(read(session_dir / "thread_start_response.json"))
    delivered = set()
    counts = Counter()
    active_turn = None
    seen_cutoff = False
    with rollout.open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith('\n'):
                    break  # The server may be appending this record during an in-turn delivery check.
                raise
            payload = event.get("payload", {})
            if event.get("type") == "turn_context" or (
                    event.get("type") == "event_msg" and payload.get("type") == "task_started"):
                active_turn = payload.get("turn_id")
                if through_turn and seen_cutoff and active_turn != through_turn:
                    break
                seen_cutoff |= bool(through_turn and active_turn == through_turn)
            if (through_turn and event.get("type") == "event_msg" and payload.get("type") == "task_complete"
                    and payload.get("turn_id") == through_turn):
                seen_cutoff = True
                break
            if event.get("type") != "response_item":
                continue
            if turn_id is not None and turn_id != active_turn:
                continue
            item = payload
            item_type = item.get("type")
            if item_type in ("custom_tool_call_output", "function_call_output"):
                parts = item.get("output")
            elif item_type == "message" and item.get("role") == "user":
                parts = item.get("content")
            else:
                continue
            if not isinstance(parts, list):
                continue
            for part in parts:
                if part.get("type") != "input_image":
                    continue
                url = part.get("image_url")
                counts["typed_images"] += 1
                if not isinstance(url, str) or not url.startswith("data:image/"):
                    continue
                payload = base64.b64decode(url.split(",", 1)[1], validate=True)
                fingerprint = hashlib.sha256(payload).hexdigest()
                delivered.update(fingerprints.get(fingerprint, ()))
    if through_turn and not seen_cutoff:
        raise ValueError("The decision turn is absent from its recorded thread: " + through_turn)
    missing = sorted(set(required_ids) - delivered)
    return dict(status="not_verified" if missing else "verified", actual_typed_images=counts["typed_images"],
                missing_image_ids=missing, verified_image_ids=sorted(set(required_ids) - set(missing)))


def verify_images(session_dir, turn_dir, registry, required_ids, *, through_turn=None):
    evidence = image_delivery_report(session_dir, registry, required_ids, through_turn=through_turn)
    if through_turn:
        evidence['decision_turn_id'] = through_turn
    report = turn_dir / "VISION_DELIVERY_AUDIT.json"
    write(report, evidence)
    if evidence["missing_image_ids"]:
        raise ValueError("Required images did not enter the visual input: " + ", ".join(evidence["missing_image_ids"]))
    return report
