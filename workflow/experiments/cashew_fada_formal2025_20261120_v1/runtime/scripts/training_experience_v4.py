#!/usr/bin/env python3
"""Append-only, lock-protected shared knowledge base for Training Expert."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("FADA_TRAINING_EXPERIENCE_ROOT", "@WORKSPACE@/training_experience"))
EVENTS = ROOT / "experience.jsonl"


def _tile_key(event: dict) -> str | None:
    """Return a stable tile key without mixing country/run history."""
    job_id = event.get("job_id")
    if not job_id:
        return None
    return str(event.get("thread_key") or job_id)


def _append(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def record(event: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"recorded_at": datetime.now(timezone.utc).isoformat(), **event}
    _append(EVENTS, payload)
    tile = _tile_key(event)
    if tile:
        _append(ROOT / "tiles" / tile / "events.jsonl", payload)


def record_once(event: dict, *, key_fields=("type", "run_id", "decision_epoch", "review_epoch")) -> None:
    """Append an event once, so interrupted-run recovery stays idempotent."""
    payload = {"recorded_at": datetime.now(timezone.utc).isoformat(), **event}
    ROOT.mkdir(parents=True, exist_ok=True)
    tile = _tile_key(event)
    tile_events = ROOT / "tiles" / tile / "events.jsonl" if tile else None
    with EVENTS.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        for line in handle:
            try:
                old = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(old.get(field) == payload.get(field) for field in key_fields):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                return
        handle.seek(0, 2)
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    if tile_events:
        # Keep the per-tile archive idempotent with the same semantic key.
        tile_events.parent.mkdir(parents=True, exist_ok=True)
        with tile_events.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            exists = False
            for line in handle:
                try:
                    old = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if all(old.get(field) == payload.get(field) for field in key_fields):
                    exists = True
                    break
            if not exists:
                handle.seek(0, 2)
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def canonical_records() -> list[dict]:
    """Return the append-only log with retry duplicates collapsed, latest wins."""
    if not EVENTS.is_file():
        return []
    ordered = {}
    serial = 0
    for line in EVENTS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type in {"agent_decision_request", "agent_decision"}:
            key = (event_type, event.get("run_id") or event.get("job_id"), event.get("epoch"))
        elif event_type in {"action_outcome", "action_outcome_reconciled"}:
            key = (event_type, event.get("run_id"), event.get("decision_epoch"), event.get("review_epoch"))
        else:
            key = ("unkeyed", serial)
            serial += 1
        ordered.pop(key, None)
        ordered[key] = event
    return list(ordered.values())


def recent(limit: int = 12) -> list[dict]:
    return canonical_records()[-int(limit):]


def recent_for_job(job_id: str, limit: int = 12) -> list[dict]:
    """Return only this tile's recent records for its isolated Agent thread."""
    # Agent packets must not reread the growing global NAS JSONL.  The
    # per-tile append-only index has exactly the needed scope.
    tile = str(job_id)
    path = ROOT / "tiles" / tile / "events.jsonl"
    if not path.is_file():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("job_id") == job_id:
            records.append(item)
    return records[-int(limit):]
