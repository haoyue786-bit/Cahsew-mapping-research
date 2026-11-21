#!/usr/bin/env python3
"""Thin persistent Codex app-server client for one tile's Luna thread."""

from __future__ import annotations

import json
import collections
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PersistentLunaRecoveryRequired(RuntimeError):
    """The last turn has an uncertain outcome; never submit it again blindly."""


class PersistentLunaTransportError(RuntimeError):
    """The app-server transport or process failed; do not issue a repair turn."""


class PersistentLunaProtocolError(RuntimeError):
    """The app-server returned an invalid or rejected JSON-RPC exchange."""


@dataclass(frozen=True)
class _StreamEvent:
    """A single terminal-safe event produced by one dedicated pipe reader."""

    kind: str
    value: Any = None


class PersistentLunaClient:
    def __init__(self, codex_binary: str = "codex", timeout_seconds: int = 900) -> None:
        self.codex_binary = codex_binary
        self.timeout_seconds = int(timeout_seconds)
        self.process: subprocess.Popen[str] | None = None
        self.request_id = 0
        self._events: list[dict[str, Any]] = []
        self.last_reconciliation: dict[str, Any] | None = None
        self._stdout_queue: queue.Queue = queue.Queue()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=200)
        self._stderr_lock = threading.Lock()
        self._stdout_reader_thread: threading.Thread | None = None
        self._stderr_reader_thread: threading.Thread | None = None

    def start_server(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        command = [self.codex_binary, "app-server", "--listen", "stdio://"]
        for feature in ("shell_tool", "unified_exec", "view_image", "apps", "plugins", "memories",
                        "multi_agent", "multi_agent_v2", "browser_use", "computer_use", "image_generation",
                        "goals", "skill_search", "tool_suggest", "in_app_browser", "in_app_local_automation",
                        "sleep_tool"):
            command.extend(("--disable", feature))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            close_fds=True,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._start_stream_readers()
        self.initialize()

    def _start_stream_readers(self) -> None:
        """Create the only consumers of app-server stdout and stderr.

        Do not mix fd readiness polling with TextIOWrapper.readline(): readline
        may have already buffered subsequent JSONL messages in user space.
        """
        self._stdout_queue = queue.Queue()
        with self._stderr_lock:
            self._stderr_tail.clear()
        self._stdout_reader_thread = threading.Thread(
            target=self._stdout_reader_loop,
            name="cashew-codex-app-server-stdout",
            daemon=True,
        )
        self._stderr_reader_thread = threading.Thread(
            target=self._stderr_reader_loop,
            name="cashew-codex-app-server-stderr",
            daemon=True,
        )
        self._stdout_reader_thread.start()
        self._stderr_reader_thread.start()

    def _stdout_reader_loop(self) -> None:
        process = self.process
        stdout = None if process is None else process.stdout
        if stdout is None:
            self._stdout_queue.put(_StreamEvent("error", RuntimeError("app-server stdout is unavailable")))
            return
        try:
            while True:
                line = stdout.readline()
                if line == "":
                    self._stdout_queue.put(_StreamEvent("eof"))
                    return
                self._stdout_queue.put(_StreamEvent("line", line))
        except BaseException as exc:  # reader failure must wake the main thread
            self._stdout_queue.put(_StreamEvent("error", exc))

    def _stderr_reader_loop(self) -> None:
        process = self.process
        stderr = None if process is None else process.stderr
        if stderr is None:
            return
        try:
            for line in stderr:
                with self._stderr_lock:
                    self._stderr_tail.append(line.rstrip("\n"))
        except BaseException as exc:
            with self._stderr_lock:
                self._stderr_tail.append(f"[stderr-reader-error] {exc!r}")

    def _stderr_tail_text(self) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr_tail)

    def _send(self, payload: dict[str, Any]) -> None:
        if self.process is None or self.process.poll() is not None or self.process.stdin is None:
            raise PersistentLunaTransportError({"app_server_not_running": True,
                "exit_code": None if self.process is None else self.process.poll(),
                "stderr": self._stderr_tail_text()})
        try:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except OSError as exc:
            raise PersistentLunaTransportError({"app_server_stdin_failed": str(exc), "stderr": self._stderr_tail_text()}) from exc

    def _read_line(self, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Codex operation exceeded deadline while waiting for stdout")
        try:
            event = self._stdout_queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("persistent app-server response timed out") from exc
        if event.kind == "eof":
            raise PersistentLunaTransportError({"app_server_exited": None if self.process is None else self.process.poll(), "stderr": self._stderr_tail_text()})
        if event.kind == "error":
            raise PersistentLunaTransportError({"app_server_stdout_reader_failed": repr(event.value), "stderr": self._stderr_tail_text()}) from event.value
        if event.kind != "line":
            raise PersistentLunaTransportError({"unknown_app_server_stream_event": event.kind, "stderr": self._stderr_tail_text()})
        line = str(event.value)
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise PersistentLunaProtocolError({"invalid_app_server_json": line[:1000]}) from exc

    def _request(self, method: str, params: dict[str, Any], timeout_seconds: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self.request_id += 1
        request_id = self.request_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + float(timeout_seconds or self.timeout_seconds)
        events: list[dict[str, Any]] = []
        while True:
            message = self._read_line(deadline)
            if message.get("id") == request_id:
                if "error" in message:
                    raise PersistentLunaProtocolError({"app_server_method": method, "error": message["error"]})
                return message.get("result", {}), events
            events.append(message)

    def initialize(self) -> dict[str, Any]:
        result, _ = self._request("initialize", {
            "clientInfo": {"name": "cashew-fada-persistent-controller", "version": "1"},
            "capabilities": {"experimentalApi": True},
        })
        self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        return result

    def start_thread(self, *, cwd: Path, model: str, reasoning_effort: str, sandbox: str, developer_instructions: str) -> dict[str, Any]:
        self._requested_sandbox = sandbox
        result, events = self._request("thread/start", {"cwd": str(cwd.resolve()), "model": model, "sandbox": sandbox, "approvalPolicy": "never", "ephemeral": False, "developerInstructions": developer_instructions, "experimentalRawEvents": False})
        self._events.extend(events)
        thread = result.get("thread", result)
        thread_id = thread.get("id")
        if not thread_id:
            raise RuntimeError({"thread_start_missing_id": result})
        return {"thread": thread, "thread_id": thread_id, "model": model, "reasoning_effort": reasoning_effort, "sandbox": sandbox, "ephemeral": False}

    def resume_thread(self, *, thread_id: str, cwd: Path, model: str, reasoning_effort: str, sandbox: str) -> dict[str, Any]:
        self._requested_sandbox = sandbox
        result, events = self._request("thread/resume", {"threadId": str(thread_id), "cwd": str(cwd.resolve()), "model": model, "sandbox": sandbox, "approvalPolicy": "never", "excludeTurns": True})
        self._events.extend(events)
        thread = result.get("thread", result)
        actual_id = thread.get("id", thread_id)
        if actual_id != thread_id:
            raise RuntimeError({"thread_resume_id_mismatch": [thread_id, actual_id]})
        return {"thread": thread, "thread_id": thread_id, "model": model, "reasoning_effort": reasoning_effort, "sandbox": sandbox, "ephemeral": False}

    def reconcile_thread(self, *, thread_id: str, turn_id: str | None = None, timeout_seconds: int | None = None) -> dict[str, Any]:
        """Read durable thread state after an uncertain turn.

        This is deliberately a read-only operation.  A timeout must never cause
        the caller to submit the same turn again until this result is known.
        """
        result, events = self._request(
            "thread/read",
            {"threadId": str(thread_id), "includeTurns": True},
            timeout_seconds,
        )
        self._events.extend(events)
        thread = result.get("thread", result)
        turns = thread.get("turns", []) if isinstance(thread, dict) else []
        matching = [item for item in turns if isinstance(item, dict) and (turn_id is None or item.get("id") == turn_id)]
        if turn_id is not None and not matching:
            self.last_reconciliation = {"thread": thread, "turn": None, "turns": turns, "events": events, "reconciled": True}
            return self.last_reconciliation
        turn = matching[-1] if matching else (turns[-1] if turns else None)
        self.last_reconciliation = {"thread": thread, "turn": turn, "turns": turns, "events": events, "reconciled": True}
        return self.last_reconciliation

    def _reconcile_with_reconnect(self, *, thread_id: str, turn_id: str | None = None, timeout_seconds: int = 60) -> dict[str, Any]:
        """Retry only the read of an existing turn; never submit another turn.

        A dead transport may require a new app-server. A running but
        interrupted turn stays unresolved; the caller must not retry it until
        its exact durable state has been checked.
        """
        last = None
        for attempt in range(3):
            try:
                if self.process is None or self.process.poll() is not None:
                    self.close()
                    self.start_server()
                return self.reconcile_thread(thread_id=thread_id, turn_id=turn_id, timeout_seconds=timeout_seconds)
            except (PersistentLunaTransportError, TimeoutError, OSError) as exc:
                last = exc
                if attempt == 2:
                    raise
                # A read timeout does not imply that the server or its model
                # turn has died. Closing a live server here could interrupt a
                # merely slow decision. Reconnect only when transport itself
                # failed or the process has actually exited.
                if not isinstance(exc, TimeoutError) or self.process is None or self.process.poll() is not None:
                    self.close()
                time.sleep(attempt + 1)
        raise last

    def _capture_failed_turn_state(self, thread_id: str, turn_id: str, timeout_seconds: int | None) -> None:
        try:
            self._reconcile_with_reconnect(thread_id=thread_id, turn_id=turn_id,
                                           timeout_seconds=min(timeout_seconds or self.timeout_seconds, 60))
        except Exception as exc:
            raise PersistentLunaRecoveryRequired({"persistent_turn_failure_reconciliation_failed": str(exc),
                "thread_id": thread_id, "turn_id": turn_id}) from exc

    @staticmethod
    def _message_text(item: Any) -> str | None:
        if isinstance(item, dict):
            if item.get("type") in {"agentMessage", "agent_message"} and isinstance(item.get("text"), str):
                return item["text"]
            content = item.get("content")
            if isinstance(content, list):
                pieces = [PersistentLunaClient._message_text(value) for value in content]
                pieces = [value for value in pieces if value]
                if pieces:
                    return "".join(pieces)
            for value in item.values():
                text = PersistentLunaClient._message_text(value)
                if text:
                    return text
        elif isinstance(item, list):
            pieces = [PersistentLunaClient._message_text(value) for value in item]
            pieces = [value for value in pieces if value]
            if pieces:
                return "".join(pieces)
        return None

    @staticmethod
    def _select_final_agent_message(items: Any) -> dict[str, str] | None:
        """Select a usable formal answer, never a commentary message.

        Current app-server messages carry ``phase=commentary`` or
        ``phase=final_answer``.  Older persisted turns can omit phase, so an
        unknown phase is retained only as a legacy fallback.
        """
        if not isinstance(items, list):
            return None
        legacy: dict[str, str] | None = None
        for item in reversed(items):
            if not isinstance(item, dict) or item.get("type") not in {"agentMessage", "agent_message"}:
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            phase = item.get("phase")
            if phase == "final_answer":
                return {"text": text, "phase": "final_answer", "item_id": str(item.get("id", ""))}
            if phase is None and legacy is None:
                legacy = {"text": text, "phase": "legacy_unknown_phase", "item_id": str(item.get("id", ""))}
        return legacy

    @staticmethod
    def _event_param(params: dict[str, Any], camel: str, snake: str) -> Any:
        return params.get(camel, params.get(snake))

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _record_selected_message(state: dict[str, Any], selected: dict[str, str], *, from_item_completed: bool, from_thread: bool = False) -> None:
        """Record only final_answer or phase-less legacy output."""
        phase = selected["phase"]
        existing_phase = state.get("selected_agent_message_phase")
        if existing_phase == "final_answer" and phase != "final_answer":
            return
        state["final_text"] = selected["text"]
        state["selected_agent_message_phase"] = phase
        state["selected_agent_message_id"] = selected.get("item_id")
        if from_item_completed:
            state["final_answer_item_completed_seen"] = True
            state["final_answer_item_completed_at"] = PersistentLunaClient._now_iso()
            state["final_answer_item_completed_monotonic"] = time.monotonic()
        if from_thread:
            state["final_answer_found_in_thread"] = True

    def _observe_event(self, message: dict[str, Any], *, thread_id: str, turn_id: str, state: dict[str, Any]) -> None:
        """Track agent-message completion and turn completion independently."""
        method = message.get("method")
        params_event = message.get("params", {})
        event_thread_id = self._event_param(params_event, "threadId", "thread_id")
        event_turn_id = self._event_param(params_event, "turnId", "turn_id")
        if event_thread_id != thread_id or (event_turn_id is not None and event_turn_id != turn_id):
            return
        if method == "item/completed":
            item = params_event.get("item", {})
            if isinstance(item, dict) and item.get("type") in {"agentMessage", "agent_message"}:
                state["agent_message_item_completed_seen"] = True
                state["agent_message_item_id"] = item.get("id")
                if item.get("phase") == "commentary":
                    state["commentary_agent_message_count"] = int(state.get("commentary_agent_message_count", 0)) + 1
                selected = self._select_final_agent_message([item])
                if selected is not None:
                    self._record_selected_message(state, selected, from_item_completed=True)
        elif method == "turn/completed":
            completed_turn = params_event.get("turn", {})
            if completed_turn.get("id") == turn_id:
                state["turn_completed_notification_seen"] = True
                state["turn_completed"] = True
                state["turn_status"] = completed_turn.get("status")
                state["turn_completed_at"] = self._now_iso()
                state["turn_completed_monotonic"] = time.monotonic()
                if not state.get("final_text"):
                    selected = self._select_final_agent_message(completed_turn.get("items", []))
                    if selected is not None:
                        self._record_selected_message(state, selected, from_item_completed=False)
        elif method == "turn/failed":
            state["turn_failed"] = True
            state["turn_failure"] = params_event

    def _finish_reconciled_turn(self, *, thread_id: str, turn_id: str, reconciliation: dict[str, Any], state: dict[str, Any], all_events: list[dict[str, Any]]) -> dict[str, Any]:
        for message in reconciliation.get("events", []):
            self._observe_event(message, thread_id=thread_id, turn_id=turn_id, state=state)
        turn = reconciliation.get("turn")
        if isinstance(turn, dict) and turn.get("id") == turn_id:
            state["turn_status"] = turn.get("status")
            if turn.get("status") == "completed":
                state["turn_completed"] = True
                selected = self._select_final_agent_message(turn.get("items", []))
                if selected is not None:
                    self._record_selected_message(state, selected, from_item_completed=False, from_thread=True)
        if state.get("turn_status") != "completed":
            raise PersistentLunaRecoveryRequired({
                "persistent_turn_reconciliation_required": True,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "turn_status": state.get("turn_status"),
                "thread_status": (reconciliation.get("thread") or {}).get("status"),
                "reconciled": True,
                "events": all_events[-8:],
            })
        if not state.get("final_text"):
            raise PersistentLunaRecoveryRequired({
                "persistent_turn_completed_without_agent_message": True,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "agent_message_item_completed": state.get("agent_message_item_completed_seen", False),
                "turn_completed": True,
                "reconciled": True,
            })
        state["completion_source"] = "durable_thread_reconciliation"
        return self._turn_result(thread_id, turn_id, state, all_events, reconciled=True)

    def _turn_result(self, thread_id: str, turn_id: str, state: dict[str, Any], all_events: list[dict[str, Any]], *, reconciled: bool) -> dict[str, Any]:
        usage = next((event.get("params", {}).get("tokenUsage") for event in reversed(all_events)
                      if event.get("method") == "thread/tokenUsage/updated"), None)
        return {
            "thread_id": thread_id,
            "turn_id": turn_id,
            "final_text": state["final_text"],
            "events": all_events,
            "event_tracking": {
                "agent_message_item_completed": bool(state.get("agent_message_item_completed_seen")),
                "agent_message_item_completed_seen": bool(state.get("agent_message_item_completed_seen")),
                "final_answer_item_completed_seen": bool(state.get("final_answer_item_completed_seen")),
                "final_answer_found_in_thread": bool(state.get("final_answer_found_in_thread")),
                "selected_agent_message_phase": state.get("selected_agent_message_phase"),
                "commentary_agent_message_count": int(state.get("commentary_agent_message_count", 0)),
                "turn_completed": bool(state.get("turn_completed")),
                "turn_completed_notification_seen": bool(state.get("turn_completed_notification_seen")),
                "turn_status": state.get("turn_status"),
                "completion_source": state.get("completion_source", "durable_thread_reconciliation" if reconciled else "event_stream"),
                "request_sent_at": state.get("request_sent_at"),
                "final_answer_item_completed_at": state.get("final_answer_item_completed_at"),
                "turn_completed_at": state.get("turn_completed_at"),
                "request_sent_monotonic": state.get("request_sent_monotonic"),
                "final_answer_item_completed_monotonic": state.get("final_answer_item_completed_monotonic"),
                "turn_completed_monotonic": state.get("turn_completed_monotonic"),
                "reconciled": reconciled,
            },
            "skill_name": state["skill_name"],
            "skill_path": state["skill_path"],
            "model": state.get("model", ""),
            "reasoning_effort": state.get("reasoning_effort", ""),
            "sandbox": getattr(self, "_requested_sandbox", "unknown"),
            "approval_policy": "never",
            "token_usage": usage,
            "persistent_thread": True,
            "fresh_ephemeral_session": False,
        }

    def run_structured_turn(self, *, thread_id: str, text: str, skill_name: str, skill_path: Path, output_schema: dict[str, Any], model: str, reasoning_effort: str, timeout_seconds: int | None = None, previous_turn_id: str | None = None, include_skill: bool = True) -> dict[str, Any]:
        if f"${skill_name}" not in text:
            raise ValueError("every persistent Agent turn must explicitly mention the controller Skill")
        params = {
            "threadId": str(thread_id),
            "model": model,
            "effort": reasoning_effort,
            "outputSchema": output_schema,
            "input": ([{"type": "skill", "name": skill_name, "path": str(skill_path.resolve())}] if include_skill else []) + [{"type": "text", "text": text}],
        }
        request_sent_at = self._now_iso()
        request_sent_monotonic = time.monotonic()
        try:
            result, events = self._request("turn/start", params, timeout_seconds)
        except (TimeoutError, PersistentLunaTransportError) as exc:
            # The turn/start response itself may have been lost.  Reconcile
            # the durable thread before deciding that recovery is required.
            process_before_reconciliation = self.process
            try:
                reconciliation = self._reconcile_with_reconnect(thread_id=thread_id, timeout_seconds=min(timeout_seconds or self.timeout_seconds, 60))
            except Exception as reconciliation_error:
                raise PersistentLunaRecoveryRequired({"persistent_turn_start_timeout_reconciliation_failed": str(reconciliation_error), "thread_id": thread_id}) from reconciliation_error
            turn = reconciliation.get("turn")
            if isinstance(turn, dict) and turn.get("id") and str(turn["id"]) != str(previous_turn_id):
                turn_id = str(turn["id"])
                state = {"skill_name": skill_name, "skill_path": str(skill_path.resolve()), "model": model, "reasoning_effort": reasoning_effort, "request_sent_at": request_sent_at, "request_sent_monotonic": request_sent_monotonic}
                if (isinstance(exc, TimeoutError) and turn.get("status") == "inProgress"
                        and process_before_reconciliation is not None
                        and self.process is process_before_reconciliation and self.process.poll() is None):
                    # The original connection is alive and identifies the new
                    # active turn. Continue consuming that same turn; never
                    # submit turn/start a second time.
                    result = {"turn": turn}
                    events = list(reconciliation.get("events", []))
                else:
                    return self._finish_reconciled_turn(thread_id=thread_id, turn_id=turn_id, reconciliation=reconciliation, state=state, all_events=list(reconciliation.get("events", [])))
            else:
                raise PersistentLunaRecoveryRequired({"persistent_turn_start_timeout_reconciled": True, "thread_id": thread_id, "reconciliation": reconciliation}) from exc
        turn = result.get("turn", result)
        turn_id = turn.get("id")
        all_events = list(events)
        if not turn_id:
            for event in events:
                if event.get("method") == "turn/started":
                    turn_id = event.get("params", {}).get("turn", {}).get("id")
                    break
        if not turn_id:
            raise PersistentLunaProtocolError({"turn_start_missing_id": result})
        deadline = time.monotonic() + float(timeout_seconds or self.timeout_seconds)
        state = {"skill_name": skill_name, "skill_path": str(skill_path.resolve()), "model": model, "reasoning_effort": reasoning_effort, "request_sent_at": request_sent_at, "request_sent_monotonic": request_sent_monotonic}
        slow_turn_extensions = 0
        # Notifications can arrive before the turn/start response.  Consume
        # those prefetched events before blocking for another line.
        for message in all_events:
            self._observe_event(message, thread_id=thread_id, turn_id=turn_id, state=state)
        while not state.get("turn_completed") and not state.get("turn_failed"):
            try:
                message = self._read_line(deadline)
            except (TimeoutError, PersistentLunaTransportError) as exc:
                process_before_reconciliation = self.process
                try:
                    reconciliation = self._reconcile_with_reconnect(thread_id=thread_id, turn_id=turn_id, timeout_seconds=min(timeout_seconds or self.timeout_seconds, 60))
                except Exception as reconciliation_error:
                    raise PersistentLunaRecoveryRequired({"persistent_turn_timeout_reconciliation_failed": str(reconciliation_error), "thread_id": thread_id, "turn_id": turn_id}) from reconciliation_error
                all_events.extend(reconciliation.get("events", []))
                reconciled_turn = reconciliation.get("turn") or {}
                # A slow live turn is neither a failed turn nor a new request.
                # Continue serving its tool calls only on the original live
                # connection, for at most two additional bounded intervals.
                if (isinstance(exc, TimeoutError) and reconciled_turn.get("id") == turn_id
                        and reconciled_turn.get("status") == "inProgress"
                        and process_before_reconciliation is not None
                        and self.process is process_before_reconciliation
                        and self.process.poll() is None and slow_turn_extensions < 2):
                    for event in reconciliation.get("events", []):
                        self._observe_event(event, thread_id=thread_id, turn_id=turn_id, state=state)
                    slow_turn_extensions += 1
                    deadline = time.monotonic() + float(timeout_seconds or self.timeout_seconds)
                    continue
                return self._finish_reconciled_turn(thread_id=thread_id, turn_id=turn_id, reconciliation=reconciliation, state=state, all_events=all_events)
            all_events.append(message)
            self._observe_event(message, thread_id=thread_id, turn_id=turn_id, state=state)
        if state.get("turn_failed"):
            # A failure notification alone is not a durable, identity-checked
            # terminal record. Preserve the exact turn state for the worker's
            # separate retry/recovery gate; never submit a replacement here.
            self._capture_failed_turn_state(thread_id, turn_id, timeout_seconds)
            raise PersistentLunaRecoveryRequired({"persistent_turn_failed": state.get("turn_failure"),
                                                   "thread_id": thread_id, "turn_id": turn_id, "events": all_events[-8:]})
        if state.get("turn_status") and state["turn_status"] != "completed":
            self._capture_failed_turn_state(thread_id, turn_id, timeout_seconds)
            raise PersistentLunaRecoveryRequired({"persistent_turn_status": state["turn_status"], "turn_id": turn_id, "events": all_events[-8:]})
        self._events.extend(all_events)
        if not state.get("final_text"):
            # A completed turn with no final item notification is only usable
            # after durable state confirms a final_answer for this exact turn.
            reconciliation = self.reconcile_thread(thread_id=thread_id, turn_id=turn_id, timeout_seconds=min(timeout_seconds or self.timeout_seconds, 60))
            all_events.extend(reconciliation.get("events", []))
            return self._finish_reconciled_turn(thread_id=thread_id, turn_id=turn_id, reconciliation=reconciliation, state=state, all_events=all_events)
        state["completion_source"] = "event_stream"
        return self._turn_result(thread_id, turn_id, state, all_events, reconciled=False)

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=10)
            except OSError:
                pass
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for reader in (self._stdout_reader_thread, self._stderr_reader_thread):
            if reader is not None and reader is not threading.current_thread():
                reader.join(timeout=2.0)

    def __enter__(self) -> "PersistentLunaClient":
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
