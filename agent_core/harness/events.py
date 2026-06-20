"""Fine-grained event stream for live supervisor observability.

Audit log (supervisor_audit.jsonl) records one entry per *completed* iter.
Event log (events.jsonl) records *within-iter* transitions as they happen:

    submit_trial_called → stage_ok → preflight_ok → sched_push_ok
                        → sched_submit → sched_terminal → sched_pull_ok
                        → classify_done

Both files are append-only JSONL with atomic single-line writes under POSIX
PIPE_BUF (~4 KB), safe for many concurrent producers and lock-free readers.

Each emit is dual-channel:
  * structured JSON line to blackboard/events.jsonl   (for dashboard)
  * logging.INFO via the "multi_agent.events" logger  (for tmux stdout)

This stream is strictly additive. Nothing in the supervisor hot path reads
events.jsonl or depends on a successful write — disk full / transient I/O
errors are swallowed so they can never break a live trial.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

from . import config

_LOG = logging.getLogger("multi_agent.events")


def _events_path() -> Path:
    return config.BLACKBOARD_DIR / "events.jsonl"


def _format_kv(kwargs: dict[str, Any]) -> str:
    """Inline kv summary for the log line; long values truncated."""
    parts: list[str] = []
    for k, v in kwargs.items():
        if v is None:
            continue
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "…"
        parts.append(f"{k}={s}")
    return " ".join(parts)


def emit(spec: str, event: str, **kwargs: Any) -> None:
    """Append a `{ts, spec, event, ...}` line to events.jsonl + log one line.

    `spec` is the specialist key ("arch", "opt", ...). `event` is a short
    lowercase tag ("sched_submit", "classify_done"). Any other kwargs are
    serialized into the JSON line and summarized in the log line. `None`
    values are dropped to keep the wire format compact.
    """
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {k: v for k, v in kwargs.items() if v is not None}
    record = {"ts": ts, "spec": spec, "event": event, **payload}

    try:
        path = _events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass

    kv = _format_kv(payload)
    if kv:
        _LOG.info("[%s] %s · %s", spec, event, kv)
    else:
        _LOG.info("[%s] %s", spec, event)
