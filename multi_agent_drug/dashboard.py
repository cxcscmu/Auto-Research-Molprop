"""Read-only status dashboard for the multi-agent supervisor run.

Strictly non-invasive: reads blackboard/results.tsv, blackboard/best.json,
blackboard/supervisor_audit.jsonl, blackboard/stop.flag. Holds no locks,
writes nothing, imports no supervisor hot-path modules besides
harness.config for path constants.

Safe against concurrent appends: TSV rows and JSONL lines are written one
line at a time via O_APPEND, which is atomic for writes under PIPE_BUF
(~4KB) on POSIX — readers never see a partial row. Malformed last lines
are silently skipped.

Audit granularity is per-iter-completion. For mid-iter visibility the
dashboard additionally reads blackboard/events.jsonl — an append-only
stream of within-iter transitions (stage_ok → preflight_ok → sched_push_ok
→ sched_submit → sched_terminal → sched_pull_ok → classify_done) emitted by
tools/submit.py. The dashboard never writes events.jsonl; the hot path
only ever appends, so the correctness guarantee still holds. For even
finer per-second granularity use `local_run.log`.

Usage:
    python -m multi_agent_drug.dashboard                # refresh every 3s
    python -m multi_agent_drug.dashboard --once         # one snapshot, exit
    python -m multi_agent_drug.dashboard --interval 10  # slower refresh
    python -m multi_agent_drug.dashboard --no-color     # strip ANSI
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _apply_state_root_to_env(argv: list[str]) -> None:
    """Pre-parse --state-root from argv and overwrite MAGENT_LOCAL_ROOT.

    Must run BEFORE `from .harness import config` because that import
    freezes config.LOCAL_ROOT from the env. The package __init__.py may
    have already run an os.environ.setdefault(MAGENT_LOCAL_ROOT, ...) by
    the time this module loads (NC/CIFAR do this; PG does not), but
    setdefault is non-overriding — using os.environ[k] = v here lets
    the CLI value win regardless. Unrecognised argv (e.g. --state-root
    without a value) is silently ignored; argparse below will surface
    the proper error message.
    """
    for i, a in enumerate(argv):
        if a == "--state-root" and i + 1 < len(argv):
            os.environ["MAGENT_LOCAL_ROOT"] = os.path.expanduser(argv[i + 1])
            return
        if a.startswith("--state-root="):
            os.environ["MAGENT_LOCAL_ROOT"] = os.path.expanduser(a.split("=", 1)[1])
            return


_apply_state_root_to_env(sys.argv[1:])

from .harness import config


# ── ANSI style bundle ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Style:
    DIM:    str = ""
    BOLD:   str = ""
    RESET:  str = ""
    GREEN:  str = ""
    YELLOW: str = ""
    RED:    str = ""
    CYAN:   str = ""
    CLEAR:  str = ""


def _make_style(no_color: bool) -> Style:
    if no_color:
        return Style()
    csi = "\x1b["
    return Style(
        DIM    = f"{csi}2m",
        BOLD   = f"{csi}1m",
        RESET  = f"{csi}0m",
        GREEN  = f"{csi}32m",
        YELLOW = f"{csi}33m",
        RED    = f"{csi}31m",
        CYAN   = f"{csi}36m",
        CLEAR  = f"{csi}2J{csi}H",
    )


def _status_color(style: Style, status: str) -> str:
    return {
        "keep":                 style.GREEN,
        "baseline":             style.CYAN,
        "discard":              style.DIM,
        "crash":                style.RED,
        "size_blocked":         style.YELLOW,
        "preflight_crash":      style.YELLOW,
        "train_budget_overrun": style.YELLOW,
        "eval_budget_overrun":  style.YELLOW,
    }.get(status, "")


# ── File readers (all tolerant of partial / missing files) ──────────────────

def _read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    except OSError:
        return []


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return out


def _read_best(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


# ── Time helpers ────────────────────────────────────────────────────────────

def _parse_iso(ts: str) -> Optional[datetime.datetime]:
    if not ts:
        return None
    try:
        return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc
        )
    except ValueError:
        return None


def _humanize_age(dt: Optional[datetime.datetime]) -> str:
    if dt is None:
        return "—"
    now = datetime.datetime.now(datetime.timezone.utc)
    s = int((now - dt).total_seconds())
    if s < 0:
        return dt.strftime("%H:%M:%S")
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m{s%60:02d}s"
    if s < 86400:
        return f"{s//3600}h{(s%3600)//60:02d}m"
    return f"{s//86400}d{(s%86400)//3600:02d}h"


# ── Snapshot assembly ───────────────────────────────────────────────────────

@dataclass
class Snapshot:
    rows:                list[dict]                = field(default_factory=list)
    best:                Optional[dict]            = None
    stop_present:        bool                      = False
    stop_reason:         str                       = ""
    super_started:       Optional[str]             = None
    super_ended:         Optional[str]             = None
    per_spec_counts:     dict[str, Counter]        = field(default_factory=lambda: defaultdict(Counter))
    per_spec_last_row:   dict[str, dict]           = field(default_factory=dict)
    per_spec_last_audit: dict[str, dict]           = field(default_factory=dict)
    per_spec_iters:      Counter                   = field(default_factory=Counter)
    global_counts:       Counter                   = field(default_factory=Counter)
    per_spec_last_event: dict[str, dict]           = field(default_factory=dict)
    # Cumulative count of PR-library tool calls per specialist (for the "pr"
    # column in the per-spec table). Sourced from tool_called events where
    # tool ∈ _PR_LIBRARY_TOOLS. Absent specialists render as 0.
    per_spec_pr_tool_calls: Counter                = field(default_factory=Counter)
    # Cumulative count of WebSearch / WebFetch calls per specialist (for the
    # "web" column). Mirror of pr_tool_calls — used to verify the prompt's
    # web-research encouragement is actually being acted on.
    per_spec_web_tool_calls: Counter               = field(default_factory=Counter)


# Events that close an iter — if the last event is one of these, the
# specialist is idle (between iters), otherwise it's mid-flight.
# session_end fires AFTER classify_done (even for no-submit iters) so it's
# the authoritative idle marker; classify_done is kept for the case where
# submit_trial completed but the SDK session is still wrapping up.
_TERMINAL_EVENTS = frozenset({
    "classify_done",
    "session_end",
    "session_error",
    "preflight_fail",
    "preflight_block",
    "sched_push_fail",
    "sched_submit_fail",
    "sched_pull_fail",
    "sched_wait_timeout",
})

# Pretty stage labels for the In-Flight panel. Ordering below mirrors the
# lifecycle of a healthy iter: session_start → tool_called* →
# submit_trial_called → stage_ok → preflight_ok → sched_push_ok → sched_submit →
# sched_phase(pending/running/…) → sched_terminal → sched_pull_ok → classify_done →
# session_end.
_STAGE_LABELS = {
    # Session-level (base.py)
    "session_start":        "thinking",
    "tool_called":          "using tool",
    "tool_result":          "tool returned",
    "session_error":        "SESSION ERROR",
    "session_end":          "session ended (idle)",
    # submit_trial pipeline (tools/submit.py)
    "submit_trial_called":  "propose handoff",
    "stage_ok":             "staging",
    "preflight_ok":         "preflight ok",
    "preflight_fail":       "preflight FAIL",
    "preflight_block":      "size blocked",
    "sched_push_ok":           "pushed to SharedFS",
    "sched_push_fail":         "push FAIL",
    "sched_submit":            "queued on SCHED",
    "sched_submit_fail":       "sched submit FAIL",
    # SCHED phase-change events (poll loop in _wait_for_job)
    "sched_phase":             "sched phase change",
    "sched_poll_fail":         "sched poll FAIL",
    "sched_job_missing":       "sched job MISSING (gave up)",
    "sched_wait_timeout":      "sched wait TIMEOUT",
    "sched_terminal":          "pod finished",
    "sched_pull_ok":           "pulled results",
    "sched_pull_fail":         "pull FAIL",
    "classify_done":        "classified",
}


# sched_phase emits a generic phase-change event whose `phase` field is what we
# really want the operator to see. These overrides rewrite the stage label to
# match the current lifecycle stage inside SCHED.
_SCHED_PHASE_LABELS = {
    "pending":   "queued on SCHED",
    "running":   "running on GPU",
    "succeeded": "pod finished",
    "failed":    "pod failed",
    "stopped":   "pod stopped",
    "unknown":   "sched phase unknown",
}

# Tools that indicate the specialist engaged with the external PR knowledge
# base this session (see agents/prompts.py + scripts/build_pr_library/). The
# dashboard sums these into the per-spec "pr" column to answer "is the PR
# library actually being used?" at a glance.
_PR_LIBRARY_TOOLS = frozenset({"read_pr_library", "read_pr_source"})
_WEB_TOOLS = frozenset({"WebSearch", "WebFetch"})


def _collect() -> Snapshot:
    snap = Snapshot()
    snap.rows = _read_tsv(config.RESULTS_TSV)
    snap.best = _read_best(config.BEST_JSON)
    snap.stop_present = config.STOP_FLAG.exists()
    if snap.stop_present:
        try:
            snap.stop_reason = config.STOP_FLAG.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    for r in snap.rows:
        spec = r.get("specialist") or "?"
        status = r.get("status") or "?"
        snap.per_spec_counts[spec][status] += 1
        snap.per_spec_last_row[spec] = r
        snap.global_counts[status] += 1

    audit_path = config.BLACKBOARD_DIR / "supervisor_audit.jsonl"
    for a in _read_jsonl(audit_path):
        event = a.get("event")
        if event == "supervisor_start":
            snap.super_started = a.get("started")
            continue
        if event == "supervisor_end":
            snap.super_ended = a.get("ended_iso")
            continue
        spec = a.get("specialist")
        if not spec:
            continue
        snap.per_spec_last_audit[spec] = a
        snap.per_spec_iters[spec] += 1

    # events.jsonl — mid-iter transitions (emitted by tools/submit.py).
    # Append-only, schema is {ts, spec, event, ...}. Last event per spec
    # tells us where that specialist currently is in its pipeline.
    events_path = config.BLACKBOARD_DIR / "events.jsonl"
    for e in _read_jsonl(events_path):
        spec = e.get("spec")
        if not spec:
            continue
        event = e.get("event")
        # Skip `tool_result` for the in-flight stage display — it's a
        # zero-signal echo of the preceding `tool_called` (carries only
        # result_bytes / is_error, no human-actionable state). Keeping
        # it here would parade "tool returned" indefinitely between
        # ToolUseBlocks, hiding which tool the agent is actually using.
        # The event is still written to events.jsonl for paper-trail.
        if event != "tool_result":
            snap.per_spec_last_event[spec] = e
        # Accumulate PR-library tool usage by scanning tool_called events.
        # Old events lack these fields — `.get` keeps the loop forward- and
        # backward-compatible across schema additions.
        if event == "tool_called":
            tool = e.get("tool")
            if tool in _PR_LIBRARY_TOOLS:
                snap.per_spec_pr_tool_calls[spec] += 1
            elif tool in _WEB_TOOLS:
                snap.per_spec_web_tool_calls[spec] += 1

    return snap


# ── Rendering ───────────────────────────────────────────────────────────────

_STATUS_COLUMNS = ("keep", "discard", "crash", "size_blocked")


def _render(snap: Snapshot, style: Style) -> str:
    out: list[str] = []
    now_local = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out.append(
        f"{style.BOLD}Drug Discovery ADMET — multi-agent dashboard{style.RESET}  "
        f"{style.DIM}{now_local}  root={config.LOCAL_ROOT}{style.RESET}"
    )

    # Supervisor state
    if snap.stop_present:
        sup = f"{style.RED}STOPPED{style.RESET}  {snap.stop_reason}"
    elif snap.super_started and not snap.super_ended:
        age = _humanize_age(_parse_iso(snap.super_started))
        sup = f"{style.GREEN}RUNNING{style.RESET}  for {age}  (started {snap.super_started})"
    elif snap.super_ended:
        sup = f"{style.DIM}ended{style.RESET}  at {snap.super_ended}"
    else:
        sup = f"{style.DIM}(no supervisor run recorded — single-specialist mode?){style.RESET}"
    out.append(f"supervisor: {sup}")

    # Best
    if snap.best:
        b = snap.best
        from agent_core import current_adapter
        score_field = current_adapter().score_field
        out.append(
            f"best:       {style.CYAN}exp_{b.get('exp_id','?')}{style.RESET}  "
            f"{score_field}={b.get(score_field,'?')}  "
            f"spec={b.get('specialist','?')}  "
            f"Δ_prev={b.get('delta_vs_prev') or '—'}"
        )
    else:
        out.append(f"best:       {style.DIM}(no keep yet){style.RESET}")

    # Global counts
    gc = snap.global_counts
    total = sum(gc.values())
    parts = [f"total={total}"]
    for st in ("keep", "discard", "crash", "size_blocked", "baseline",
               "preflight_crash", "train_budget_overrun", "eval_budget_overrun"):
        n = gc.get(st, 0)
        if n:
            parts.append(f"{_status_color(style, st)}{st}={n}{style.RESET}")
    out.append("trials:     " + " | ".join(parts))
    out.append("")

    # Per-specialist table. The `pr` column counts read_pr_library +
    # read_pr_source calls; the `web` column counts WebSearch + WebFetch.
    # Together they answer "is the agent actually doing research, and from
    # which source?" at a glance. Absent / pre-telemetry specialists render
    # as "0".
    hdr = (f"{style.BOLD}{'spec':<6} {'iters':>5} {'keep':>4} {'disc':>4} "
           f"{'crash':>5} {'size':>4} {'pr':>4} {'web':>4}  {'last iter':<12} "
           f"{'last trial':<60}{style.RESET}")
    out.append(hdr)
    out.append(style.DIM + "─" * 116 + style.RESET)

    # Only list real specialists — filter out synthetic rows like `baseline`
    # (which would both break column alignment and add no signal: it appears
    # once at bootstrap, is already visible in `trials:` and `best:`).
    known = set(config.ALL_DOMAINS)
    all_specs = sorted(
        known
        | (set(snap.per_spec_counts.keys()) & known)
        | (set(snap.per_spec_last_audit.keys()) & known)
    )
    for spec in all_specs:
        counts = snap.per_spec_counts.get(spec, Counter())
        last_audit = snap.per_spec_last_audit.get(spec)
        last_row = snap.per_spec_last_row.get(spec)
        iters = snap.per_spec_iters.get(spec, 0)

        if last_audit:
            ts = last_audit.get("iter_end") or last_audit.get("iter_start", "")
            age = _humanize_age(_parse_iso(ts))
            if last_audit.get("error"):
                audit_cell = f"{style.RED}err{style.RESET} {age}"
            else:
                audit_cell = f"ok  {age}"
        else:
            audit_cell = f"{style.DIM}—{style.RESET}"

        if last_row:
            st = last_row.get("status", "?")
            col = _status_color(style, st)
            from agent_core import current_adapter
            _sf = current_adapter().score_field
            last_cell = (
                f"exp_{last_row.get('exp_id','?')} "
                f"{col}{st}{style.RESET} "
                f"{_sf}={last_row.get(_sf) or '—'} "
                f"Δ={last_row.get('delta_vs_best') or '—'}"
            )
        else:
            last_cell = f"{style.DIM}—{style.RESET}"

        pr_calls = snap.per_spec_pr_tool_calls.get(spec, 0)
        web_calls = snap.per_spec_web_tool_calls.get(spec, 0)
        out.append(
            f"{spec:<6} {iters:>5} "
            f"{counts.get('keep',0):>4} "
            f"{counts.get('discard',0):>4} "
            f"{counts.get('crash',0):>5} "
            f"{counts.get('size_blocked',0):>4} "
            f"{pr_calls:>4} "
            f"{web_calls:>4}  "
            f"{_pad_ansi(audit_cell, 12)} "
            f"{last_cell}"
        )
    out.append("")

    # In-Flight panel (mid-iter transitions from events.jsonl)
    out.append(f"{style.BOLD}In-Flight (current stage per specialist){style.RESET}")
    inflight_hdr = (f"{style.BOLD}{'spec':<6} {'stage':<20} {'age':<8} "
                    f"{'info':<60}{style.RESET}")
    out.append(inflight_hdr)
    out.append(style.DIM + "─" * 96 + style.RESET)
    for spec in all_specs:
        ev = snap.per_spec_last_event.get(spec)
        if ev is None:
            out.append(f"{spec:<6} {style.DIM}{'(no events yet)':<20} "
                       f"{'—':<8} —{style.RESET}")
            continue
        etype = ev.get("event", "?")
        # sched_phase is special: its stage label depends on the phase payload
        # rather than the event name (pending/running/succeeded/...).
        if etype == "sched_phase":
            phase = str(ev.get("phase") or "").lower()
            label = _SCHED_PHASE_LABELS.get(phase, f"sched phase: {phase or '?'}")
        else:
            label = _STAGE_LABELS.get(etype, etype)
        age = _humanize_age(_parse_iso(ev.get("ts", "")))
        is_terminal = etype in _TERMINAL_EVENTS
        is_fail = (etype.endswith("_fail")
                   or etype == "preflight_block"
                   or etype == "session_error"
                   or etype == "sched_wait_timeout")
        stage_color = (
            style.RED if is_fail
            else style.DIM if is_terminal
            else style.GREEN
        )
        info = _format_event_info(ev)
        out.append(
            f"{spec:<6} "
            f"{_pad_ansi(f'{stage_color}{label}{style.RESET}', 20)} "
            f"{age:<8} "
            f"{style.DIM}{info[:60]}{style.RESET}"
        )
    out.append("")

    # Recent trials tail (5 most recent, oldest first)
    out.append(f"{style.BOLD}Recent trials{style.RESET}")
    recent = snap.rows[-5:]
    if recent:
        for r in recent:
            st = r.get("status", "?")
            col = _status_color(style, st)
            ts = r.get("timestamp", "")
            ts_short = ts[11:19] if len(ts) >= 19 else ts
            hyp = (r.get("hypothesis", "") or "").strip()
            if len(hyp) > 70:
                hyp = hyp[:67] + "…"
            spec_short = (r.get("specialist", "?") or "?")[:5]
            from agent_core import current_adapter
            _sf = current_adapter().score_field
            out.append(
                f"  {ts_short}  exp_{str(r.get('exp_id','?')):<4} "
                f"{spec_short:<5} "
                f"{col}{st:<14}{style.RESET} "
                f"{_sf}={(r.get(_sf) or '—'):<10} "
                f"{style.DIM}{hyp}{style.RESET}"
            )
    else:
        out.append(f"  {style.DIM}(none yet){style.RESET}")

    return "\n".join(out) + "\n"


def _pad_ansi(s: str, width: int) -> str:
    """Left-pad to `width` counting only visible chars (strips ANSI for length)."""
    import re
    visible = re.sub(r"\x1b\[[0-9;]*m", "", s)
    pad = max(0, width - len(visible))
    return s + " " * pad


def _format_event_info(ev: dict) -> str:
    """Pick the 1-2 most relevant fields from an event for the In-Flight info col."""
    etype = ev.get("event", "")
    # Small per-event pickers — each returns a short key=val summary.
    if etype == "session_start":
        return (f"model={ev.get('model','?')} "
                f"max_turns={ev.get('max_turns','?')}")
    if etype == "tool_called":
        return f"tool={ev.get('tool','?')} turn={ev.get('turn','?')}"
    if etype == "session_error":
        return f"err={(ev.get('err') or '')[:40]}"
    if etype == "session_end":
        return (f"turns={ev.get('tool_calls','?')} "
                f"submit={'yes' if ev.get('submit_captured') else 'NO'}"
                + (" err" if ev.get("error") else ""))
    if etype == "submit_trial_called":
        h = ev.get("hypothesis") or ""
        return f"hypothesis={h}" if h else ""
    if etype == "preflight_ok":
        sz = ev.get("size_bytes")
        return f"size={sz}" if sz is not None else ""
    if etype == "preflight_block":
        return f"size={ev.get('size_bytes')} limit={ev.get('limit_bytes')}"
    if etype == "sched_submit":
        return f"job={ev.get('job','?')} priority={ev.get('priority','?')}"
    if etype == "sched_phase":
        prev = ev.get("prev") or "—"
        return f"job={ev.get('job','?')} {prev}→{ev.get('phase','?')}"
    if etype == "sched_wait_timeout":
        return f"job={ev.get('job','?')} phase={ev.get('phase','?')}"
    if etype == "sched_terminal":
        return f"job={ev.get('job','?')} phase={ev.get('phase','?')}"
    if etype == "sched_pull_ok" or etype == "sched_push_ok":
        return f"job={ev.get('job') or ev.get('remote','')}"
    if etype == "classify_done":
        from agent_core import current_adapter
        _sf = current_adapter().score_field
        return (f"exp_{ev.get('exp_id','?')} {ev.get('status','?')} "
                f"{_sf}={ev.get(_sf,'—')} Δ={ev.get('delta','—')}")
    if etype.endswith("_fail"):
        return f"err={(ev.get('err') or '')[:40]}"
    return ""


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Read-only status dashboard for the multi-agent supervisor run.",
    )
    p.add_argument("--interval", type=float, default=3.0,
                   help="Refresh interval in seconds (default: 3).")
    p.add_argument("--once", action="store_true",
                   help="Print one snapshot and exit (useful for scripts).")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI color.")
    # --state-root is pre-parsed at module top via _apply_state_root_to_env;
    # listed here so --help and argparse error messages reflect it.
    p.add_argument("--state-root", type=str, default=None, metavar="PATH",
                   help=(
                       "Override MAGENT_LOCAL_ROOT for this dashboard "
                       "instance. Pre-parsed at module top so config.LOCAL_ROOT "
                       "(frozen at import time) sees the right value. "
                       "Equivalent to MAGENT_LOCAL_ROOT=PATH in the shell."
                   ))
    args = p.parse_args(argv)

    style = _make_style(no_color=args.no_color or not sys.stdout.isatty())

    if args.once:
        sys.stdout.write(_render(_collect(), style))
        sys.stdout.flush()
        return 0

    try:
        prev_lines = 0
        while True:
            content = _render(_collect(), style)
            if style.CLEAR and prev_lines > 0:
                # WSL-friendly in-place refresh: move cursor up prev_lines
                # rows then erase from cursor to end of screen.
                # More reliable than \x1b[2J\x1b[H which causes appending
                # in some WSL terminal emulators.
                sys.stdout.write(f"\x1b[{prev_lines}A\x1b[J")
            sys.stdout.write(content)
            sys.stdout.flush()
            prev_lines = content.count("\n")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
