"""Supervisor core — coroutine-per-specialist orchestration.

Layout of one supervised run:

    run(specialists, deadline_hours, ...):
        ensure_dirs(); bootstrap baseline if needed
        started = time.monotonic()
        tasks = []
        for i, spec in enumerate(specialists):
            await asyncio.sleep(LAUNCH_STAGGER_S * i)
            tasks.append(asyncio.create_task(_doer_loop(spec, started)))
        watcher = asyncio.create_task(_termination_watcher(started))
        await asyncio.gather(watcher, *tasks, return_exceptions=True)
        write_audit_summary()

Each `_doer_loop(specialist, started)` runs until stop.flag appears:

    while not blackboard.should_stop():
        rec = await DoerClass().run_once()
        _append_audit(rec)
        if rec.error:
            await asyncio.sleep(retry_backoff)
            retry_backoff = min(retry_backoff*2, MAX_RETRY_BACKOFF_S)
        else:
            retry_backoff = BASE_RETRY_BACKOFF_S

The supervisor does not decide what a specialist tries — it only decides
when to spawn and when to cancel. All research state lives on the
blackboard; every new iter reads it fresh.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..agents.base import DoerBase, DoerConfig
from ..harness import baseline_audit, blackboard, config, tracker
from . import termination

_LOG = logging.getLogger("multi_agent.supervisor")

# ── Tunables ─────────────────────────────────────────────────────────────────

LAUNCH_STAGGER_S        = 10.0   # gap between sequential specialist launches
BASE_RETRY_BACKOFF_S    = 30.0   # first retry delay after an SDK error
MAX_RETRY_BACKOFF_S     = 900.0  # cap — prevents runaway back-off on persistent outage
TERMINATION_POLL_S      = 30.0   # watcher cadence
DOER_CANCEL_GRACE_S     = 60.0   # how long to wait for a doer to exit cleanly

# ── Specialist → DoerBase subclass registry (resolved via task adapter) ─────
# Lazy-init: populated once from current_adapter().specialist_classes() at
# first use. register_doer can override entries afterward (for tests).

_DOER_CLASSES: dict[str, type[DoerBase]] = {}


def _ensure_doer_classes() -> dict[str, type[DoerBase]]:
    """Populate _DOER_CLASSES from the adapter on first call; idempotent."""
    if not _DOER_CLASSES:
        from agent_core import current_adapter
        _DOER_CLASSES.update(current_adapter().specialist_classes())
    return _DOER_CLASSES


def register_doer(specialist: str, cls: type[DoerBase],
                  replace: bool = False) -> None:
    """Register (or replace) a DoerBase subclass for one specialist.

    Phase-5 modules register new specialists at import time. Tests pass
    replace=True to override the production class with a fake.
    """
    classes = _ensure_doer_classes()
    if (specialist in classes
            and classes[specialist] is not cls
            and not replace):
        raise ValueError(f"doer for {specialist!r} already registered")
    classes[specialist] = cls


# ── Audit log ────────────────────────────────────────────────────────────────

def _audit_path() -> Path:
    return config.BLACKBOARD_DIR / "supervisor_audit.jsonl"


def _append_audit(entry: dict) -> None:
    """Append one JSONL line to supervisor_audit.jsonl.

    Writes are append-only and single-line-per-entry, so concurrent doer
    loops can all write without coordination (POSIX guarantees atomicity
    of small writes to files opened O_APPEND).
    """
    path = _audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _final_row_summary(final_row) -> dict | None:
    """Compact `final` summary for the audit JSONL — task score field comes
    from the adapter so nc/cifar see their own metric, not val_bpb."""
    if not final_row:
        return None
    from agent_core import current_adapter
    score_field = current_adapter().score_field
    return {
        "exp_id":        final_row.get("exp_id", ""),
        "status":        final_row.get("status", ""),
        score_field:     final_row.get(score_field, ""),
        "delta_vs_best": final_row.get("delta_vs_best", ""),
    }


def _record_to_audit_entry(rec, specialist: str, iter_n: int) -> dict:
    """IterRecord → JSONL-friendly dict."""
    return {
        "specialist": specialist,
        "iter":       iter_n,
        "iter_start": rec.iter_start,
        "iter_end":   rec.iter_end,
        "session_id": rec.session_id,
        "error":      rec.error,
        "tool_calls": rec.tool_calls,
        "usage":      rec.usage,
        "final":      _final_row_summary(rec.final_row),
    }


# ── Per-specialist loop ──────────────────────────────────────────────────────

async def _doer_loop(
    specialist: str,
    started_at: float,
    doer_cfg: Optional[DoerConfig] = None,
) -> None:
    """Run iter after iter for one specialist until stop.flag appears."""
    cls = _ensure_doer_classes().get(specialist)
    if cls is None:
        _LOG.error("no DoerBase subclass registered for %r", specialist)
        return

    cfg = doer_cfg or DoerConfig(specialist=specialist)
    iter_n = 0
    retry_backoff = BASE_RETRY_BACKOFF_S

    while not blackboard.should_stop():
        iter_n += 1
        t_iter = time.monotonic()
        _LOG.info("[%s] iter %d starting", specialist, iter_n)

        try:
            doer = cls(cfg=cfg)
            rec = await doer.run_once()
        except Exception as e:  # noqa: BLE001
            # Defence-in-depth: DoerBase already catches inside run_once,
            # but if the constructor itself throws we don't want the
            # supervisor to die.
            _LOG.exception("[%s] doer failed outside run_once", specialist)
            _append_audit({
                "specialist": specialist,
                "iter":       iter_n,
                "iter_start": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "iter_end":   datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error":      f"supervisor-caught {type(e).__name__}: {e}",
                "final":      None,
            })
            await asyncio.sleep(retry_backoff)
            retry_backoff = min(retry_backoff * 2, MAX_RETRY_BACKOFF_S)
            continue

        _append_audit(_record_to_audit_entry(rec, specialist, iter_n))

        if rec.error:
            _LOG.warning("[%s] iter %d error: %s — backing off %.0fs",
                         specialist, iter_n, rec.error, retry_backoff)
            await asyncio.sleep(retry_backoff)
            retry_backoff = min(retry_backoff * 2, MAX_RETRY_BACKOFF_S)
        else:
            retry_backoff = BASE_RETRY_BACKOFF_S
            status = (rec.final_row or {}).get("status", "no-submit")
            dur = time.monotonic() - t_iter
            _LOG.info("[%s] iter %d done in %.0fs → %s",
                      specialist, iter_n, dur, status)

        # Cheap cooperative check — cancels promptly when the watcher
        # drops stop.flag between iters.
        if blackboard.should_stop():
            break

    _LOG.info("[%s] exiting (ran %d iters)", specialist, iter_n)


# ── Termination watcher ──────────────────────────────────────────────────────

async def _termination_watcher(started_at: float,
                               deadline_hours: float,
                               no_improvement_grace_s: float,
                               max_trials: Optional[int] = None,
                               ) -> termination.TerminationVerdict:
    """Poll the termination predicate until it trips, then return the verdict.

    The verdict-writing side effect (stop.flag + reason) is performed by
    termination.request_stop_if_triggered the first time it trips.
    """
    while True:
        v = termination.request_stop_if_triggered(
            started_at,
            deadline_hours=deadline_hours,
            no_improvement_grace_s=no_improvement_grace_s,
            max_trials=max_trials,
        )
        if v.should_stop:
            _LOG.info("termination: %s (trials_completed=%d)",
                      v.reason, v.trials_completed)
            return v
        if max_trials is not None:
            _LOG.debug("termination check: %d/%d trials, elapsed=%.0fs",
                       v.trials_completed, max_trials, v.elapsed_s)
        await asyncio.sleep(TERMINATION_POLL_S)


# ── Top-level entrypoint ─────────────────────────────────────────────────────

@dataclass(slots=True)
class RunSummary:
    """End-of-run state reported back to the CLI."""
    started_iso:    str
    ended_iso:      str
    elapsed_s:      float
    stop_reason:    str
    specialists:    list[str]
    iters_per_spec: dict[str, int]
    final_best:     Optional[dict]


async def run(
    specialists: Iterable[str],
    *,
    deadline_hours: float = config.DEADLINE_HOURS,
    no_improvement_grace_s: float = config.NO_IMPROVEMENT_GRACE_S,
    max_trials: Optional[int] = None,
    launch_stagger_s: float = LAUNCH_STAGGER_S,
    doer_cfg_overrides: Optional[dict[str, DoerConfig]] = None,
    reset_stale_workdirs: bool = False,
) -> RunSummary:
    """Run the swarm until termination, then return a summary.

    `reset_stale_workdirs=True`: before launching any specialist, wipe
    `workdir_<spec>/train_gpt.py` files whose hash differs from the
    package-root baseline. Next iter's `_stage_workdir` will re-seed
    from the current baseline. Use after a baseline migration (new PR
    seed committed to `multi_agent/train_gpt.py`) when specialists
    should abandon their prior-era edits and start from the new seed.
    """
    config.ensure_dirs()
    classes = _ensure_doer_classes()
    specs = list(dict.fromkeys(specialists))        # dedupe, preserve order
    for s in specs:
        if s not in classes:
            raise ValueError(f"no doer registered for specialist {s!r}")

    overrides = doer_cfg_overrides or {}
    started_at = time.monotonic()
    started_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if max_trials is not None:
        _LOG.info("supervisor starting: specialists=%s max_trials=%d "
                  "deadline=%.1fh(safety) grace=disabled",
                  specs, max_trials, deadline_hours)
    else:
        _LOG.info("supervisor starting: specialists=%s deadline=%.1fh grace=%.0fs",
                  specs, deadline_hours, no_improvement_grace_s)

    # Resolve and surface per-specialist model + sched_priority assignments.
    # `swarm_config.json` is the SINGLE source of truth for both — log them
    # explicitly at startup so a restart is unambiguous about what's live.
    resolved_models = {s: config.model_for(s) for s in specs}
    resolved_priorities = {s: config.sched_priority_for(s) for s in specs}
    _LOG.info("model assignment (from swarm_config.json):")
    for s in specs:
        _LOG.info("  %-6s  model=%-22s  sched_prio=%d",
                  s, resolved_models[s], resolved_priorities[s])

    # Resolve sandbox decision once at startup (probes bwrap if env unset).
    # Cached in config module so all 10 specialist sessions see the same
    # value; logged here so a restart's stderr makes it unambiguous.
    sandbox_disabled = config.should_disable_sandbox()

    # Baseline audit: compare each workdir's train_gpt.py against the
    # package-root baseline. Reports matched / fresh / stale. If the
    # operator passed --reset-stale-workdirs, also wipes stale files so
    # _stage_workdir re-seeds them on first iter. Zero blackboard
    # coupling; safe to run every startup.
    baseline_audit.audit(specs, reset_stale=reset_stale_workdirs)

    # Per-spec resolved max_turns from any doer_cfg_overrides (default
    # DoerConfig() value when no override given). Recorded so a glance
    # at supervisor_audit.jsonl tells us the session-shape used.
    _default_max_turns = DoerConfig(specialist=specs[0]).max_turns if specs else 0
    resolved_max_turns = {
        s: (overrides[s].max_turns if s in overrides else _default_max_turns)
        for s in specs
    }
    _append_audit({
        "event":       "supervisor_start",
        "started":     started_iso,
        "specialists": specs,
        "models":      resolved_models,
        "sched_priorities": resolved_priorities,
        "sandbox":     not sandbox_disabled,
        "no_lineage":  os.environ.get("MAGENT_NO_LINEAGE", "0") == "1",
        "max_turns":   resolved_max_turns,
        "max_trials":  max_trials,
        "state_root":         os.environ.get("MAGENT_LOCAL_ROOT", ""),
        "remote_root":        os.environ.get("MAGENT_REMOTE_ROOT", ""),
        "remote_sync_prefix": os.environ.get("MAGENT_REMOTE_SYNC_PREFIX", ""),
        "sched_name_prefix":     config.active_sched_name_prefix(),
    })

    # ── Launch doers with stagger ────────────────────────────────────────────
    doer_tasks: dict[str, asyncio.Task] = {}
    for i, spec in enumerate(specs):
        if i > 0:
            await asyncio.sleep(launch_stagger_s)
        t = asyncio.create_task(
            _doer_loop(spec, started_at, overrides.get(spec)),
            name=f"doer:{spec}",
        )
        doer_tasks[spec] = t

    # ── Termination watcher ──────────────────────────────────────────────────
    watcher = asyncio.create_task(
        _termination_watcher(started_at, deadline_hours, no_improvement_grace_s,
                             max_trials=max_trials),
        name="termination_watcher",
    )

    # Wait for watcher to declare stop
    verdict = await watcher

    # stop.flag is now present — doer loops will exit at their next
    # `should_stop()` check. We give them a grace period, then cancel
    # any stragglers.
    try:
        await asyncio.wait_for(
            asyncio.gather(*doer_tasks.values(), return_exceptions=True),
            timeout=DOER_CANCEL_GRACE_S,
        )
    except asyncio.TimeoutError:
        _LOG.warning("grace period expired — cancelling %d doer task(s)",
                     sum(1 for t in doer_tasks.values() if not t.done()))
        for t in doer_tasks.values():
            if not t.done():
                t.cancel()
        await asyncio.gather(*doer_tasks.values(), return_exceptions=True)

    # ── Summarise ────────────────────────────────────────────────────────────
    ended_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    iters_per_spec = _count_iters_per_spec()
    summary = RunSummary(
        started_iso=started_iso,
        ended_iso=ended_iso,
        elapsed_s=verdict.elapsed_s,
        stop_reason=verdict.reason,
        specialists=specs,
        iters_per_spec=iters_per_spec,
        final_best=blackboard.read_best(),
    )
    _append_audit({"event": "supervisor_end", **asdict(summary)})
    return summary


def _count_iters_per_spec() -> dict[str, int]:
    """Post-run: count completed iters per specialist from the audit log."""
    out: dict[str, int] = {}
    path = _audit_path()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        spec = rec.get("specialist")
        if spec:
            out[spec] = out.get(spec, 0) + 1
    return out
