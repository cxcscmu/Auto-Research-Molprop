"""Termination predicates for the supervisor.

Primary condition (when max_trials is set):
  * total completed trials ≥ max_trials  ← paper budget control

Safety-net conditions (always active):
  * wall-clock elapsed ≥ deadline_hours  ← prevents runaway cost / hang
  * no `keep` for NO_IMPROVEMENT_GRACE_S ← only active when max_trials
    is NOT set (disabled when max_trials is set so the swarm always
    reaches its full budget)

OR-semantics — first trigger wins.

Nothing here mutates the blackboard. `request_stop_if_triggered()`
delegates to blackboard.request_stop so the stop-reason string is
written atomically alongside stop.flag.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from ..harness import blackboard, config, tracker


@dataclass(frozen=True, slots=True)
class TerminationVerdict:
    """Snapshot of termination state at one moment."""
    should_stop:      bool
    reason:           str          # "" when should_stop is False
    elapsed_s:        float
    last_keep_s:      float        # seconds since the most recent keep (or supervisor start)
    trials_completed: int = 0      # total completed trial evaluations at check time


def _count_completed_trials() -> int:
    """Count real trial evaluations, excluding the synthetic baseline row.

    The supervisor bootstrap inserts one row with status='baseline' (exp_id=000)
    before any agent runs. This must not count toward the trial budget so that
    --max-trials 100 means exactly 100 agent-submitted evaluations.
    """
    _SYNTHETIC_STATUSES = {"baseline"}
    return sum(
        1 for r in tracker.read_results()
        if r.get("status", "") not in _SYNTHETIC_STATUSES
    )


def _latest_keep_timestamp_iso() -> Optional[str]:
    """Return the most recent keep row's ISO timestamp, or None."""
    rows = tracker.read_results()
    for r in reversed(rows):
        if r.get("status") == "keep":
            ts = r.get("timestamp", "")
            if ts:
                return ts
    return None


def _iso_to_epoch(iso: str) -> Optional[float]:
    """Parse the exact ISO-8601 shape blackboard writes. None on mismatch."""
    # blackboard writes e.g. "2026-04-21T02:39:07Z" — strip trailing Z and
    # interpret as UTC. datetime.fromisoformat handles this in 3.11+.
    import datetime
    if iso.endswith("Z"):
        iso = iso[:-1]
    try:
        dt = datetime.datetime.fromisoformat(iso).replace(
            tzinfo=datetime.timezone.utc
        )
        return dt.timestamp()
    except ValueError:
        return None


def evaluate(
    started_at_monotonic: float,
    *,
    deadline_hours: float = config.DEADLINE_HOURS,
    no_improvement_grace_s: float = config.NO_IMPROVEMENT_GRACE_S,
    max_trials: Optional[int] = None,
) -> TerminationVerdict:
    """Return a TerminationVerdict for the caller's stop-or-continue decision.

    Termination priority:
      1. max_trials (primary — paper budget control)
      2. deadline_hours (safety net — always active)
      3. no_improvement_grace_s (disabled when max_trials is set, so the
         swarm always reaches its full budget)

    `started_at_monotonic` is time.monotonic() at supervisor start.
    """
    now = time.monotonic()
    elapsed = now - started_at_monotonic
    deadline_s = deadline_hours * 3600.0
    n_trials = _count_completed_trials()

    # ── PRIMARY: trial count ─────────────────────────────────────────────────
    if max_trials is not None and n_trials >= max_trials:
        return TerminationVerdict(
            should_stop=True,
            reason=f"max_trials reached: {n_trials}/{max_trials} trials completed",
            elapsed_s=elapsed,
            last_keep_s=elapsed,
            trials_completed=n_trials,
        )

    # ── SAFETY NET: wall-clock deadline ──────────────────────────────────────
    if elapsed >= deadline_s:
        return TerminationVerdict(
            should_stop=True,
            reason=f"deadline reached: elapsed={elapsed:.0f}s ≥ {deadline_s:.0f}s",
            elapsed_s=elapsed,
            last_keep_s=elapsed,
            trials_completed=n_trials,
        )

    # ── No-improvement grace (only when max_trials is NOT set) ────────────────
    # Disabled under max_trials so the swarm always exhausts its full budget.
    # Without this, a stagnating search would stop early and make budget
    # comparisons between ablation conditions unfair.
    last_keep_s = elapsed   # default: measure from supervisor start
    if max_trials is None:
        last_iso = _latest_keep_timestamp_iso()
        if last_iso is not None:
            last_epoch = _iso_to_epoch(last_iso)
            if last_epoch is not None:
                last_keep_s = max(0.0, time.time() - last_epoch)

        if last_keep_s >= no_improvement_grace_s:
            return TerminationVerdict(
                should_stop=True,
                reason=(f"no-improvement grace exceeded: "
                        f"{last_keep_s:.0f}s since last keep ≥ {no_improvement_grace_s:.0f}s"),
                elapsed_s=elapsed,
                last_keep_s=last_keep_s,
                trials_completed=n_trials,
            )

    return TerminationVerdict(
        should_stop=False,
        reason="",
        elapsed_s=elapsed,
        last_keep_s=last_keep_s,
        trials_completed=n_trials,
    )


def request_stop_if_triggered(started_at_monotonic: float,
                               max_trials: Optional[int] = None,
                               **kw) -> TerminationVerdict:
    """Evaluate + drop stop.flag the first time a condition trips.

    Idempotent: once stop.flag exists, we neither re-check nor re-write it.
    Returns the current verdict either way (callers still use it for the
    end-of-run summary).
    """
    if blackboard.should_stop():
        return TerminationVerdict(
            should_stop=True,
            reason="stop.flag already present",
            elapsed_s=time.monotonic() - started_at_monotonic,
            last_keep_s=0.0,
        )
    v = evaluate(started_at_monotonic, max_trials=max_trials, **kw)
    if v.should_stop:
        blackboard.request_stop(v.reason)
    return v
