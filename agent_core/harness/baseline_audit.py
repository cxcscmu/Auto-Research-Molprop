"""Baseline audit — detect workdir <> package-root baseline mismatches.

At supervisor startup, hash `multi_agent/train_gpt.py` (the canonical
seed) and each existing `workdir_<spec>/train_gpt.py` (agent's current
file). Any workdir whose hash differs from the baseline is "stale" —
left over from a previous baseline era that a recent commit / operator
edit has since replaced at the package root.

By default we only REPORT. Wiping stale train_gpt.py files (so that
`tools.submit._stage_workdir` re-seeds them from the new baseline on
the next iter) requires explicit operator opt-in via
`--reset-stale-workdirs`.

Design notes
------------
* Zero blackboard coupling. We only touch `train_gpt.py` under
  `multi_agent/` and each `workdir_<spec>/`. `results.tsv`, `best.json`,
  `tree.tsv`, `events.jsonl`, `snapshots/` are all untouched. Historical
  data survives every audit run.
* Runs once at supervisor startup (core.run), never per-iter. Hashing
  a ~86 KB file 11 times is microseconds; negligible.
* `fresh` (workdir has no train_gpt.py yet — e.g. first supervisor run
  on a new pod) is a silent, expected state: `_stage_workdir` will seed
  it on first iter. We just note it in the audit log.

Contract with `_stage_workdir`
------------------------------
`_stage_workdir` (tools/submit.py) seeds only when
`workdir/train_gpt.py` is MISSING. So deleting the file here is both
necessary AND sufficient to trigger a re-seed — we don't need to touch
any other workdir content.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import config


_LOG = logging.getLogger("multi_agent.supervisor")

_BASELINE_FILE = "train_gpt.py"
_SHORT_HASH_CHARS = 12


@dataclass
class WorkdirStatus:
    """One specialist's workdir baseline state.

    stale=True iff the workdir has a train_gpt.py whose hash differs from
    the package-root baseline. `exists=False` means the workdir either
    doesn't exist or has never been seeded — a benign fresh state.
    """
    specialist: str
    path: Path
    exists: bool
    sha: str | None
    size: int | None
    stale: bool


def _sha256_of(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def audit(
    specialists: Iterable[str],
    *,
    reset_stale: bool = False,
) -> list[WorkdirStatus]:
    """Hash-compare baseline vs each workdir's train_gpt.py; log a report.

    If `reset_stale` is True, delete the train_gpt.py of each stale
    workdir so the next iter's `_stage_workdir` re-seeds it from the
    package-root baseline.

    Returns the per-specialist status list for programmatic consumers
    (currently informational only).
    """
    from agent_core import current_adapter
    adapter = current_adapter()
    baseline_path = adapter.pkg_root / adapter.baseline_filename
    baseline_sha = _sha256_of(baseline_path)
    if baseline_sha is None:
        _LOG.warning("[baseline audit] %s is missing — skipping audit",
                     baseline_path)
        return []

    baseline_size = baseline_path.stat().st_size
    _LOG.info(
        "[baseline audit] %s  sha=%s  (%d bytes)",
        baseline_path, baseline_sha[:_SHORT_HASH_CHARS], baseline_size,
    )

    statuses: list[WorkdirStatus] = []
    baseline_filename = adapter.baseline_filename
    for spec in specialists:
        wd = config.workdir_for(spec)
        target = wd / baseline_filename
        if not target.is_file():
            statuses.append(WorkdirStatus(
                specialist=spec, path=target,
                exists=False, sha=None, size=None, stale=False,
            ))
            continue
        sha = _sha256_of(target)
        try:
            size = target.stat().st_size
        except OSError:
            size = None
        stale = (sha is not None) and (sha != baseline_sha)
        statuses.append(WorkdirStatus(
            specialist=spec, path=target,
            exists=True, sha=sha, size=size, stale=stale,
        ))

    matched = [s for s in statuses if s.exists and not s.stale]
    fresh = [s for s in statuses if not s.exists]
    stale = [s for s in statuses if s.stale]

    if matched:
        _LOG.info(
            "[baseline audit]  %d spec(s) match baseline: %s",
            len(matched),
            ", ".join(s.specialist for s in matched),
        )
    if fresh:
        _LOG.info(
            "[baseline audit]  %d spec(s) fresh (will seed on first iter): %s",
            len(fresh),
            ", ".join(s.specialist for s in fresh),
        )

    if not stale:
        return statuses

    _LOG.warning(
        "[baseline audit] ⚠ %d spec(s) STALE (differ from baseline):",
        len(stale),
    )
    for s in stale:
        _LOG.warning(
            "[baseline audit]    %-6s  sha=%s  (%s bytes)",
            s.specialist,
            (s.sha or "—")[:_SHORT_HASH_CHARS],
            s.size if s.size is not None else "?",
        )

    if reset_stale:
        wiped = 0
        for s in stale:
            try:
                s.path.unlink()
                wiped += 1
                _LOG.warning("[baseline audit]  wiped %s", s.path)
            except OSError as e:
                _LOG.error("[baseline audit]  failed to wipe %s: %s",
                           s.path, e)
        _LOG.warning(
            "[baseline audit] %d stale workdir(s) reset; "
            "next iter will re-seed each from the baseline.",
            wiped,
        )
    else:
        _LOG.warning(
            "[baseline audit] To re-seed stale workdirs from the current "
            "baseline, restart supervisor with --reset-stale-workdirs. "
            "Otherwise specialists will continue editing their existing "
            "(non-baseline) train_gpt.py."
        )

    return statuses
