"""Cold-start baseline calibration.

Replaces `adapter.baseline_score_default` (a placeholder guess) with a
score actually measured on this pod. Run ≥1 baseline trial through the
live pipeline however your task usually does it (subprocess, sched job,
notebook), read the score(s), then feed them in here:

    python -m multi_agent_<task>.calibrate_baseline \\
        --score 0.301 [--score 0.296 --score 0.305] [--note "h100 pod, seed 0/1/2"]

The script:
  1. Aborts if blackboard already has rows (don't overwrite real state).
  2. Averages the supplied scores; prints stddev when N≥2.
  3. Writes blackboard's `baseline` row + best.json with the measured value.
  4. Logs the inputs to <blackboard>/calibration.log so reviewers can audit.

Why operator-driven: each task's submission_mode (local subprocess, sched
job, sched notebook) already has a tested run path. Reproducing that here
would duplicate ~1000 lines (verify_candidate.py); the cold-start
helper has no business owning that machinery — it's a one-shot
bootstrap, not a runtime component.
"""

from __future__ import annotations

import argparse
import datetime
import math
import statistics
import sys
from typing import Optional


def _finite_float(s: str) -> float:
    """argparse type: float, but reject nan/inf to avoid poisoning best.json."""
    v = float(s)
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"score must be finite, got {s!r}")
    return v

from . import current_adapter
from .harness import blackboard, config, tracker


def _emit_calibration_log(scores: list[float], avg: float, note: str) -> None:
    log = config.BLACKBOARD_DIR / "calibration.log"
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    score_field = current_adapter().score_field
    lines = [
        f"# {ts} baseline calibration",
        f"task_pkg     = {current_adapter().pkg_root.name}",
        f"score_field  = {score_field}",
        f"n_trials     = {len(scores)}",
        f"scores       = {', '.join(f'{s:.6f}' for s in scores)}",
        f"avg          = {avg:.6f}",
    ]
    if len(scores) >= 2:
        lines.append(f"stddev       = {statistics.stdev(scores):.6f}")
    if note:
        lines.append(f"note         = {note}")
    lines.append("")
    log.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="calibrate_baseline", description=__doc__)
    p.add_argument(
        "--score", action="append", type=_finite_float, required=True, metavar="SCORE",
        help="Measured baseline score from one trial. Pass multiple times to "
             "average (recommended: ≥3 to estimate seed noise). Rejects "
             "nan/inf.",
    )
    p.add_argument("--note", default="",
                   help="Free-form provenance note (seeds, pod, date).")
    args = p.parse_args(argv)

    adapter = current_adapter()
    score_field = adapter.score_field

    rows = tracker.read_results()
    if rows:
        existing = blackboard.read_best() or {}
        kind = "baseline" if all(r.get("status") == "baseline" for r in rows) else "swarm"
        print(f"ERROR: blackboard already has {len(rows)} {kind} row(s) at "
              f"{score_field}={existing.get(score_field, '?')}. "
              f"To re-calibrate, wipe the blackboard manually: "
              f"`rm -rf {config.BLACKBOARD_DIR}` (loses ALL state).",
              file=sys.stderr)
        return 2

    avg = statistics.fmean(args.score)
    if not math.isfinite(avg):                       # belt + suspenders
        print(f"ERROR: averaged score {avg!r} is non-finite", file=sys.stderr)
        return 2
    n = len(args.score)
    sigma = statistics.stdev(args.score) if n >= 2 else 0.0
    print(f"calibrating: {n} trial(s), avg {score_field}={avg:.6f}"
          + (f" (σ={sigma:.6f})" if n >= 2 else ""))

    hypothesis = (
        f"calibrated baseline ({n} trial{'s' if n != 1 else ''}, "
        f"avg={avg:.6f}" + (f", σ={sigma:.6f}" if n >= 2 else "") + ")"
    )
    blackboard.bootstrap_from_baseline({
        score_field:     f"{avg:.6f}",
        "hypothesis":    hypothesis,
        "snapshot_path": "",
    })
    _emit_calibration_log(args.score, avg, args.note)
    print(f"wrote {config.BEST_JSON}")
    print(f"wrote {config.BLACKBOARD_DIR / 'calibration.log'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
