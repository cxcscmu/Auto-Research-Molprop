"""run_classify.py — Drug Discovery result classifier (staged to workdir).

Reads the JSONL written by run_trial_drug.py, canonicalises status,
and rewrites the file in the shape that agent_core's submit.py
expects (full_eval_results/<workdir_name>/run_seed0.jsonl).

In the Drug Discovery pipeline run_trial_drug.py already writes a
well-formed JSONL directly; this script provides the preflight-crash
injection path (--preflight-status crash) that submit.py uses when
the syntax_check / size_check phase fails before the trial even runs.

Usage:
  python run_classify.py --out <path>                       # normal (reads existing jsonl)
  python run_classify.py --out <path> --preflight-status crash
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True,
                   help="Path to the result JSONL (read and optionally rewritten).")
    p.add_argument("--preflight-status", type=str, default=None,
                   help='Set to "crash" to synthesise a PREFLIGHT_CRASH row.')
    p.add_argument("--kill-reason", type=str, default="",
                   help="Human-readable reason for crash (used with --preflight-status).")
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.preflight_status == "crash":
        rec = {
            "status":          "PREFLIGHT_CRASH",
            "aggregate_score": None,
            "n_tasks_ok":      0,
            "elapsed_s":       0.0,
            "kill_reason":     args.kill_reason or "preflight phase failed",
            "per_task":        {},
        }
        with args.out.open("w") as f:
            json.dump(rec, f)
            f.write("\n")
        return 0

    # Normal path: JSONL already written by run_trial_drug.py — just validate.
    if not args.out.is_file():
        rec = {
            "status":          "CRASH",
            "aggregate_score": None,
            "n_tasks_ok":      0,
            "elapsed_s":       0.0,
            "kill_reason":     "run_trial_drug.py did not produce output JSONL",
            "per_task":        {},
        }
        with args.out.open("w") as f:
            json.dump(rec, f)
            f.write("\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
