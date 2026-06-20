"""Polaris (biogen adme-fang) cold-start baseline calibration.

Runs the unedited MapLight baseline through the shared pipeline once to write per-task
baseline metrics to multi_agent_polaris/knowledge/baseline_scores.json (the normalisation
reference). Same mechanism as multi_agent_drug.calibrate_baseline, retargeted at the
Polaris provider (MAGENT_TASK=polaris) + polaris_data.

Usage:  python -m multi_agent_polaris.calibrate_baseline
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import multi_agent_polaris  # noqa: F401  (registers PolarisTaskAdapter)
from agent_core import current_adapter
from multi_agent_drug.harness.config import VENV_PYTHON   # shared venvs

_PKG_ROOT = Path(__file__).resolve().parent
_POLARIS_DATA_DIR = _PKG_ROOT.parent / "polaris_data"


def _run_baseline() -> dict:
    print("Running Polaris adme-fang baseline pipeline for calibration...")
    python = VENV_PYTHON if VENV_PYTHON != "skip" else sys.executable
    runner = _PKG_ROOT / "run_trial_drug.py"   # symlink → shared benchmark-agnostic runner

    with tempfile.TemporaryDirectory(prefix="polaris_calib_") as tmpdir:
        tmp = Path(tmpdir)
        import shutil
        # copy2/copytree follow the symlinks → real drug experiment.py + pipeline/
        shutil.copy(_PKG_ROOT / "experiment.py", tmp / "experiment.py")
        shutil.copytree(_PKG_ROOT / "pipeline", tmp / "pipeline")

        out_jsonl = tmp / "result.jsonl"
        env = dict(os.environ)
        env["HARNESS_POLARIS_DATA_DIR"] = str(_POLARIS_DATA_DIR)
        env["HARNESS_WORKDIR"]         = str(tmp)
        env["HARNESS_BASELINE_SCORES"] = ""        # bootstrap: no normalisation yet
        env["HARNESS_WALL_LIMIT_S"]    = "5400"    # 90 min headroom
        env["HARNESS_PYTHON"]          = python
        env["HARNESS_ABLATION_MODE"]   = "joint"   # baseline = unmodified pipeline
        env["HARNESS_PKG_ROOT"]        = str(_PKG_ROOT)
        env["MAGENT_TASK"]             = "polaris"
        env["PYTHONUNBUFFERED"]        = "1"

        t0 = time.monotonic()
        result = subprocess.run(
            [python, str(runner), "--out", str(out_jsonl)],
            env=env, capture_output=False, timeout=5600,
        )
        elapsed = time.monotonic() - t0
        if not out_jsonl.is_file():
            print(f"ERROR: runner produced no output (rc={result.returncode})")
            sys.exit(1)
        with open(out_jsonl) as f:
            data = json.loads(f.readline())
        print(f"Baseline run finished in {elapsed:.1f}s")
        return data


def main() -> int:
    if os.environ.get("HARNESS_SINGLE_TASK"):
        print("ERROR: calibrate_baseline must not run with HARNESS_SINGLE_TASK set "
              "(would write a partial baseline_scores.json).")
        return 1

    data = _run_baseline()
    per_task = data.get("per_task", {})
    if not per_task:
        print("ERROR: no per_task results in baseline run")
        return 1

    baseline_scores: dict = {}
    for task_name, info in per_task.items():
        if info.get("status") == "ok" and info.get("val_metric") is not None:
            baseline_scores[task_name] = {
                "metric":    info["val_metric"],
                "task_type": info.get("task_type", "regression"),
            }
    if not baseline_scores:
        print("ERROR: no valid per-task metrics")
        return 1

    EXPECTED = current_adapter().data_provider().expected_n_tasks()
    if len(baseline_scores) < EXPECTED:
        failed = sorted(set(per_task) - set(baseline_scores))
        print(f"ERROR: only {len(baseline_scores)}/{EXPECTED} tasks succeeded. "
              f"Refusing to write an incomplete baseline.\nFailed: {failed}")
        return 1

    baseline_path = _PKG_ROOT / "knowledge" / "baseline_scores.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline_scores, f, indent=2)
    print(f"Wrote {len(baseline_scores)} per-task baseline scores to {baseline_path}")
    for t, v in baseline_scores.items():
        print(f"  {t}: {v['metric']:.4f} ({v['task_type']})")

    print("\nBaseline calibration complete. aggregate_score for baseline = 0.0\n"
          "Launch with: python -m multi_agent_polaris.supervisor --baseline-score 0.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
