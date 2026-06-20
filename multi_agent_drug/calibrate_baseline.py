"""Drug Discovery cold-start baseline calibration.

Run the unedited baseline through the live pipeline once before launching
the swarm to calibrate the aggregate_score starting point. Also writes
per-task baseline metrics to knowledge/baseline_scores.json for use by
run_trial_drug.py's normalisation formula.

Usage:
    python -m multi_agent_drug.calibrate_baseline

The script runs the baseline pipeline in a temp workdir (not PKG_ROOT),
computes per-task internal val metrics, then writes:
  - knowledge/baseline_scores.json  (per-task baseline metrics for normalisation)

After calibration, aggregate_score for the baseline is defined as 0.0
(it is the reference point). All subsequent trial scores are normalised
improvements relative to these per-task baseline metrics.

After calibration, launch the swarm with:
    python -m multi_agent_drug.supervisor --baseline-score <measured_score>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import multi_agent_drug  # noqa: F401  (registers adapter)
from agent_core import register_task_adapter
from multi_agent_drug.task_config import DrugTaskAdapter
from multi_agent_drug.harness.config import TDC_DATA_DIR, VENV_PYTHON, PKG_ROOT


def _run_baseline() -> dict:
    """Run baseline pipeline in a temp workdir (not PKG_ROOT) to avoid polluting source."""
    print("Running baseline pipeline for calibration...")

    python = VENV_PYTHON if VENV_PYTHON != "skip" else sys.executable
    runner  = PKG_ROOT / "run_trial_drug.py"

    with tempfile.TemporaryDirectory(prefix="drug_calib_") as tmpdir:
        tmp = Path(tmpdir)

        # Stage experiment.py + pipeline/ into temp workdir so model_*.pkl
        # go there, not into the source tree.
        import shutil
        shutil.copy(PKG_ROOT / "experiment.py", tmp / "experiment.py")
        shutil.copytree(PKG_ROOT / "pipeline", tmp / "pipeline")

        out_jsonl = tmp / "result.jsonl"

        env = dict(os.environ)
        env["HARNESS_TDC_DATA_DIR"]    = TDC_DATA_DIR
        env["HARNESS_WORKDIR"]         = str(tmp)
        env["HARNESS_BASELINE_SCORES"] = ""   # bootstrap: no normalisation yet
        env["HARNESS_WALL_LIMIT_S"]    = "3600"
        env["HARNESS_PYTHON"]          = python
        env["HARNESS_ABLATION_MODE"]   = "joint"  # baseline measures unmodified pipeline
        # run_trial_drug.py now imports its benchmark-data provider; it needs the
        # package root (to put the project root on sys.path) and the task selector.
        env["HARNESS_PKG_ROOT"]        = str(PKG_ROOT)
        env["MAGENT_TASK"]             = "drug"
        env["PYTHONUNBUFFERED"]        = "1"

        t0 = time.monotonic()
        result = subprocess.run(
            [python, str(runner), "--out", str(out_jsonl)],
            env=env, capture_output=False, timeout=3700,
        )
        elapsed = time.monotonic() - t0

        if not out_jsonl.is_file():
            print(f"ERROR: runner did not produce output (rc={result.returncode})")
            sys.exit(1)

        with open(out_jsonl) as f:
            data = json.loads(f.readline())

        print(f"Baseline run finished in {elapsed:.1f}s")
        return data


def main() -> int:
    # Guard: single-task debug mode must not overwrite the 22-task baseline.
    if os.environ.get("HARNESS_SINGLE_TASK"):
        print(
            f"ERROR: calibrate_baseline must not run with HARNESS_SINGLE_TASK="
            f"'{os.environ['HARNESS_SINGLE_TASK']}' — this would overwrite "
            f"knowledge/baseline_scores.json with a single-task result and "
            f"corrupt the 22-task reward normalization.\n"
            f"Unset HARNESS_SINGLE_TASK before running calibrate_baseline."
        )
        return 1

    data = _run_baseline()

    per_task = data.get("per_task", {})
    if not per_task:
        print("ERROR: no per_task results in baseline run")
        return 1

    # Extract per-task baseline metrics.
    baseline_scores: dict = {}
    for task_name, info in per_task.items():
        if info.get("status") == "ok" and info.get("val_metric") is not None:
            baseline_scores[task_name] = {
                "metric":    info["val_metric"],
                "task_type": info.get("task_type", "classification"),
            }

    if not baseline_scores:
        print("ERROR: no valid per-task metrics")
        return 1

    # Hard fail if baseline is incomplete — partial results must not become
    # source of truth. Tasks without a baseline get norm=None → 0.0, which
    # silently corrupts aggregate_score for all 22-task reward calculations.
    from agent_core import current_adapter
    EXPECTED_TASKS = current_adapter().data_provider().expected_n_tasks()
    if len(baseline_scores) < EXPECTED_TASKS:
        failed = sorted(set(per_task) - set(baseline_scores))
        print(
            f"ERROR: only {len(baseline_scores)}/{EXPECTED_TASKS} tasks succeeded. "
            f"Refusing to overwrite baseline_scores.json with incomplete results.\n"
            f"Failed tasks: {failed}\n"
            f"Fix the pipeline for these tasks and re-run calibrate_baseline."
        )
        return 1

    # Write baseline_scores.json to knowledge/.
    baseline_path = PKG_ROOT / "knowledge" / "baseline_scores.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_scores, f, indent=2)
    print(f"Wrote per-task baseline scores to {baseline_path}")

    print(f"\nPer-task baseline metrics ({len(baseline_scores)} tasks):")
    for t, v in baseline_scores.items():
        print(f"  {t}: {v['metric']:.4f} ({v['task_type']})")

    # The baseline IS the reference point, so its aggregate_score = 0.0.
    # Subsequent trials compute normalised improvement over these per-task
    # metrics; the baseline itself has zero improvement by definition.
    print(
        "\nBaseline calibration complete. aggregate_score for baseline = 0.0\n"
        "\nNow launch the swarm with:\n"
        "  python -m multi_agent_drug.supervisor --baseline-score 0.0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
