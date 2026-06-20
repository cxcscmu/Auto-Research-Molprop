"""Recompute the cross-benchmark intervention profile from saved evaluations."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.005

SUITES = {
    "TDC ADMET": {
        "feature": "drug_dev_maplight_feature",
        "model": "drug_dev_maplight_model_only",
        "data": "drug_dev_maplight_data_v2",
    },
    "MoleculeNet": {
        "feature": "molnet_dev_feature",
        "model": "molnet_dev_model",
        "data": "molnet_dev_data",
    },
    "Polaris": {
        "feature": "polaris_dev_feature",
        "model": "polaris_dev_model",
        "data": "polaris_dev_data",
    },
}


def evaluation_records(run_dir: str):
    pattern = "blackboard/snapshots/*/eval/run_seed0.jsonl"
    for path in (ROOT / run_dir).glob(pattern):
        with path.open() as handle:
            line = handle.readline().strip()
        if line:
            yield json.loads(line)


def axis_summary(run_dir: str):
    best_aggregate = float("-inf")
    endpoint_maxima: dict[str, float] = {}
    for record in evaluation_records(run_dir):
        score = record.get("aggregate_score")
        if score is not None:
            best_aggregate = max(best_aggregate, float(score))
        for task, result in (record.get("per_task") or {}).items():
            value = result.get("norm_improvement")
            if value is not None:
                endpoint_maxima[task] = max(
                    endpoint_maxima.get(task, 0.0), float(value)
                )
    return best_aggregate, endpoint_maxima


def best_score_within(run_dir: str, max_exp: int):
    path = ROOT / run_dir / "blackboard/results.tsv"
    best = float("-inf")
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["exp_id"]) > max_exp or not row["aggregate_score"]:
                continue
            best = max(best, float(row["aggregate_score"]))
    return best


def main():
    for suite, axes in SUITES.items():
        summaries = {axis: axis_summary(path) for axis, path in axes.items()}
        tasks = sorted({
            task
            for _, endpoint_maxima in summaries.values()
            for task in endpoint_maxima
        })
        wins = Counter()
        routed = []
        for task in tasks:
            values = {
                axis: endpoint_maxima.get(task, 0.0)
                for axis, (_, endpoint_maxima) in summaries.items()
            }
            winner, value = max(values.items(), key=lambda item: item[1])
            wins[winner if value > THRESHOLD else "none"] += 1
            routed.append(value)

        aggregates = " ".join(
            f"{axis}={summary[0]:.6f}" for axis, summary in summaries.items()
        )
        profile = " ".join(
            f"{axis}={wins[axis]}" for axis in ("model", "data", "feature", "none")
        )
        print(f"{suite}: n={len(tasks)} {aggregates}")
        print(f"  routed={sum(routed) / len(routed):.6f} {profile}")

    tdc = SUITES["TDC ADMET"]
    matched = " ".join(
        f"{axis}={best_score_within(path, 30):.6f}"
        for axis, path in tdc.items()
    )
    print(f"TDC first 30 trials: {matched}")


if __name__ == "__main__":
    main()
