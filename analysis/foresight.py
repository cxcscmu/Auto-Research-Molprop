"""Foresight layer: can task PRIOR attributes (known before running any search)
predict the best intervention axis for each endpoint?

Builds the unified 36-endpoint table across TDC / MoleculeNet / Polaris:
  benchmark, endpoint, n_train (unaugmented feature-axis internal train size),
  task_type, per-axis max norm_improvement, observed best axis (tau=0.005).

n_train is taken from the FEATURE axis only, i.e. the unaugmented internal
training size, so it is a genuine prior attribute (the data axis inflates
n_train by merging external rows). Local-only; reads the pulled run dirs.
"""
import json, glob
from collections import defaultdict

TAU = 0.005
BENCH = {
    "TDC": {
        "feature": "drug_dev_maplight_feature",
        "model":   "drug_dev_maplight_model_only",
        "data":    "drug_dev_maplight_data_v2",
    },
    "MolNet": {
        "feature": "molnet_dev_feature",
        "model":   "molnet_dev_model",
        "data":    "molnet_dev_data",
    },
    "Polaris": {
        "feature": "polaris_dev_feature",
        "model":   "polaris_dev_model",
        "data":    "polaris_dev_data",
    },
}


def per_endpoint_max(d):
    """Max norm_improvement per endpoint over an axis's kept snapshots."""
    best = defaultdict(lambda: -9.9)
    for f in glob.glob(f"{d}/blackboard/snapshots/*/eval/run_seed0.jsonl"):
        try:
            rec = json.loads(open(f).readline())
        except Exception:
            continue
        for t, info in (rec.get("per_task", {}) or {}).items():
            ni = info.get("norm_improvement")
            if ni is not None and ni > best[t]:
                best[t] = ni
    return best


def prior_meta(d):
    """n_train + task_type from the unaugmented (feature) axis."""
    nt, tt = {}, {}
    for f in glob.glob(f"{d}/blackboard/snapshots/*/eval/run_seed0.jsonl"):
        try:
            rec = json.loads(open(f).readline())
        except Exception:
            continue
        for t, info in (rec.get("per_task", {}) or {}).items():
            if t not in nt and info.get("n_train"):
                nt[t] = info["n_train"]
            if t not in tt and info.get("task_type"):
                tt[t] = info["task_type"]
    return nt, tt


def build_rows():
    rows = []
    for bench, groups in BENCH.items():
        mx = {g: per_endpoint_max(d) for g, d in groups.items()}
        nt, tt = prior_meta(groups["feature"])
        for e in sorted(nt, key=lambda x: nt[x]):
            f = mx["feature"].get(e, 0.0)
            m = mx["model"].get(e, 0.0)
            da = mx["data"].get(e, 0.0)
            vals = {"feature": f, "model": m, "data": da}
            w = max(vals, key=vals.get)
            best_axis = w if vals[w] > TAU else "flat"
            rows.append({
                "benchmark": bench, "endpoint": e, "n_train": nt[e],
                "task_type": tt.get(e, "?"),
                "feature": f, "model": m, "data": da, "best_axis": best_axis,
            })
    return rows


if __name__ == "__main__":
    rows = build_rows()
    hdr = f"{'bench':8s}{'endpoint':34s}{'n_tr':>6}{'type':>6}{'feat':>8}{'model':>8}{'data':>8}  best"
    print(hdr); print("-" * len(hdr))
    tally = defaultdict(int)
    for r in rows:
        tally[r["best_axis"]] += 1
        print(f"{r['benchmark']:8s}{r['endpoint']:34s}{r['n_train']:>6}{r['task_type'][:5]:>6}"
              f"{r['feature']:+8.3f}{r['model']:+8.3f}{r['data']:+8.3f}  {r['best_axis']}")
    print("-" * len(hdr))
    print(f"total endpoints: {len(rows)}   best-axis tally: {dict(tally)}")
