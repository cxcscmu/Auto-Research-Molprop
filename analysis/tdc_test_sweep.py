"""TDC official-test sweep: refit baseline + best single agent pipeline on the
internal-train split, score on internal-val (sanity, must reproduce reported val)
AND the held-out TDC OFFICIAL test split, for all 22 endpoints.

Agent config = 098_meta (model axis, best single-pipeline aggregate on val). This
is one frozen pipeline, not per-endpoint snapshot cherry-picking (which would
re-introduce winner's curse). Baseline = the unmodified source MapLight pipeline.

Uni-Mol on the official test is a separate predict-only step (GPU venv); see
tdc_unimol_test.py.

Resumable: endpoints already in the results json are skipped. Run with .venv_drug.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import test_eval as te  # noqa: E402

ROOT = te.ROOT
AGENT_CFG = str(ROOT / "drug_dev_maplight_model_only/blackboard/snapshots/098_meta")
BASE_CFG = str(ROOT / "multi_agent_drug")
VENV = str(ROOT / ".venv_drug/bin/python")
OUT = ROOT / "analysis/tdc_official_test_results.json"


def main():
    prov = te.get_provider("tdc")
    tasks = prov.task_names()
    rows = json.loads(OUT.read_text()) if OUT.exists() else {}
    for i, t in enumerate(tasks, 1):
        if t in rows and rows[t].get("agent_test") is not None:
            print(f"[{i}/{len(tasks)}] SKIP {t} (done)", flush=True)
            continue
        metric = prov.task_metric(t)
        rep_base = te.baseline_for("tdc", t)
        print(f"[{i}/{len(tasks)}] {t} ({metric}) ...", flush=True)
        try:
            b = te.fit_then_test(BASE_CFG, "tdc", t, VENV)
        except Exception as e:
            print(f"   baseline FAIL: {str(e)[:300]}", flush=True)
            b = {"val": None, "test": None, "metric": metric}
        try:
            a = te.fit_then_test(AGENT_CFG, "tdc", t, VENV)
        except Exception as e:
            print(f"   agent FAIL: {str(e)[:300]}", flush=True)
            a = {"val": None, "test": None, "metric": metric}
        rows[t] = {
            "metric": metric,
            "reported_baseline_val": rep_base,
            "baseline_val": b["val"], "baseline_test": b["test"],
            "agent_val": a["val"], "agent_test": a["test"],
        }
        OUT.write_text(json.dumps(rows, indent=2))
        print(f"   baseline: val={b['val']} test={b['test']} | "
              f"agent: val={a['val']} test={a['test']}", flush=True)
    print(f"@@@@ TDC TEST SWEEP DONE @@@@ wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
