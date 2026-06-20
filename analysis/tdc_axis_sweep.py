"""Generic single-axis TDC official-test sweep: refit one frozen config on the
internal-train split, score internal-val (reproduction check) + official test, for
all 22 endpoints. Writes analysis/tdc_test_<label>.json {task:{val,test,metric}}.

  --config-dir   staged snapshot dir (experiment.py + pipeline/)
  --label        output key, e.g. feature / data
  --data-external-dir  (data axis) dir holding external_data/<task>.csv to merge
                       into train via the harness leakage-safe filter

Resumable. Run with .venv_drug.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import test_eval as te  # noqa: E402

ROOT = te.ROOT
VENV = str(ROOT / ".venv_drug/bin/python")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="tdc", choices=["tdc", "molnet", "polaris"])
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--data-external-dir", default=None)
    args = ap.parse_args()
    out = ROOT / f"analysis/{args.bench}_test_{args.label}.json"

    prov = te.get_provider(args.bench)
    tasks = prov.task_names()
    N = len(tasks)
    rows = json.loads(out.read_text()) if out.exists() else {}
    for i, t in enumerate(tasks, 1):
        if t in rows and rows[t].get("test") is not None:
            print(f"[{i}/{N}] SKIP {t}", flush=True)
            continue
        print(f"[{i}/{N}] {t} ...", flush=True)
        try:
            r = te.fit_then_test(args.config_dir, args.bench, t, VENV,
                                 data_external_dir=args.data_external_dir)
            rows[t] = {"metric": r["metric"], "val": r["val"], "test": r["test"]}
            print(f"   {t}: val={r['val']} test={r['test']}", flush=True)
        except Exception as e:
            print(f"   FAIL {t}: {str(e)[:300]}", flush=True)
            rows[t] = {"metric": prov.task_metric(t), "val": None, "test": None}
        out.write_text(json.dumps(rows, indent=2))
    print(f"@@@@ {args.bench}/{args.label} SWEEP DONE @@@@ {out}", flush=True)


if __name__ == "__main__":
    main()
