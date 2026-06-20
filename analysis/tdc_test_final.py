"""Final TDC official-test consolidation across 5 configs (baseline / feature /
model / data / combined). All trained on the same internal-train, evaluated on the
same official test. Reports, per endpoint and in aggregate (norm vs baseline-TEST):
  - each single axis on test
  - routed   : per endpoint pick the val-best axis (>0.005), take ITS test score
  - combined : the spliced feature+model+data pipeline on test
  - best5    : per endpoint pick the val-best of {baseline,feature,model,data,combined},
               take ITS test score = the honest achievable upper bound (val-decided)
Routing/selection use VAL only; test never influences any pick. Read-only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multi_agent_drug.benchmark_data import TdcAdmetDataProvider  # noqa: E402

prov = TdcAdmetDataProvider()
A = ROOT / "analysis"
SWEEP = json.loads((A / "tdc_official_test_results.json").read_text())   # baseline + model(agent)
FEAT = json.loads((A / "tdc_test_feature.json").read_text())
DATA = json.loads((A / "tdc_test_data.json").read_text())
_cp = A / "tdc_test_combined.json"
COMB = json.loads(_cp.read_text()) if _cp.exists() else {}
HAVE_COMB = len(COMB) > 0


def norm(x, base, m):
    return prov.normalise(x, base, m) if (x is not None and base) else None


def main():
    tasks = list(SWEEP.keys())
    agg = {k: 0.0 for k in ["feature", "model", "data", "combined", "routed", "best5"]}
    n = 0
    print(f"{'endpoint':28s}{'met':>7} | test-norm vs baseline-test:  {'feat':>7}{'model':>7}{'data':>7}{'comb':>7} | {'routed':>7}{'best5':>7}  pick")
    print("-" * 118)
    for t in tasks:
        m = SWEEP[t]["metric"]
        bval, btest = SWEEP[t]["baseline_val"], SWEEP[t]["baseline_test"]
        cfg = {
            "baseline": (bval, btest),
            "feature": (FEAT[t]["val"], FEAT[t]["test"]),
            "model": (SWEEP[t]["agent_val"], SWEEP[t]["agent_test"]),
            "data": (DATA[t]["val"], DATA[t]["test"]),
        }
        if HAVE_COMB and t in COMB:
            cfg["combined"] = (COMB[t]["val"], COMB[t]["test"])
        # test-norm of each config vs baseline-test
        tn = {k: norm(v[1], btest, m) for k, v in cfg.items()}
        # val-norm of each (for selection), baseline=0
        vn = {k: (norm(v[0], bval, m) or 0.0) for k, v in cfg.items()}
        # routed: best val axis among feature/model/data, >0.005 else baseline
        ax = max(["feature", "model", "data"], key=lambda k: vn[k])
        routed = tn[ax] if vn[ax] > 0.005 else 0.0
        # best: best val among all available configs, take its test-norm
        b5 = max(cfg, key=lambda k: vn[k])
        best5 = tn[b5] if tn[b5] is not None else 0.0
        axes_present = ["feature", "model", "data"] + (["combined"] if "combined" in cfg else [])
        for k in axes_present:
            agg[k] += tn[k] if tn[k] is not None else 0.0
        agg["routed"] += routed
        agg["best5"] += best5
        n += 1

        def f(x):
            return f"{x:+.3f}" if x is not None else "  --"
        print(f"{t:28s}{m:>7} | {'':>20}{f(tn['feature']):>7}{f(tn['model']):>7}{f(tn['data']):>7}{f(tn.get('combined')):>7} | "
              f"{f(routed):>7}{f(best5):>7}  {b5}")
    print("-" * 118)
    print(f"AGGREGATE test-norm vs baseline-test (mean over {n}):")
    keys = ["feature", "model", "data"] + (["combined"] if HAVE_COMB else []) + ["routed", "best5"]
    for k in keys:
        tag = "" if (k != "combined" and (HAVE_COMB or k not in ("best5",))) else ""
        note = " (best5 excl. combined)" if (k == "best5" and not HAVE_COMB) else ""
        print(f"   {k:9s}: {agg[k]/n:+.4f}{note}")
    print("\n(reference: model VAL aggregate was +0.0412; collapsed to model TEST above)")


if __name__ == "__main__":
    main()
