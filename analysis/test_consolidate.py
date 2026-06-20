"""Generalized official-test consolidation for any benchmark. Reads
<bench>_test_{baseline,feature,model,data,combined}.json (each {task:{val,test,metric}}),
all trained on internal-train + scored on the held-out official test. Reports per
endpoint and aggregate (norm vs baseline-TEST): single axes, routed (val-best axis
-> test), combined, best-of-N (val-best config -> test). Selection uses VAL only.
Run with .venv_drug:  python test_consolidate.py --bench {tdc,molnet,polaris}
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import test_eval as te  # noqa: E402

A = ROOT / "analysis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["tdc", "molnet", "polaris"])
    args = ap.parse_args()
    bench = args.bench
    prov = te.get_provider(bench)

    def load(label):
        p = A / f"{bench}_test_{label}.json"
        return json.loads(p.read_text()) if p.exists() else None

    B, F, M, D = (load(x) for x in ["baseline", "feature", "model", "data"])
    C = load("combined")
    HAVE_C = C is not None and len(C) > 0
    if not all([B, F, M, D]):
        missing = [n for n, x in zip("baseline feature model data".split(), [B, F, M, D]) if not x]
        print(f"missing {bench} files: {missing}")
        return

    def nrm(x, base, m):
        return prov.normalise(x, base, m) if (x is not None and base) else None

    tasks = [t for t in B if B[t].get("test") is not None]
    agg = {k: 0.0 for k in ["feature", "model", "data", "combined", "routed", "best"]}
    n = 0
    print(f"=== {bench.upper()} official-test ({len(tasks)} endpoints) — test-norm vs baseline-test ===")
    print(f"{'endpoint':28s}{'met':>8}{'feat':>8}{'model':>8}{'data':>8}{'comb':>8}{'routed':>8}{'best':>8}  pick")
    print("-" * 104)
    for t in tasks:
        m = B[t]["metric"]
        bt, bv = B[t]["test"], B[t]["val"]
        cfg = {"baseline": (bv, bt), "feature": (F[t]["val"], F[t]["test"]),
               "model": (M[t]["val"], M[t]["test"]), "data": (D[t]["val"], D[t]["test"])}
        if HAVE_C and t in C:
            cfg["combined"] = (C[t]["val"], C[t]["test"])
        tn = {k: nrm(v[1], bt, m) for k, v in cfg.items()}
        vn = {k: (nrm(v[0], bv, m) or 0.0) for k, v in cfg.items()}
        ax = max(["feature", "model", "data"], key=lambda k: vn[k])
        routed = tn[ax] if vn[ax] > 0.005 else 0.0
        bk = max(cfg, key=lambda k: vn[k])
        best = tn[bk] if tn[bk] is not None else 0.0
        for k in ["feature", "model", "data"] + (["combined"] if "combined" in cfg else []):
            agg[k] += tn[k] or 0.0
        agg["routed"] += routed
        agg["best"] += best
        n += 1

        def f(x):
            return f"{x:+.3f}" if x is not None else "   --"
        print(f"{t:28s}{m:>8}{f(tn['feature']):>8}{f(tn['model']):>8}{f(tn['data']):>8}"
              f"{f(tn.get('combined')):>8}{f(routed):>8}{f(best):>8}  {bk}")
    print("-" * 104)
    print(f"AGGREGATE (mean over {n}):")
    keys = ["feature", "model", "data"] + (["combined"] if HAVE_C else []) + ["routed", "best"]
    for k in keys:
        note = " (excl. combined)" if (k == "best" and not HAVE_C) else ""
        print(f"   {k:9s}: {agg[k]/n:+.4f}{note}")


if __name__ == "__main__":
    main()
