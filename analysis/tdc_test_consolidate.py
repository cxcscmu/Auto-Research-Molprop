"""Consolidate the TDC OFFICIAL-test three-way comparison: baseline / Uni-Mol /
agent(098 best single pipeline), all trained on the same internal-train split and
evaluated on the same TDC official test split. Reports per-endpoint test scores,
who wins on test (direction-aware), the agent's honest test-norm vs baseline-TEST,
and the val->test gap. Read-only."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multi_agent_drug.benchmark_data import TdcAdmetDataProvider  # noqa: E402

SWEEP = json.loads((ROOT / "analysis/tdc_official_test_results.json").read_text())
UNI = json.loads((ROOT / "analysis/tdc_unimol_official_test.json").read_text())
prov = TdcAdmetDataProvider()
HI = prov.metric_higher_is_better


def better(a, b, metric):
    """Return True if a is strictly better than b under the metric direction."""
    if a is None or b is None:
        return None
    return a > b if HI(metric) else a < b


def main():
    tasks = list(SWEEP.keys())
    print(f"{'endpoint':30s}{'metric':>8} {'base_test':>10}{'uni_test':>10}{'agent_test':>11}  {'best':>8} {'A>B?':>5} {'aΔval→test':>11}")
    print("-" * 100)
    tally = {"agent_beats_base": 0, "base_beats_agent": 0,
             "agent_beats_uni": 0, "uni_beats_agent": 0,
             "best_agent": 0, "best_base": 0, "best_uni": 0}
    agg_val, agg_test, n = 0.0, 0.0, 0
    for t in tasks:
        r = SWEEP[t]
        m = r["metric"]
        bt, at = r["baseline_test"], r["agent_test"]
        ut = UNI.get(t, {}).get("unimol_test")
        # winner among the three on test
        cands = {"base": bt, "uni": ut, "agent": at}
        cands = {k: v for k, v in cands.items() if v is not None}
        win = (max if HI(m) else min)(cands, key=cands.get)
        tally[f"best_{win}"] += 1
        # agent vs baseline on test
        ab = better(at, bt, m)
        if ab is True:
            tally["agent_beats_base"] += 1
        elif ab is False:
            tally["base_beats_agent"] += 1
        # agent vs uni on test
        au = better(at, ut, m)
        if au is True:
            tally["agent_beats_uni"] += 1
        elif au is False:
            tally["uni_beats_agent"] += 1
        # agent honest test-norm vs baseline-TEST, and val->test drift
        ntest = prov.normalise(at, bt, m) if (at is not None and bt) else None
        if ntest is not None:
            agg_val += prov.normalise(r["agent_val"], r["baseline_val"], m) or 0.0
            agg_test += ntest
            n += 1
        a_drift = (at - r["agent_val"]) if (at is not None and r["agent_val"] is not None) else None
        uts = f"{ut:.4f}" if ut is not None else "  N/A"
        print(f"{t:30s}{m:>8} {bt:>10.4f}{uts:>10}{at:>11.4f}  {win:>8} "
              f"{('Y' if ab else 'n'):>5} {(f'{a_drift:+.4f}' if a_drift is not None else ''):>11}")
    print("-" * 100)
    print("ON THE OFFICIAL TEST (22 endpoints):")
    print(f"  agent vs baseline : agent wins {tally['agent_beats_base']}, baseline wins {tally['base_beats_agent']}")
    print(f"  agent vs Uni-Mol  : agent wins {tally['agent_beats_uni']}, Uni-Mol wins {tally['uni_beats_agent']} (of {sum(1 for t in tasks if UNI.get(t,{}).get('unimol_test') is not None)} with Uni-Mol)")
    print(f"  best-of-3 on test : agent {tally['best_agent']}, baseline {tally['best_base']}, Uni-Mol {tally['best_uni']}")
    print(f"  agent mean norm vs baseline:  VAL={agg_val/n:+.4f}  ->  TEST={agg_test/n:+.4f}   (n={n})")


if __name__ == "__main__":
    main()
