"""Val-vs-test trajectory for a lineage: evaluate each (sampled) kept snapshot on
BOTH internal-val and the held-out official test, over all endpoints, to see where
the val and test curves diverge (overfitting onset). Writes per-snapshot per-endpoint
raw metrics; norm-aggregation + plotting done locally. Run with .venv_drug on AWS.

  --lineage  e.g. drug_dev_maplight_model_only
  --bench    tdc/molnet/polaris
  --sample   take every Nth kept snapshot (1 = all). First+last always included.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_eval as te  # noqa: E402

ROOT = te.ROOT
VENV = str(ROOT / ".venv_drug/bin/python")


def kept_snapshots(lineage):
    d = ROOT / lineage / "blackboard" / "snapshots"
    snaps = sorted([p for p in d.glob("[0-9]*_*") if (p / "pipeline").exists()],
                   key=lambda p: int(p.name.split("_")[0]))
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", required=True)
    ap.add_argument("--bench", default="tdc")
    ap.add_argument("--sample", type=int, default=2)
    ap.add_argument("--data-external", action="store_true",
                    help="data axis: merge each snapshot's own external_data")
    args = ap.parse_args()

    snaps = kept_snapshots(args.lineage)
    idx = sorted(set([0, len(snaps) - 1] + list(range(0, len(snaps), args.sample))))
    chosen = [snaps[i] for i in idx]
    print(f"{args.lineage}: {len(snaps)} kept, evaluating {len(chosen)}: "
          f"{[s.name for s in chosen]}", flush=True)

    prov = te.get_provider(args.bench)
    tasks = prov.task_names()
    out = ROOT / f"analysis/{args.bench}_traj_{Path(args.lineage).name}.json"
    traj = json.loads(out.read_text()) if out.exists() else {}
    for si, snap in enumerate(chosen, 1):
        exp = snap.name
        if exp in traj and len(traj[exp]) >= len(tasks):
            print(f"[{si}/{len(chosen)}] SKIP {exp}", flush=True)
            continue
        print(f"[{si}/{len(chosen)}] {exp} ...", flush=True)
        rec = traj.get(exp, {})
        ext = str(snap) if args.data_external else None
        for t in tasks:
            if t in rec:
                continue
            try:
                r = te.fit_then_test(str(snap), args.bench, t, VENV, data_external_dir=ext)
                rec[t] = {"metric": r["metric"], "val": r["val"], "test": r["test"]}
            except Exception as e:
                print(f"     FAIL {t}: {str(e)[:160]}", flush=True)
                rec[t] = {"metric": prov.task_metric(t), "val": None, "test": None}
            traj[exp] = rec
            out.write_text(json.dumps(traj, indent=2))
        print(f"   {exp} done ({sum(1 for v in rec.values() if v['test'] is not None)}/{len(tasks)})", flush=True)
    print(f"@@@@ TRAJ DONE @@@@ {out}", flush=True)


if __name__ == "__main__":
    main()
