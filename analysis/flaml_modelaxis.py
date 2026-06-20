"""Non-agent AutoML (FLAML) baseline on the MODEL axis, for the benchmark where the
agent's model axis gains are largest (Polaris). Isolates "model+HP search" so we can
ask: are the agent's model-axis gains reachable by a standard AutoML, or do they need
the LLM?

Fair setup — identical to what the agent's model_only axis saw:
  * SAME frozen MapLight features (features.featurize(df, task_name=None), 2563-dim).
  * SAME scaffold split (frac 0.20 / seed 42) for internal train/val; SAME official test.
  * FLAML selects the model on the SAME internal-val split (eval_method=holdout),
    optimizing the SAME metric (pearson), within the SAME model family
    (lgbm / xgboost / catboost), under a matched search budget (max_iter).
Then the chosen model is scored once on val + official test, normalised vs baseline.

  python analysis/flaml_modelaxis.py --bench polaris --budget 30 --seed 42
Writes analysis/<bench>_flaml_modelaxis.json {task:{metric,val,test,best_estimator}}.
Run with .venv_drug.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import test_eval as te  # noqa: E402

# baseline (frozen) MapLight featurizer — same module the baseline + model axis use
FEAT_DIR = {
    "polaris": ROOT / "polaris_dev_model/blackboard/snapshots/001_modl/pipeline",
    "molnet": ROOT / "molnet_dev_model/blackboard/snapshots",  # filled below if needed
    "tdc": ROOT / "drug_dev_maplight_model_only/blackboard/snapshots/006_modl/pipeline",
}


def pearson_loss(X_val, y_val, estimator, labels, X_train, y_train,
                 weight_val=None, weight_train=None, *args, **kwargs):
    pred = np.asarray(estimator.predict(X_val))
    if np.std(pred) == 0 or np.std(y_val) == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(np.asarray(y_val), pred)[0, 1])
    return -r, {"pearson": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="polaris")
    ap.add_argument("--budget", type=int, default=30, help="FLAML max_iter (matched to agent trials)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale-target", action="store_true",
                    help="apply the same target transform the baseline uses "
                         "(log1p if y>0 & skewed, then z-score); fairer for regression")
    args = ap.parse_args()
    suffix = "_scaled" if args.scale_target else ""

    feat_dir = FEAT_DIR[args.bench]
    sys.path.insert(0, str(feat_dir))
    import features as F  # noqa: E402
    from agent_core.harness.splits import scaffold_partition_indices  # noqa: E402
    from flaml import AutoML  # noqa: E402

    prov = te.get_provider(args.bench)
    group = prov.load_group()
    out = ROOT / f"analysis/{args.bench}_flaml_modelaxis{suffix}.json"
    rows = json.loads(out.read_text()) if out.exists() else {}

    for t in prov.task_names():
        if t in rows and rows[t].get("test") is not None:
            print(f"SKIP {t}", flush=True)
            continue
        d = group.get(t)
        tv, test = d["train_val"], d["test"]
        tr_idx, va_idx = scaffold_partition_indices(tv["Drug"], 0.20, 42)
        train, val = tv.iloc[tr_idx], tv.iloc[va_idx]
        metric = prov.task_metric(t)
        print(f"==== {t} (metric={metric}, n_train={len(train)}) ====", flush=True)
        Xtr = F.featurize(train, "Drug", None)
        Xva = F.featurize(val, "Drug", None)
        Xte = F.featurize(test, "Drug", None)
        ytr = train["Y"].values.astype(float)
        yva = val["Y"].values.astype(float)

        # Fair target transform (match baseline's _ScaledCatBoostRegressor): optional
        # log1p for positive skewed targets, then z-score. CatBoost trains differently
        # on the transformed target; pearson on inverse-transformed preds is the metric.
        use_log = bool(args.scale_target and (ytr > 0).all() and
                       abs(float(((ytr - ytr.mean())**3).mean() / (ytr.std()**3 + 1e-9))) > 1.0)

        def fwd(y):
            y = np.log1p(y) if use_log else y
            return (y - mu) / sd

        def inv(z):
            y = z * sd + mu
            return np.expm1(y) if use_log else y

        if args.scale_target:
            base = np.log1p(ytr) if use_log else ytr
            mu, sd = float(base.mean()), float(base.std() or 1.0)
            ytr_in, yva_in = fwd(ytr), fwd(yva)
        else:
            ytr_in, yva_in = ytr, yva

        am = AutoML()
        am.fit(Xtr, ytr_in, X_val=Xva, y_val=yva_in,
               task="regression", metric=pearson_loss, eval_method="holdout",
               estimator_list=["lgbm", "xgboost", "catboost"],
               max_iter=args.budget, seed=args.seed, verbose=0)
        if args.scale_target:
            vp, tp = inv(am.predict(Xva)), inv(am.predict(Xte))
        else:
            vp, tp = am.predict(Xva), am.predict(Xte)
        vs = prov.compute_metric(val["Y"].values, vp, metric)
        ts = prov.compute_metric(test["Y"].values, tp, metric)
        rows[t] = {"metric": metric, "val": vs, "test": ts,
                   "best_estimator": am.best_estimator}
        print(f"   {t}: val={vs:.4f} test={ts:.4f} best={am.best_estimator}", flush=True)
        out.write_text(json.dumps(rows, indent=2))
    print(f"@@@@ FLAML {args.bench} DONE @@@@ {out}", flush=True)


if __name__ == "__main__":
    main()
