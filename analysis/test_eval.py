"""Held-out TEST-set evaluation for a frozen pipeline (compat layer over the 3 benchmarks).

Reuses the existing harness end-to-end, changing only WHICH split we score on:
  - provider.load_group().get(task)["test"] = the held-out test (with labels) that
    the agent search never touched (run_trial_drug.py strips the test Y during search).
  - experiment.py --mode predict applies a fitted model (model_<task>.pkl living in
    --model-dir, a staged workdir whose pipeline/ matches the pkl) to the test SMILES.
  - provider.compute_metric / normalise = the SAME metric + normalisation the search
    used, so the number is directly comparable to the reported val number, but on test.

The core search harness (run_trial_drug.py / experiment.py / pipeline.py) is NOT
modified — this only orchestrates the existing fit/predict surfaces on the test split.
Writes only to a temp dir; never touches any lineage. Run with .venv_drug.

Modes:
  --reuse  : predict only, using an existing model_<task>.pkl in --model-dir. The
             model-dir must be a staged workdir (it carries experiment.py + pipeline/
             matching the pkl). Used to verify the plumbing with an existing weight.
  (fit-then-test, freezing a snapshot config and retraining on the internal-train
   split, is the next step for the full sweep.)
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root (portable: local + AWS)
os.environ.setdefault("HARNESS_TDC_DATA_DIR", str(ROOT / "tdc_data"))
os.environ.setdefault("HARNESS_MOLNET_DATA_DIR", str(ROOT / "molnet_data"))
os.environ.setdefault("HARNESS_POLARIS_DATA_DIR", str(ROOT / "polaris_data"))

PROVIDERS = {
    "tdc": ("multi_agent_drug.benchmark_data", "TdcAdmetDataProvider"),
    "molnet": ("multi_agent_molnet.benchmark_data", "MolNetDataProvider"),
    "polaris": ("multi_agent_polaris.benchmark_data", "PolarisDataProvider"),
}
BASELINE_JSON = {
    "tdc": ROOT / "multi_agent_drug/knowledge/baseline_scores.json",
    "molnet": ROOT / "multi_agent_molnet/knowledge/baseline_scores.json",
    "polaris": ROOT / "multi_agent_polaris/knowledge/baseline_scores.json",
}


def get_provider(bench):
    sys.path.insert(0, str(ROOT))
    mod, cls = PROVIDERS[bench]
    return getattr(importlib.import_module(mod), cls)()


def baseline_for(bench, task):
    p = BASELINE_JSON[bench]
    if not p.exists():
        return None
    rec = json.loads(p.read_text()).get(task)
    if rec is None:
        return None
    return rec["metric"] if isinstance(rec, dict) else rec


def predict_on_test(model_dir, task, test_df, venv_py):
    """Run experiment.py --mode predict from the staged model_dir on test SMILES."""
    model_dir = Path(model_dir)
    pkl = model_dir / f"model_{task}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"no fitted model: {pkl}")
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "test_x.csv"
        out = Path(td) / "pred.csv"
        cols = test_df.copy()
        if "Drug_ID" not in cols.columns:
            cols["Drug_ID"] = range(len(cols))
        cols[["Drug_ID", "Drug"]].to_csv(inp, index=False)  # SMILES only, no Y
        cmd = [venv_py, str(model_dir / "experiment.py"), "--task", task,
               "--mode", "predict", "--input", str(inp), "--output", str(out),
               "--model-dir", str(model_dir)]
        r = subprocess.run(cmd, cwd=str(model_dir), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"experiment.py predict rc={r.returncode}\n{r.stderr[-1200:]}")
        return pd.read_csv(out)["Y"].values


# ── fit-then-test (protocol a): freeze a config, refit on internal-train,
#    predict on BOTH internal-val (must reproduce the reported val) and the
#    held-out test. Split params are the SAME the search used. ───────────────
VAL_SPLIT_FRAC = 0.20
SPLIT_SEED = 42


def _internal_split(train_val):
    sys.path.insert(0, str(ROOT))
    from agent_core.harness.splits import scaffold_partition_indices
    tr_idx, va_idx = scaffold_partition_indices(train_val["Drug"], VAL_SPLIT_FRAC, SPLIT_SEED)
    return (train_val.iloc[tr_idx].reset_index(drop=True),
            train_val.iloc[va_idx].reset_index(drop=True))


def _run_experiment(config_dir, args_list, env, venv_py):
    config_dir = Path(config_dir).resolve()  # absolute, so cwd=config_dir + abs cmd path don't double up
    cmd = [venv_py, str(config_dir / "experiment.py")] + args_list
    r = subprocess.run(cmd, cwd=str(config_dir), capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"experiment.py {args_list[:4]} rc={r.returncode}\n{r.stderr[-1500:]}")


def fit_then_test(config_dir, bench, task, venv_py, data_external_dir=None):
    """Refit `config_dir` on internal-train; score on internal-val AND held-out test.

    If data_external_dir is given (a dir with external_data/<task>.csv), the data-axis
    external data is merged into the train set FIRST, through the harness's own
    leakage-safe filter (_merge_external_data: L1 identity dedup / L2 same-source
    reject / L3 analog filter vs the official test) — faithfully reproducing the
    data-axis augmented training set. The merge touches train only; val/test untouched.
    """
    prov = get_provider(bench)
    group = prov.load_group()
    d = group.get(task)
    train_df, val_df = _internal_split(d["train_val"])
    test_df = d["test"]
    ttype = prov.task_type(task)
    metric = prov.task_metric(task)

    if data_external_dir is not None:
        import shutil
        import multi_agent_drug.run_trial_drug as rtd
        test_index = rtd._build_test_dedup_index(group, [task])
        # The harness removes rejected files so they do not carry into later
        # search trials. Re-evaluation must not mutate the frozen snapshot.
        with tempfile.TemporaryDirectory() as filter_td:
            filter_root = Path(filter_td)
            external_src = Path(data_external_dir) / "external_data"
            if external_src.is_dir():
                shutil.copytree(external_src, filter_root / "external_data")
            train_df, audit = rtd._merge_external_data(
                train_df, val_df, task, filter_root,
                test_index.get(task, {}), ttype)
        print(f"   [data merge] {task}: verdict={audit.get('verdict')} "
              f"n_train_after={len(train_df)}", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MAGENT_TASK_TYPE"] = ttype           # pipeline.fit reads this for task_type
    env["MAGENT_TASK"] = {"tdc": "drug", "molnet": "molnet", "polaris": "polaris"}[bench]

    def smiles_only(df):
        df = df.copy()
        if "Drug_ID" not in df.columns:
            df["Drug_ID"] = range(len(df))
        return df[["Drug_ID", "Drug"]]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mdl = td / "mdl"; mdl.mkdir()
        train_csv = td / "train.csv"; valx = td / "valx.csv"; testx = td / "testx.csv"
        tr = train_df.copy()
        if "Drug_ID" not in tr.columns:
            tr["Drug_ID"] = range(len(tr))
        tr[["Drug_ID", "Drug", "Y"]].to_csv(train_csv, index=False)
        smiles_only(val_df).to_csv(valx, index=False)
        smiles_only(test_df).to_csv(testx, index=False)

        _run_experiment(config_dir, ["--task", task, "--mode", "fit", "--train", str(train_csv),
                                     "--val-x", str(valx), "--model-dir", str(mdl)], env, venv_py)
        vp = td / "vp.csv"; tp = td / "tp.csv"
        _run_experiment(config_dir, ["--task", task, "--mode", "predict", "--input", str(valx),
                                     "--output", str(vp), "--model-dir", str(mdl)], env, venv_py)
        _run_experiment(config_dir, ["--task", task, "--mode", "predict", "--input", str(testx),
                                     "--output", str(tp), "--model-dir", str(mdl)], env, venv_py)
        vpred = pd.read_csv(vp)["Y"].values
        tpred = pd.read_csv(tp)["Y"].values

    vscore = prov.compute_metric(val_df["Y"].values, vpred, metric)
    tscore = prov.compute_metric(test_df["Y"].values, tpred, metric)
    return {"metric": metric, "n_val": len(val_df), "n_test": len(test_df),
            "val": vscore, "test": tscore}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=list(PROVIDERS))
    ap.add_argument("--task", required=True)
    ap.add_argument("--model-dir", help="(--reuse) staged workdir holding model_<task>.pkl")
    ap.add_argument("--config-dir", help="(--fit) staged config dir (snapshot/source) with experiment.py + pipeline/")
    ap.add_argument("--reuse", action="store_true", help="predict with an existing pkl (no retrain)")
    ap.add_argument("--fit", action="store_true", help="refit config on internal-train, score val+test")
    ap.add_argument("--data-external-dir", default=None,
                    help="(data axis) dir with external_data/<task>.csv to merge into train")
    args = ap.parse_args()
    venv_py = str(ROOT / ".venv_drug/bin/python")
    base = baseline_for(args.bench, args.task)

    if args.fit:
        r = fit_then_test(args.config_dir, args.bench, args.task, venv_py,
                          data_external_dir=args.data_external_dir)
        nv = prov_norm(args.bench, r["val"], base, r["metric"])
        nt = prov_norm(args.bench, r["test"], base, r["metric"])
        print(f"[{args.bench}/{args.task}] metric={r['metric']} n_val={r['n_val']} n_test={r['n_test']} "
              f"baseline(val)={base}\n"
              f"   VAL ={_f(r['val'])}  norm={_f(nv,'+')}   (should match reported val)\n"
              f"   TEST={_f(r['test'])}  norm={_f(nt,'+')}")
        return

    prov = get_provider(args.bench)
    test = prov.load_group().get(args.task)["test"]
    metric = prov.task_metric(args.task)
    preds = predict_on_test(args.model_dir, args.task, test, venv_py)
    if len(preds) != len(test):
        raise RuntimeError(f"pred/test length mismatch: {len(preds)} vs {len(test)}")
    score = prov.compute_metric(test["Y"].values, preds, metric)
    norm = prov_norm(args.bench, score, base, metric)
    print(f"[{args.bench}/{args.task}] metric={metric} n_test={len(test)} | "
          f"TEST={_f(score)} | baseline(val)={base} | norm_vs_base={_f(norm,'+')}")


def _f(x, sign=""):
    if x is None:
        return "None"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def prov_norm(bench, score, base, metric):
    if score is None or base is None:
        return None
    return get_provider(bench).normalise(score, base, metric)


if __name__ == "__main__":
    main()
