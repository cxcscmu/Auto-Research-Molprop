"""ChemBERTa + XGBoost baseline for all 22 TDC ADMET tasks.

Uses the same scaffold split (seed=42, 80/20 from train_val) as the harness
so results are directly comparable to the RDKit baseline.

Molecular representation: mean-pooled last hidden states from
seyonec/ChemBERTa-zinc-base-v1 (768-dim, ZINC-pretrained).

Usage:
    .venv_drug/bin/python baselines/run_pretrained_baseline.py
    .venv_drug/bin/python baselines/run_pretrained_baseline.py --model DeepChem/ChemBERTa-77M-MTR

Output:
    knowledge/chemberta_baseline_scores.json   (per-task metrics)
    baselines/chemberta_results.txt            (human-readable comparison table)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── project root on sys.path so we can import pipeline helpers ────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "multi_agent_drug"))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")


# ── Constants ─────────────────────────────────────────────────────────────────

TDC_DATA_DIR      = os.environ.get("MAGENT_DRUG_TDC_DATA_DIR",
                                   os.path.expanduser("~/drug_dev/tdc_data"))
PKG_ROOT          = _ROOT / "multi_agent_drug"
BASELINE_JSON     = PKG_ROOT / "knowledge" / "baseline_scores.json"
OUT_JSON          = PKG_ROOT / "knowledge" / "chemberta_baseline_scores.json"
OUT_TABLE         = _HERE / "chemberta_results.txt"
VAL_SPLIT_FRAC    = 0.20
SPLIT_SEED        = 42
EMBED_BATCH_SIZE  = 64


# ── Scaffold split (identical to run_trial_drug.py) ───────────────────────────

def scaffold_split(df: pd.DataFrame,
                   val_frac: float = VAL_SPLIT_FRAC,
                   seed: int = SPLIT_SEED):
    rng = np.random.default_rng(seed)
    scaffolds: dict[str, list[int]] = {}
    for i, smi in enumerate(df["Drug"]):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            sca = (MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                   if mol else "")
        except Exception:
            sca = ""
        scaffolds.setdefault(sca, []).append(i)

    groups = list(scaffolds.values())
    rng.shuffle(groups)
    n_val = int(len(df) * val_frac)
    val_idx, train_idx = [], []
    for grp in groups:
        if len(val_idx) < n_val:
            val_idx.extend(grp)
        else:
            train_idx.extend(grp)
    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True))


# ── ChemBERTa embedding ───────────────────────────────────────────────────────

def load_chemberta(model_name: str):
    from transformers import AutoTokenizer, AutoModel
    import torch
    print(f"  Loading {model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_smiles(smiles_list: list[str],
                 tokenizer, model,
                 batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """Mean-pool last hidden states → (n_molecules, hidden_dim)."""
    import torch

    all_embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc)
        # mean pool over non-padding tokens
        mask   = enc["attention_mask"].unsqueeze(-1).float()
        hidden = out.last_hidden_state          # (B, seq_len, hidden)
        emb    = (hidden * mask).sum(1) / mask.sum(1)   # (B, hidden)
        all_embeddings.append(emb.numpy().astype(np.float32))

    return np.vstack(all_embeddings)


# ── Model helpers (mirrors pipeline/models.py) ────────────────────────────────

TASK_TYPES = {
    "caco2_wang": "regression", "hia_hou": "classification",
    "pgp_broccatelli": "classification", "bioavailability_ma": "classification",
    "lipophilicity_astrazeneca": "regression", "solubility_aqsoldb": "regression",
    "bbb_martins": "classification", "ppbr_az": "regression",
    "vdss_lombardo": "regression", "cyp2d6_veith": "classification",
    "cyp3a4_veith": "classification", "cyp2c9_veith": "classification",
    "cyp2d6_substrate_carbonmangels": "classification",
    "cyp3a4_substrate_carbonmangels": "classification",
    "cyp2c9_substrate_carbonmangels": "classification",
    "half_life_obach": "regression", "clearance_microsome_az": "regression",
    "clearance_hepatocyte_az": "regression", "herg": "classification",
    "ames": "classification", "dili": "classification", "ld50_zhu": "regression",
}


def build_xgb(task_type: str):
    import xgboost as xgb
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                  n_jobs=-1, verbosity=0, early_stopping_rounds=30)
    if task_type == "classification":
        return xgb.XGBClassifier(objective="binary:logistic",
                                  eval_metric="auc", **params)
    return xgb.XGBRegressor(objective="reg:squarederror", **params)


def fit_and_predict(model, X_train, y_train, X_val, task_type: str):
    rng = np.random.default_rng(42)
    n_es = max(1, int(len(X_train) * 0.15))
    idx  = rng.permutation(len(X_train))
    X_tr, y_tr = X_train[idx[n_es:]], y_train[idx[n_es:]]
    X_es, y_es = X_train[idx[:n_es]], y_train[idx[:n_es]]
    model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    if task_type == "classification":
        return model.predict_proba(X_val)[:, 1].astype(np.float64)
    return model.predict(X_val).astype(np.float64)


def compute_metric(y_true, y_pred, task_type: str):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        return None
    try:
        if task_type == "classification":
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) < 2:
                return None
            return float(roc_auc_score(y_true, y_pred))
        return float(np.mean(np.abs(y_true - y_pred)))
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main(model_name: str) -> int:
    from tdc.benchmark_group import admet_group

    print(f"ChemBERTa baseline — model: {model_name}", flush=True)
    print(f"TDC data: {TDC_DATA_DIR}", flush=True)

    tokenizer, bert_model = load_chemberta(model_name)

    print("Loading TDC ADMET group ...", flush=True)
    group    = admet_group(path=TDC_DATA_DIR)
    all_tasks = list(group.dataset_names)
    print(f"  {len(all_tasks)} tasks", flush=True)

    rdkit_scores: dict = {}
    if BASELINE_JSON.is_file():
        with open(BASELINE_JSON) as f:
            rdkit_scores = json.load(f)

    results: dict[str, dict] = {}
    t0_total = time.monotonic()

    for idx, task_name in enumerate(all_tasks, 1):
        t0 = time.monotonic()
        task_type = TASK_TYPES.get(task_name, "classification")
        print(f"[{idx:2d}/{len(all_tasks)}] {task_name} ({task_type[:3]}) ...",
              end="  ", flush=True)
        try:
            bm           = group.get(task_name)
            train_val_df = bm["train_val"]
            train_df, val_df = scaffold_split(train_val_df)

            train_smiles = train_df["Drug"].tolist()
            val_smiles   = val_df["Drug"].tolist()
            y_train      = train_df["Y"].values
            y_val        = val_df["Y"].values

            X_train = embed_smiles(train_smiles, tokenizer, bert_model)
            X_val   = embed_smiles(val_smiles,   tokenizer, bert_model)

            # replace any inf/nan with 0
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)

            model    = build_xgb(task_type)
            y_pred   = fit_and_predict(model, X_train, y_train, X_val, task_type)
            metric   = compute_metric(y_val, y_pred, task_type)

            rdkit_metric = rdkit_scores.get(task_name, {}).get("metric")
            if metric is not None and rdkit_metric is not None:
                if task_type == "classification":
                    delta = metric - rdkit_metric
                    pct   = delta / rdkit_metric * 100
                else:
                    delta = rdkit_metric - metric
                    pct   = delta / rdkit_metric * 100
                diff_str = f"Δ={delta:+.4f} ({pct:+.1f}%)"
            else:
                diff_str = "no rdkit ref"

            elapsed = time.monotonic() - t0
            print(f"metric={metric:.4f}  rdkit={rdkit_metric:.4f}  {diff_str}  [{elapsed:.1f}s]",
                  flush=True)

            results[task_name] = {
                "metric":    metric,
                "task_type": task_type,
                "n_train":   len(train_df),
                "n_val":     len(val_df),
            }

        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"CRASH: {exc}  [{elapsed:.1f}s]", flush=True)
            results[task_name] = {"metric": None, "task_type": task_type}

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    lines = [
        f"ChemBERTa baseline  ({model_name})",
        f"{'Task':<42} {'Type':<4} {'RDKit':>8} {'ChemBERTa':>10} {'Δ':>8}",
        "-" * 76,
    ]
    n_better = 0
    valid    = 0
    for task_name in all_tasks:
        r         = results.get(task_name, {})
        metric    = r.get("metric")
        task_type = r.get("task_type", TASK_TYPES.get(task_name, "cls"))
        rdkit_m   = rdkit_scores.get(task_name, {}).get("metric")
        if metric is None:
            lines.append(f"  {task_name:<40} {'cls' if 'cls' in task_type else 'reg':<4} {'':>8} {'CRASH':>10}")
            continue
        valid += 1
        if rdkit_m is not None:
            if task_type == "classification":
                delta = metric - rdkit_m
            else:
                delta = rdkit_m - metric   # positive = ChemBERTa is better (lower MAE)
            if delta > 0:
                n_better += 1
            delta_str = f"{delta:+.4f}"
        else:
            rdkit_m   = float("nan")
            delta_str = "n/a"
        t_abbr = "cls" if task_type == "classification" else "reg"
        lines.append(
            f"  {task_name:<40} {t_abbr:<4} {rdkit_m:>8.4f} {metric:>10.4f} {delta_str:>8}"
        )

    lines += [
        "-" * 76,
        f"Win rate vs RDKit: {n_better}/{valid} tasks",
        f"Total elapsed: {time.monotonic() - t0_total:.1f}s",
    ]
    table = "\n".join(lines)
    print("\n" + table)
    OUT_TABLE.write_text(table)
    print(f"\nWrote {OUT_TABLE}", flush=True)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="seyonec/ChemBERTa-zinc-base-v1",
                   help="HuggingFace model name")
    args = p.parse_args()
    sys.exit(main(args.model))
