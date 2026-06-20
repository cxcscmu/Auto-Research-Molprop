"""MoleculeNet loader — admet_group-equivalent over DeepChem's public CSVs.

`MOLNET_TASKS` is the single source of truth for the 10 endpoints (source file, SMILES
column, label column, task type, metric, recommended split). `MolNetGroup` mimics TDC's
`admet_group` duck type (`.dataset_names` + `.get(name)->{"train_val","test"}` with
`Drug`/`Y`/`Drug_ID` columns) so the shared `run_trial_drug.py` runner stays
benchmark-agnostic.

Leakage-safe multi-label handling (the central correctness point): each source file's
molecule set is split ONCE into train_val/test (by the dataset's recommended method);
per-task label columns are then NaN-dropped *after* partitioning. So a molecule never
crosses the train_val/test boundary across the sub-tasks that share a file (e.g. the
three Tox21 assays / two SIDER labels) — NaN-drop only removes a row from whichever side
it already fell on.

Splits follow MoleculeNet's recommendations (per-endpoint): ESOL/BACE/HIV use scaffold,
FreeSolv/Tox21/SIDER/ClinTox use random, at the standard 80/10/10 (test = 10%).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# task_name → metadata. task_name matches ^[a-z][a-z0-9_]+$ (runner cache regex).
# Endpoints + split methods confirmed against the live CSVs (2026-06-06).
MOLNET_TASKS: dict[str, dict] = {
    "freesolv":            {"file": "SAMPL.csv",             "smiles": "smiles", "label": "expt",                                     "type": "regression",     "metric": "rmse",    "split": "random"},
    "esol":                {"file": "delaney-processed.csv", "smiles": "smiles", "label": "measured log solubility in mols per litre", "type": "regression",     "metric": "rmse",    "split": "scaffold"},
    "bace":                {"file": "bace.csv",              "smiles": "mol",    "label": "Class",                                    "type": "classification", "metric": "roc-auc", "split": "scaffold"},
    "hiv":                 {"file": "HIV.csv",               "smiles": "smiles", "label": "HIV_active",                               "type": "classification", "metric": "roc-auc", "split": "scaffold"},
    "tox21_nr_ar":         {"file": "tox21.csv.gz",          "smiles": "smiles", "label": "NR-AR",                                    "type": "classification", "metric": "roc-auc", "split": "random"},
    "tox21_sr_mmp":        {"file": "tox21.csv.gz",          "smiles": "smiles", "label": "SR-MMP",                                   "type": "classification", "metric": "roc-auc", "split": "random"},
    "tox21_sr_p53":        {"file": "tox21.csv.gz",          "smiles": "smiles", "label": "SR-p53",                                   "type": "classification", "metric": "roc-auc", "split": "random"},
    "sider_hepatobiliary": {"file": "sider.csv.gz",          "smiles": "smiles", "label": "Hepatobiliary disorders",                  "type": "classification", "metric": "roc-auc", "split": "random"},
    "sider_reproductive":  {"file": "sider.csv.gz",          "smiles": "smiles", "label": "Reproductive system and breast disorders", "type": "classification", "metric": "roc-auc", "split": "random"},
    "clintox_ct_tox":      {"file": "clintox.csv.gz",        "smiles": "smiles", "label": "CT_TOX",                                   "type": "classification", "metric": "roc-auc", "split": "random"},
}

_TEST_FRAC = 0.10   # MoleculeNet standard 80/10/10 → test = 10%
_SPLIT_SEED = 42


class MolNetGroup:
    """admet_group-equivalent: `.dataset_names` + `.get(name)->{"train_val","test"}`."""

    def __init__(self, data_dir, tasks: dict | None = None,
                 test_frac: float = _TEST_FRAC, seed: int = _SPLIT_SEED):
        self._dir = Path(data_dir)
        self._tasks = tasks if tasks is not None else MOLNET_TASKS
        self._test_frac = test_frac
        self._seed = seed
        self._files: dict[str, pd.DataFrame] = {}
        self._splits: dict[tuple, tuple] = {}

    @property
    def dataset_names(self):
        return list(self._tasks.keys())

    def _load_file(self, fname: str) -> pd.DataFrame:
        if fname not in self._files:
            self._files[fname] = pd.read_csv(self._dir / fname)   # pandas auto-decompresses .gz
        return self._files[fname]

    def _outer_split(self, fname: str, split_method: str, smiles_col: str):
        """Split the whole molecule set of `fname` ONCE → (train_val_idx, test_idx).

        Cached per (file, method) and shared across all sub-tasks of the same file, so
        test molecules can never leak into another assay's training set.
        """
        key = (fname, split_method)
        if key not in self._splits:
            df = self._load_file(fname)
            n = len(df)
            if split_method == "scaffold":
                from agent_core.harness.splits import scaffold_partition_indices
                # large = train_val (90%), small = test (10%)
                tv_idx, test_idx = scaffold_partition_indices(df[smiles_col], self._test_frac, self._seed)
            else:  # random
                rng = np.random.default_rng(self._seed)
                perm = rng.permutation(n)
                n_test = int(n * self._test_frac)
                test_idx = list(perm[:n_test])
                tv_idx = list(perm[n_test:])
            self._splits[key] = (tv_idx, test_idx)
        return self._splits[key]

    def get(self, name: str) -> dict:
        meta = self._tasks[name]
        df = self._load_file(meta["file"])
        tv_idx, test_idx = self._outer_split(meta["file"], meta["split"], meta["smiles"])
        smi, lab = meta["smiles"], meta["label"]

        def _slice(idx):
            sub = df.iloc[idx][[smi, lab]].copy()
            sub = sub.dropna(subset=[lab])                 # drop rows missing THIS task's label
            sub = sub.rename(columns={smi: "Drug", lab: "Y"})
            sub.insert(0, "Drug_ID", [f"{name}_{i}" for i in range(len(sub))])
            return sub.reset_index(drop=True)

        return {"train_val": _slice(tv_idx), "test": _slice(test_idx)}


__all__ = ["MOLNET_TASKS", "MolNetGroup"]
