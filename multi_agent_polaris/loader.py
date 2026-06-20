"""Polaris adme-fang loader — admet_group-equivalent over the exported biogen CSV.

`POLARIS_TASKS` is the single source of truth for the 4 endpoints (source file, SMILES
column, label column, task type, metric, split). `PolarisGroup` mimics TDC's
`admet_group` duck type (`.dataset_names` + `.get(name)->{"train_val","test"}` with
`Drug`/`Y`/`Drug_ID` columns) so the shared `run_trial_drug.py` runner stays
benchmark-agnostic.

Data source: biogen/adme-fang-v1 (Fang 2023, JCIM), downloaded via polaris-lib and
exported once to a static CSV (polaris_data/adme_fang_v1.csv) from the Polaris hub
(see data/README.md) — the runner/provider read the CSV with ZERO polaris-lib
dependency (mirrors molnet_data).

Leakage-safe handling: the molecule set is split ONCE into train_val/test (scaffold);
per-endpoint label columns are then NaN-dropped *after* partitioning, so a molecule never
crosses the train_val/test boundary across the 4 endpoints that share the file (each
endpoint measured a different subset of the 3521 molecules). NaN-drop only removes a row
from whichever side it already fell on.

All 4 endpoints are REGRESSION on already-log10 values (LOG_*); metric = pearson (the
Polaris adme-fang official metric is pearsonr). Scaffold 80:20 (Fang uses random 80:20;
scaffold is stricter and matches our split-hygiene convention across TDC/MolNet).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# task_name → metadata. task_name matches ^[a-z][a-z0-9_]+$ (runner cache regex).
POLARIS_TASKS: dict[str, dict] = {
    "adme_hlm":  {"file": "adme_fang_v1.csv", "smiles": "MOL_smiles", "label": "LOG_HLM_CLint",    "type": "regression", "metric": "pearson", "split": "scaffold"},
    "adme_rlm":  {"file": "adme_fang_v1.csv", "smiles": "MOL_smiles", "label": "LOG_RLM_CLint",    "type": "regression", "metric": "pearson", "split": "scaffold"},
    "adme_mdr1": {"file": "adme_fang_v1.csv", "smiles": "MOL_smiles", "label": "LOG_MDR1-MDCK_ER", "type": "regression", "metric": "pearson", "split": "scaffold"},
    "adme_solu": {"file": "adme_fang_v1.csv", "smiles": "MOL_smiles", "label": "LOG_SOLUBILITY",   "type": "regression", "metric": "pearson", "split": "scaffold"},
}

_TEST_FRAC = 0.20   # Fang 2023 random 80:20; we use scaffold 80:20 (stricter, split hygiene)
_SPLIT_SEED = 42


class PolarisGroup:
    """admet_group-equivalent: `.dataset_names` + `.get(name)->{"train_val","test"}`."""

    def __init__(self, data_dir, tasks: dict | None = None,
                 test_frac: float = _TEST_FRAC, seed: int = _SPLIT_SEED):
        self._dir = Path(data_dir)
        self._tasks = tasks if tasks is not None else POLARIS_TASKS
        self._test_frac = test_frac
        self._seed = seed
        self._files: dict[str, pd.DataFrame] = {}
        self._splits: dict[tuple, tuple] = {}

    @property
    def dataset_names(self):
        return list(self._tasks.keys())

    def _load_file(self, fname: str) -> pd.DataFrame:
        if fname not in self._files:
            self._files[fname] = pd.read_csv(self._dir / fname)
        return self._files[fname]

    def _outer_split(self, fname: str, split_method: str, smiles_col: str):
        """Split the whole molecule set of `fname` ONCE → (train_val_idx, test_idx).

        Cached per (file, method) and shared across all 4 endpoints of the file, so a
        test molecule can never leak into another endpoint's training set.
        """
        key = (fname, split_method)
        if key not in self._splits:
            df = self._load_file(fname)
            n = len(df)
            if split_method == "scaffold":
                from agent_core.harness.splits import scaffold_partition_indices
                tv_idx, test_idx = scaffold_partition_indices(df[smiles_col], self._test_frac, self._seed)
            else:  # random
                rng = np.random.default_rng(self._seed)
                perm = rng.permutation(n)
                n_test = int(n * self._test_frac)
                test_idx = list(perm[:n_test]); tv_idx = list(perm[n_test:])
            self._splits[key] = (tv_idx, test_idx)
        return self._splits[key]

    def get(self, name: str) -> dict:
        meta = self._tasks[name]
        df = self._load_file(meta["file"])
        tv_idx, test_idx = self._outer_split(meta["file"], meta["split"], meta["smiles"])
        smi, lab = meta["smiles"], meta["label"]

        def _slice(idx):
            sub = df.iloc[idx][[smi, lab]].copy()
            sub = sub.dropna(subset=[lab])                 # drop rows missing THIS endpoint's label
            sub = sub.rename(columns={smi: "Drug", lab: "Y"})
            sub.insert(0, "Drug_ID", [f"{name}_{i}" for i in range(len(sub))])
            return sub.reset_index(drop=True)

        return {"train_val": _slice(tv_idx), "test": _slice(test_idx)}


__all__ = ["POLARIS_TASKS", "PolarisGroup"]
