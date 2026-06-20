"""DrugPipeline — main interface called by experiment.py and run_trial_drug.py.

The agent edits features.py, models.py, calibration.py. This file
orchestrates them into a fit/predict/save/load interface.

Interface contract (do NOT change method signatures — run_trial_drug.py
depends on them):
  DrugPipeline.fit(train_df, val_df, task_name)   → None
  DrugPipeline.predict(test_df, task_name)         → np.ndarray (1D float)
  DrugPipeline.save(path)                          → None
  DrugPipeline.load(path)                          → None
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.features import featurize
from pipeline.models import build_model, fit_model, predict_model, get_task_type
from pipeline.calibration import calibrate


class DrugPipeline:
    """Per-task drug property prediction pipeline."""

    def __init__(self) -> None:
        self._model = None
        self._task_name: str = ""
        self._task_type: str = ""

    def fit(self, train_df: pd.DataFrame, val_x_df: pd.DataFrame | None,
            task_name: str) -> None:
        """Train model on train_df.

        Args:
            train_df:  DataFrame with columns ['Drug', 'Y'] (and optionally 'Drug_ID').
            val_x_df:  Optional val SMILES DataFrame — NO Y column. Passed for
                       potential feature-side use only (e.g. corpus statistics).
                       The baseline ignores it; early stopping uses an internal
                       holdout carved from train_df instead.
            task_name: TDC task name, e.g. 'caco2_wang'.

        Design note: val Y is intentionally absent to prevent the agent from
        memorising validation labels and returning them from predict().
        """
        self._task_name = task_name
        # task_type is a benchmark-data property. The harness (run_trial_drug.py)
        # passes it via MAGENT_TASK_TYPE so the agent venv (no benchmark package) gets
        # the correct value for any benchmark; get_task_type is the TDC fallback.
        self._task_type = os.environ.get("MAGENT_TASK_TYPE") or get_task_type(task_name)

        X_train = featurize(train_df, task_name=task_name)
        y_train = train_df["Y"].values

        model = build_model(task_name, self._task_type)
        self._model = fit_model(model, X_train, y_train, self._task_type)

    def predict(self, test_df: pd.DataFrame, task_name: str) -> np.ndarray:
        """Return predictions for test_df (no Y column expected or used).

        Args:
            test_df:   DataFrame with 'Drug' column (SMILES). No labels.
            task_name: TDC task name (should match what was used in fit).

        Returns:
            1D float array of predictions, length = len(test_df).
        """
        if self._model is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        X_test = featurize(test_df, task_name=task_name)
        raw    = predict_model(self._model, X_test, self._task_type)
        return calibrate(raw, self._task_type)

    def save(self, path: str) -> None:
        """Persist model to disk (pickle)."""
        with open(path, "wb") as f:
            pickle.dump({
                "model":     self._model,
                "task_name": self._task_name,
                "task_type": self._task_type,
            }, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._model     = state["model"]
        self._task_name = state["task_name"]
        self._task_type = state["task_type"]
