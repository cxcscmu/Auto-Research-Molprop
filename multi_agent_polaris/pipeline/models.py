"""Model definitions — agent's secondary edit surface (modl specialist).

Baseline: MapLight's default CatBoost — the TDC ADMET strong baseline
(github.com/maplightrx/MapLight-TDC, arXiv:2310.00174), with NO per-task
hyperparameter tuning. Faithful to MapLight's submission config:
  - Classification (binary): CatBoostClassifier(loss=Logloss), predict_proba[:, 1]
  - Regression:              CatBoostRegressor(loss=MAE) on a *scaled* target
        (offset → optional log10 → StandardScaler), prediction inverse-transformed.
  - Shared: random_strength=2, random_seed=42, default iterations (1000),
        no early stopping (fit on full train).
  - The ONLY per-task choice is the target log-scale flag for 4 skewed
    regression endpoints (vdss / half_life / clearance×2) — a data-distribution
    decision, not hyperparameter tuning, exactly as in MapLight.

Agents (modl / calib) may add: LightGBM, RandomForest, MLP, multitask,
hyperparameter search, target transforms, ensemble heads.

Task type is inferred from task_name using TDC metadata; agents may
also pass it explicitly via the `task_type` kwarg.
"""

from __future__ import annotations

import os

import numpy as np

# TDC ADMET task → type and the MapLight log-scale regression set.
#
# AUTHORITATIVE SOURCE = multi_agent_drug/benchmark_data.py (_TASK_TYPES /
# _LOG_SCALE_TASKS) — imported here so there is a single source of truth. The
# harness venv can import it; the STRIPPED AGENT VENV (.venv_drug_agent, with no
# multi_agent_drug on its path) cannot, so it falls back to the local copies below.
# Reward-authoritative task type goes through the provider in run_trial_drug.py;
# this map only drives the agent's own modelling (build_model's log-scale flag), so
# the fallback is a modelling convenience, not a reward path.
# KEEP THE FALLBACK COPIES IN SYNC with benchmark_data.py.
try:
    from multi_agent_drug.benchmark_data import (
        _TASK_TYPES as TASK_TYPES,
        _LOG_SCALE_TASKS,
    )
except Exception:  # stripped agent venv — no multi_agent_drug on path
    # Source: https://tdcommons.ai/benchmark/admet_group/overview/
    # Verified against TDC ADMET group runtime (2026-05-30).
    TASK_TYPES: dict[str, str] = {
        "caco2_wang":                     "regression",
        "hia_hou":                        "classification",
        "pgp_broccatelli":                "classification",
        "bioavailability_ma":             "classification",
        "lipophilicity_astrazeneca":      "regression",
        "solubility_aqsoldb":             "regression",
        "bbb_martins":                    "classification",
        "ppbr_az":                        "regression",
        "vdss_lombardo":                  "regression",
        "cyp2d6_veith":                   "classification",
        "cyp3a4_veith":                   "classification",
        "cyp2c9_veith":                   "classification",
        "cyp2d6_substrate_carbonmangels": "classification",
        "cyp3a4_substrate_carbonmangels": "classification",
        "cyp2c9_substrate_carbonmangels": "classification",
        "half_life_obach":                "regression",
        "clearance_microsome_az":         "regression",
        "clearance_hepatocyte_az":        "regression",
        "herg":                           "classification",
        "ames":                           "classification",
        "dili":                           "classification",
        "ld50_zhu":                       "regression",
    }
    # MapLight: regression endpoints whose target is log10-scaled before fitting.
    # Skewed, strictly-positive distributions spanning orders of magnitude.
    _LOG_SCALE_TASKS = frozenset({
        "vdss_lombardo",
        "half_life_obach",
        "clearance_hepatocyte_az",
        "clearance_microsome_az",
    })


def get_task_type(task_name: str) -> str:
    """Return 'classification' or 'regression' for a TDC task name."""
    return TASK_TYPES.get(task_name, "classification")


class _TargetScaler:
    """MapLight regression target scaler (mirrors maplight.py `scaler`).

    Pipeline: shift to non-negative (offset) → optional log10(y+1) →
    StandardScaler. inverse_transform undoes all three. Defined at module
    level so a fitted _ScaledCatBoostRegressor stays picklable (pipeline.save).
    """

    def __init__(self, log: bool = False) -> None:
        self.log = log
        self.offset = 0.0
        self._scaler = None

    def fit(self, y) -> "_TargetScaler":
        from sklearn.preprocessing import StandardScaler

        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        self.offset = float(min(float(y.min()), 0.0))
        y = y - self.offset
        if self.log:
            y = np.log10(y + 1.0)
        self._scaler = StandardScaler().fit(y)
        return self

    def transform(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1) - self.offset
        if self.log:
            y = np.log10(y + 1.0)
        return self._scaler.transform(y).reshape(-1)

    def inverse_transform(self, y_scaled) -> np.ndarray:
        y = self._scaler.inverse_transform(
            np.asarray(y_scaled, dtype=np.float64).reshape(-1, 1))
        if self.log:
            y = 10.0 ** y - 1.0
        return (y + self.offset).reshape(-1)


class _ScaledCatBoostRegressor:
    """CatBoostRegressor wrapped with MapLight target scaling.

    Minimal fit/predict surface so pipeline.py treats it like any estimator.
    Module-level (picklable). fit() ignores eval_set/early_stopping kwargs —
    MapLight fits on full train with default iterations and no early stopping.
    """

    def __init__(self, log_scale: bool, **params) -> None:
        from catboost import CatBoostRegressor

        self._scaler = _TargetScaler(log=log_scale)
        self._model = CatBoostRegressor(**params)

    def fit(self, X, y, **kwargs) -> "_ScaledCatBoostRegressor":
        self._scaler.fit(y)
        self._model.fit(X, self._scaler.transform(y))
        return self

    def predict(self, X) -> np.ndarray:
        return self._scaler.inverse_transform(self._model.predict(X))


def build_model(task_name: str, task_type: str | None = None, **kwargs):
    """Build a default CatBoost model — MapLight's TDC ADMET strong baseline.

    No per-task hyperparameter tuning; the same config is applied to all 22
    tasks. The only per-task choice is the regression target log-scale flag
    (4 skewed endpoints), exactly as in MapLight's submission. Returns an
    unfitted estimator (classifier, or scaling regressor wrapper).
    Agents: replace or wrap this function to try LightGBM, RF, MLP, etc.
    """
    from catboost import CatBoostClassifier

    ttype = task_type or get_task_type(task_name)

    # thread_count matches the per-specialist taskset CPU budget (run_trial.sh
    # exports OMP_NUM_THREADS=8) so CatBoost's pool doesn't oversubscribe the
    # 8-core affinity slice. 0/unset → CatBoost default. Operational only —
    # does not affect the fitted model's numerics.
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "0") or 0) or None

    params = dict(
        random_strength=2,           # MapLight submission
        random_seed=42,
        verbose=0,
        thread_count=n_threads,
        allow_writing_files=False,   # don't litter catboost_info/ in the workdir
    )
    params.update(kwargs)

    if ttype == "classification":
        params["loss_function"] = "Logloss"
        return CatBoostClassifier(**params)
    else:
        params["loss_function"] = "MAE"
        return _ScaledCatBoostRegressor(
            log_scale=(task_name in _LOG_SCALE_TASKS), **params)


def fit_model(model, X_train: np.ndarray, y_train: np.ndarray,
              task_type: str, **kwargs):
    """Fit on full train — MapLight uses default iterations, no early stopping.

    Val labels are never passed from the harness (anti-memorisation); MapLight
    does not use a validation holdout for fitting, so none is created here.
    For regression, the wrapper scales the target internally.
    """
    model.fit(X_train, y_train)
    return model


def predict_model(model, X: np.ndarray, task_type: str) -> np.ndarray:
    """Return predictions as 1D float array.

    Classification → predict_proba[:, 1] (probability of positive class).
    Regression     → predict (already inverse-transformed to raw units).
    """
    if task_type == "classification":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1].astype(np.float64)
        return np.asarray(model.predict(X)).ravel().astype(np.float64)
    return np.asarray(model.predict(X)).ravel().astype(np.float64)
