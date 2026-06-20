"""BenchmarkDataProvider — lightweight benchmark-data contract for the staged runner.

`run_trial_drug.py` runs in the harness venv (`.venv_drug`) as a *staged copy* and
must NOT import the full `TaskAdapter` — its `specialist_classes()` / `bind_tools()` /
prompt assembly pull in claude-agent-sdk + the orchestration stack. This ABC carves
out only the benchmark-data surface the runner actually needs: group loading, the
task list, per-task metric/type, the metric math, and the isolation knobs.

Concrete subclasses MUST stay import-light — only stdlib + numpy + rdkit + sklearn +
scipy + the benchmark's own database (e.g. `tdc`, or plain CSV). No SDK / supervisor /
agents / MCP-tool imports, so they load cleanly inside the staged runner.

Two entry points:
  - orchestration side (calibrate, write_external_data) → `TaskAdapter.data_provider()`
  - staged runner → constructs the provider directly, selected by `MAGENT_TASK`
    (see `run_trial_drug.py:_load_data_provider`).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Optional


# Metrics where a higher value is better. Everything else (mae, rmse) is
# lower-is-better. Mirrors run_trial_drug.py's former `_HIGHER_IS_BETTER`.
_HIGHER_IS_BETTER_DEFAULT = frozenset({"roc-auc", "pr-auc", "spearman", "pearson"})


class BenchmarkDataProvider(ABC):
    """Benchmark-data contract consumed by the staged trial runner + calibrate.

    Keep subclasses import-light (see module docstring).
    """

    # ── identity / data location ───────────────────────────────────────────────

    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """Short identifier, e.g. 'tdc_admet' / 'moleculenet'."""

    @abstractmethod
    def data_dir(self) -> str:
        """Absolute path to the dataset root.

        Also the Layer-1 bind-mount target that hides ground-truth from the agent
        subprocess (run_trial_drug.py `_run_subprocess`).
        """

    # ── benchmark group / tasks ─────────────────────────────────────────────────

    @abstractmethod
    def load_group(self) -> Any:
        """Return an object mirroring TDC's `admet_group` duck type:

          - `.dataset_names` — iterable of task names
          - `.get(name) -> {"train_val": DataFrame, "test": DataFrame}` whose frames
            carry a 'Drug' (SMILES) and 'Y' (label) column (+ optional 'Drug_ID').

        The runner consumes only those two surfaces, so it stays benchmark-agnostic.
        """

    @abstractmethod
    def task_names(self) -> list[str]:
        """Full ordered task list (drives the per-task loop + aggregation)."""

    def expected_n_tasks(self) -> int:
        """Number of tasks calibrate insists must all succeed before writing
        baseline_scores.json. Default = len(task_names()); TDC overrides to 22."""
        return len(self.task_names())

    @abstractmethod
    def task_metric(self, task: str) -> str:
        """Official metric name: 'mae' | 'spearman' | 'roc-auc' | 'pr-auc' | 'rmse' | 'pearson'."""

    @abstractmethod
    def task_type(self, task: str) -> str:
        """'classification' or 'regression'."""

    @abstractmethod
    def log_scale_tasks(self) -> frozenset:
        """Regression endpoints whose target is log10-scaled before fitting
        (MapLight's skewed strictly-positive set for TDC; empty for MolNet)."""

    # ── isolation ───────────────────────────────────────────────────────────────

    def isolation_block_modules(self) -> tuple[str, ...]:
        """Module names whose import is blocked inside the agent subprocess via the
        Layer-2 PYTHONPATH stub. TDC: ('tdc',); MolNet: ('tdc', 'deepchem')."""
        return ()

    # ── metric math (shared concrete impl; subclasses rarely override) ───────────

    def metric_higher_is_better(self, metric: str) -> bool:
        return metric in _HIGHER_IS_BETTER_DEFAULT

    def compute_metric(self, y_true, y_pred, metric: str) -> Optional[float]:
        """Internal-validation metric matching the benchmark's official metric.

        Returns None on failure (non-finite values, single-class labels for
        AUROC/PR-AUC, etc.). Verbatim port of run_trial_drug.py's former
        `_compute_metric`, with an added 'rmse' branch for MoleculeNet regression
        (dead code for TDC, which never requests rmse).
        """
        import numpy as np
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
            return None
        try:
            if metric == "mae":
                return float(np.mean(np.abs(y_true - y_pred)))
            if metric == "rmse":
                return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            if metric == "spearman":
                from scipy.stats import spearmanr
                rho = spearmanr(y_true, y_pred).correlation
                return float(rho) if rho is not None and np.isfinite(rho) else None
            if metric == "pearson":
                # Polaris adme-fang official metric is pearsonr. Needs non-constant
                # inputs (constant → undefined correlation → nan).
                if np.std(y_true) == 0 or np.std(y_pred) == 0:
                    return None
                from scipy.stats import pearsonr
                r = pearsonr(y_true, y_pred)[0]
                return float(r) if r is not None and np.isfinite(r) else None
            # roc-auc / pr-auc need at least two classes present in y_true.
            if len(np.unique(y_true)) < 2:
                return None
            if metric == "roc-auc":
                from sklearn.metrics import roc_auc_score
                return float(roc_auc_score(y_true, y_pred))
            if metric == "pr-auc":
                from sklearn.metrics import average_precision_score
                return float(average_precision_score(y_true, y_pred))
            return None
        except Exception:
            return None

    def normalise(self, metric: float, baseline: float, metric_name: str) -> Optional[float]:
        """Normalise metric relative to baseline. Returns None if baseline is
        0 / non-finite. Direction from metric_higher_is_better (mae/rmse are
        lower-is-better). Verbatim port of run_trial_drug.py's former `_normalise`.
        """
        if baseline is None or not math.isfinite(baseline) or baseline == 0.0:
            return None
        if self.metric_higher_is_better(metric_name):
            # higher better → positive improvement is good.
            return (metric - baseline) / abs(baseline)
        else:
            # mae / rmse: lower better → positive normalised = reduction in error.
            return (baseline - metric) / abs(baseline)


__all__ = ["BenchmarkDataProvider", "_HIGHER_IS_BETTER_DEFAULT"]
