"""TdcAdmetDataProvider — TDC ADMET benchmark-data contract.

Single source of truth for the TDC task universe (`_TASK_TYPES` / `_LOG_SCALE_TASKS`)
and the wrapper around TDC's `admet_group` / `admet_metrics`. Lets run_trial_drug.py
stay benchmark-agnostic: the staged runner constructs this by MAGENT_TASK and drives
it through the BenchmarkDataProvider interface.

Import-light: stdlib + lazy `tdc` only — loads inside the harness venv staged runner.
pipeline/models.py imports `_TASK_TYPES` / `_LOG_SCALE_TASKS` from here with a
stripped-agent-venv fallback (that venv has neither this package nor tdc).
"""

from __future__ import annotations

import os
from typing import Any

from agent_core.benchmark_data import BenchmarkDataProvider


# ── TDC task universe — single source of truth ─────────────────────────────────
# (was pipeline/models.py:TASK_TYPES, moved here verbatim). Source:
# https://tdcommons.ai/benchmark/admet_group/overview/ ; verified against the TDC
# ADMET group runtime (2026-05-30). Names come from group.dataset_names.
_TASK_TYPES: dict[str, str] = {
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
# (was pipeline/models.py:_LOG_SCALE_TASKS, moved here verbatim.)
_LOG_SCALE_TASKS = frozenset({
    "vdss_lombardo",
    "half_life_obach",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
})


class TdcAdmetDataProvider(BenchmarkDataProvider):
    """TDC ADMET (22-task) data contract. Wraps admet_group / admet_metrics."""

    def __init__(self) -> None:
        self._group: Any = None

    @property
    def benchmark_name(self) -> str:
        return "tdc_admet"

    def data_dir(self) -> str:
        # Verbatim match of run_trial_drug.py's former TDC_DATA_DIR resolution
        # (run_trial_drug.py:63-64) — keeps the bind-mount target unchanged.
        return os.environ.get("HARNESS_TDC_DATA_DIR",
                              os.path.expanduser("~/drug_dev/tdc_data"))

    def load_group(self) -> Any:
        # Cached: the runner used to call _load_group() once; reuse keeps that.
        if self._group is None:
            from tdc.benchmark_group import admet_group
            self._group = admet_group(path=self.data_dir())
        return self._group

    def task_names(self) -> list[str]:
        return list(self.load_group().dataset_names)

    def expected_n_tasks(self) -> int:
        return 22

    def task_metric(self, task: str) -> str:
        # Verbatim match of run_trial_drug.py's former _task_metric.
        try:
            from tdc.metadata import admet_metrics
            return admet_metrics.get(task, "roc-auc")
        except Exception:
            return "roc-auc"

    def task_type(self, task: str) -> str:
        return _TASK_TYPES.get(task, "classification")

    def log_scale_tasks(self) -> frozenset:
        return _LOG_SCALE_TASKS

    def isolation_block_modules(self) -> tuple[str, ...]:
        return ("tdc",)

    # compute_metric / metric_higher_is_better / normalise are inherited from
    # BenchmarkDataProvider → byte-identical to run_trial_drug.py's former impls.


__all__ = ["TdcAdmetDataProvider", "_TASK_TYPES", "_LOG_SCALE_TASKS"]
