"""MolNetDataProvider — MoleculeNet benchmark-data contract.

Wraps MolNetGroup + MOLNET_TASKS behind the BenchmarkDataProvider interface so the
shared run_trial_drug.py drives MoleculeNet exactly like TDC. Import-light
(stdlib + numpy + pandas + lazy agent_core.harness.splits); no SDK — loads cleanly in
the harness venv staged runner.

Isolation note: MoleculeNet data is plain CSV (no package to omit from the agent venv),
so the venv layer gives MolNet no protection — the bind-mount over data_dir() is the
primary filesystem control, plus blocking `tdc`/`deepchem` imports defensively.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent_core.benchmark_data import BenchmarkDataProvider
from multi_agent_molnet.loader import MOLNET_TASKS, MolNetGroup


class MolNetDataProvider(BenchmarkDataProvider):
    """MoleculeNet (10-endpoint) data contract. Wraps MolNetGroup."""

    def __init__(self) -> None:
        self._group: Any = None

    @property
    def benchmark_name(self) -> str:
        return "moleculenet"

    def data_dir(self) -> str:
        d = os.environ.get("HARNESS_MOLNET_DATA_DIR", "")
        if d:
            return d
        # Fallback: <project_root>/molnet_data via HARNESS_PKG_ROOT (= multi_agent_molnet/).
        pkg = os.environ.get("HARNESS_PKG_ROOT", "")
        if pkg:
            return str(Path(pkg).resolve().parent / "molnet_data")
        return os.path.expanduser("~/molnet_data")

    def load_group(self) -> Any:
        if self._group is None:
            self._group = MolNetGroup(self.data_dir())
        return self._group

    def task_names(self) -> list[str]:
        return list(MOLNET_TASKS.keys())

    def expected_n_tasks(self) -> int:
        return len(MOLNET_TASKS)

    def task_metric(self, task: str) -> str:
        return MOLNET_TASKS[task]["metric"]

    def task_type(self, task: str) -> str:
        return MOLNET_TASKS[task]["type"]

    def log_scale_tasks(self) -> frozenset:
        # ESOL is already log-solubility; FreeSolv is signed hydration ΔG — neither is
        # the skewed strictly-positive shape MapLight log-scales. No log scaling.
        return frozenset()

    def isolation_block_modules(self) -> tuple[str, ...]:
        return ("tdc", "deepchem")


__all__ = ["MolNetDataProvider"]
