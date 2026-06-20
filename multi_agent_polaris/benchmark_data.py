"""PolarisDataProvider — Polaris (biogen adme-fang) benchmark-data contract.

Wraps PolarisGroup + POLARIS_TASKS behind the BenchmarkDataProvider interface so the
shared run_trial_drug.py drives Polaris exactly like TDC/MolNet. Import-light
(stdlib + numpy + pandas + lazy agent_core.harness.splits); NO polaris-lib — the data is a
static CSV exported once from the Polaris hub (see data/README.md), so the runner has
zero polaris dependency.

Isolation: data is plain CSV (no package to omit from the agent venv), so the bind-mount
over data_dir() is the primary filesystem control, plus blocking tdc/deepchem/polaris
imports defensively (a curious agent must not `import polaris` to re-fetch the test set).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent_core.benchmark_data import BenchmarkDataProvider
from multi_agent_polaris.loader import POLARIS_TASKS, PolarisGroup


class PolarisDataProvider(BenchmarkDataProvider):
    """Polaris adme-fang (4-endpoint, all regression) data contract. Wraps PolarisGroup."""

    def __init__(self) -> None:
        self._group: Any = None

    @property
    def benchmark_name(self) -> str:
        return "polaris_adme_fang"

    def data_dir(self) -> str:
        d = os.environ.get("HARNESS_POLARIS_DATA_DIR", "")
        if d:
            return d
        pkg = os.environ.get("HARNESS_PKG_ROOT", "")
        if pkg:
            return str(Path(pkg).resolve().parent / "polaris_data")
        return os.path.expanduser("~/polaris_data")

    def load_group(self) -> Any:
        if self._group is None:
            self._group = PolarisGroup(self.data_dir())
        return self._group

    def task_names(self) -> list[str]:
        return list(POLARIS_TASKS.keys())

    def expected_n_tasks(self) -> int:
        return len(POLARIS_TASKS)

    def task_metric(self, task: str) -> str:
        return POLARIS_TASKS[task]["metric"]

    def task_type(self, task: str) -> str:
        return POLARIS_TASKS[task]["type"]

    def log_scale_tasks(self) -> frozenset:
        # adme-fang endpoints are already LOG_* (log10-transformed at the source);
        # no further log scaling.
        return frozenset()

    def isolation_block_modules(self) -> tuple[str, ...]:
        return ("tdc", "deepchem", "polaris")


__all__ = ["PolarisDataProvider"]
