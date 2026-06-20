"""multi_agent_polaris — Polaris (biogen adme-fang) task package (cross-benchmark validation).

Thin wrapper over agent_core + multi_agent_drug's pipeline/runner/experiment (symlinked
in this directory). Registers PolarisTaskAdapter with agent_core on import.

The benchmark-data contract lives in benchmark_data.py (PolarisDataProvider) + loader.py
(PolarisGroup / POLARIS_TASKS); the shared run_trial_drug.py constructs the provider by
MAGENT_TASK=polaris, so the runner stays benchmark-agnostic.
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MAGENT_LOCAL_ROOT", str(_PROJECT_ROOT / "polaris_dev"))

from agent_core import register_task_adapter
from multi_agent_polaris.task_config import PolarisTaskAdapter

register_task_adapter(PolarisTaskAdapter())
