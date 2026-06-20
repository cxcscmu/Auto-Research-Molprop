"""multi_agent_molnet — MoleculeNet task package (P1/P2 cross-benchmark validation).

Thin wrapper over agent_core + multi_agent_drug's pipeline/runner/experiment (symlinked
in this directory). Registers MolNetTaskAdapter with agent_core on import.

The benchmark-data contract lives in benchmark_data.py (MolNetDataProvider) + loader.py
(MolNetGroup / MOLNET_TASKS); the shared run_trial_drug.py constructs the provider by
MAGENT_TASK=molnet, so the runner stays benchmark-agnostic.
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MAGENT_LOCAL_ROOT", str(_PROJECT_ROOT / "molnet_dev"))

from agent_core import register_task_adapter
from multi_agent_molnet.task_config import MolNetTaskAdapter

register_task_adapter(MolNetTaskAdapter())
