"""Drug Discovery harness configuration shim.

Re-exports the task-agnostic core config. Drug-specific constants:
  * PKG_ROOT      — multi_agent_drug/ directory
  * VENV_PYTHON   — path to .venv_drug interpreter (for local subprocess)
  * TDC_DATA_DIR  — where TDC downloads / caches ADMET datasets
  * DRUG_BASELINE_SCORES — per-task baseline scores used by aggregate_score formula
"""

from __future__ import annotations

import os
from pathlib import Path

from agent_core.harness.config import *           # noqa: F401, F403
from agent_core.harness.config import (           # noqa: F401
    _env_int,
    _load_swarm_config,
    _SWARM_CFG,
    _SCHED_PRIO_CFG,
    _MODEL_CFG,
    _CONTAINER_VIRT_DISABLE,
    _bwrap_pivot_proc_works,
    _detect_container_virt,
)

PKG_ROOT = Path(__file__).resolve().parent.parent

# ── Specialist registry ────────────────────────────────────────────────────
DOER_DOMAINS: tuple[str, ...] = (
    "fphs",   # physchem feature discovery
    "fsub",   # substructure feature discovery
    "lit",    # literature / endpoint mechanism
    "data",   # data quality, scaffold split, imbalance
    "daugm",  # data augmentation via external datasets (data_only / joint)
    "modl",   # model backbone search
    "calib",  # calibration, threshold, uncertainty
    "meta",   # meta analyst
)
ANALYST_DOMAINS: tuple[str, ...] = ()
ALL_DOMAINS: tuple[str, ...] = DOER_DOMAINS + ANALYST_DOMAINS

# Harness Python: full venv with PyTDC (for run_trial_drug.py itself).
VENV_PYTHON = os.environ.get(
    "MAGENT_VENV",
    str(PKG_ROOT.parent / ".venv_drug" / "bin" / "python"),
)

# Agent Python: stripped venv WITHOUT PyTDC (for experiment.py subprocesses).
# Hard isolation: TDC literally cannot be imported in the agent subprocess.
AGENT_PYTHON = os.environ.get(
    "MAGENT_AGENT_VENV",
    str(PKG_ROOT.parent / ".venv_drug_agent" / "bin" / "python"),
)

# TDC ADMET dataset cache directory. TDC downloads on first access and
# caches here. Operator may point this at a pre-populated shared path.
TDC_DATA_DIR = os.environ.get(
    "MAGENT_DRUG_TDC_DATA_DIR",
    str(PKG_ROOT.parent / "tdc_data"),   # 项目内自闭环（原 ~/drug_dev/tdc_data）
)
