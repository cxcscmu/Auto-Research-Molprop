"""multi_agent_drug — Drug Discovery ADMET task package.

Importing this package:
  1. Sets per-task root env defaults — only via setdefault so explicit
     operator overrides via shell env still win.
  2. Registers `DrugTaskAdapter` with `agent_core`.

Drug Discovery uses local submission mode (no SCHED) so there is no
MAGENT_REMOTE_ROOT or MAGENT_REMOTE_SYNC_PREFIX. Trials run as
local subprocesses in .venv_drug.

Idempotent: re-importing does not re-register a fresh adapter.
"""

from __future__ import annotations

import os
from pathlib import Path

# Experiment state (blackboard, workdirs) lives inside the project tree.
# TDC dataset cache stays at its download location (large, not project state).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MAGENT_LOCAL_ROOT",
                      str(_PROJECT_ROOT / "drug_dev"))

from agent_core import register_task_adapter
from multi_agent_drug.task_config import DrugTaskAdapter

register_task_adapter(DrugTaskAdapter())
