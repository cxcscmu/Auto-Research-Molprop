"""MoleculeNet supervisor CLI shim.

Forwards to `agent_core.supervisor.__main__:main` after ensuring the MolNet
TaskAdapter is registered (importing the package triggers register_task_adapter).
This sidesteps agent_core's _TASK_PKG_MAP exactly like multi_agent_drug does.

Entry point: `python -m multi_agent_molnet.supervisor [args...]`.
"""
from __future__ import annotations

import sys

import multi_agent_molnet  # noqa: F401  (triggers register_task_adapter)
from agent_core.supervisor.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
