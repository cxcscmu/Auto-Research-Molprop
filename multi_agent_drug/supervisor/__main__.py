"""Drug Discovery supervisor CLI shim.

Forwards to `agent_core.supervisor.__main__:main` after ensuring
the Drug TaskAdapter is registered.

Entry point: `python -m multi_agent_drug.supervisor [args...]`.
"""

from __future__ import annotations

import sys

import multi_agent_drug  # noqa: F401  (triggers register_task_adapter)
from agent_core.supervisor.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
