"""Drug Discovery agent runner shim.

`python -m multi_agent_drug.agents.runner --specialist <X>` imports
multi_agent_drug (registers DrugTaskAdapter) then forwards to core main().
"""

from __future__ import annotations

import sys

import multi_agent_drug                              # noqa: F401  (registers adapter)
from agent_core.agents.runner import main

if __name__ == "__main__":
    sys.exit(main())
