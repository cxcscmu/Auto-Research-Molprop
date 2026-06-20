"""agent_core — task-agnostic skeleton for the multi-agent research harness.

This package holds all pipeline logic that does NOT depend on a specific
research task (Parameter Golf, nanochat, cifar-airbench, ...). Each task
has its own sibling package (`multi_agent_pg`, `multi_agent_nc`,
`multi_agent_cifar`) that:

  1. Implements a `TaskAdapter` subclass exposing task-specific knobs
     (baseline filename, TSV schema, scoring metric, specialist list, ...).
  2. Calls `register_task_adapter(adapter)` at import time.

The supervisor / agents / tools modules in this package then read those
knobs through `current_adapter()` instead of hard-coding any one task's
constants.

Day 1 status: skeleton only — task adapter is a placeholder. Full
abstract base class arrives in Day 2.
"""

from __future__ import annotations

from typing import Optional


# ── Task adapter registry ────────────────────────────────────────────────────
#
# A single global slot, set by the active task package's __init__.py. Reads
# happen all over (config, prompts, submit, dashboard) so a singleton is the
# pragmatic choice; tests that need isolation can call set the slot directly.

_active_adapter: Optional[object] = None


def register_task_adapter(adapter: object) -> None:
    """Install the active task adapter. Called once by a task package on import."""
    global _active_adapter
    _active_adapter = adapter


def current_adapter() -> object:
    """Return the active task adapter; raises if none is registered yet."""
    if _active_adapter is None:
        raise RuntimeError(
            "no task adapter registered — import a task package first "
            "(e.g. `import multi_agent_pg`) so it can call register_task_adapter()"
        )
    return _active_adapter


__all__ = ["register_task_adapter", "current_adapter"]
