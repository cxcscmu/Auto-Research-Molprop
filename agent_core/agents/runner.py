"""CLI entrypoint for a single specialist iteration (task-agnostic).

Usage (after a task package has been imported, which registers an adapter):

    python -m multi_agent_pg.agents.runner --specialist arch
    python -m multi_agent_nc.agents.runner --specialist arch
    python -m multi_agent_cifar.agents.runner --specialist arch

Runs DoerBase.run_once() for the named specialist and prints a concise
summary. The supervisor will orchestrate many of these in parallel; this
module is the single-specialist debugging surface.

Each task package's `agents/runner.py` is a thin shim: it imports its task
package (so the adapter registers) then forwards to `main()` here.

Specialists, the score-field name shown in the end-of-iter summary, and
all task-shaped strings come from `current_adapter()`. CLI flags
(`--thinking-budget / --max-turns / --no-web / --dry-run`) are unchanged
from the pre-Gap-4 PG runner.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Optional

# `.base` and harness modules are DEFERRED. Same reason as core.supervisor:
# task __init__.py may setdefault MAGENT_LOCAL_ROOT before any code reads
# config env. .base imports config at top, so importing .base here would
# freeze LOCAL_ROOT before _ensure_task_package_registered runs.
# Imports happen inside main() / _dry_run() after task registration.


# ── Task package selector — same as core.supervisor ──────────────────────────

_TASK_PKG_MAP = {
    "pg":     "multi_agent_pg",
    "nc":     "multi_agent_nc",
    "cifar":  "multi_agent_cifar",
}


def _peek_task_arg(argv: Optional[list[str]]) -> Optional[str]:
    args = argv if argv is not None else sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--task" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--task="):
            return a.split("=", 1)[1]
    return None


def _ensure_task_package_registered(argv: Optional[list[str]]) -> None:
    """Import the active task package so register_task_adapter runs.

    No-op if an adapter is already registered (the task wrapper path
    imports its package before forwarding here, which is the common
    case). Direct invocation `python -m agent_core.agents.runner
    --task pg ...` resolves through MAGENT_TASK / --task.
    """
    from agent_core import _active_adapter
    if _active_adapter is not None:
        return
    name = _peek_task_arg(argv) or os.environ.get("MAGENT_TASK", "").strip().lower()
    if not name:
        raise SystemExit(
            "no task package registered. Set MAGENT_TASK=pg|nc|cifar, "
            "pass --task <name>, or invoke via a task wrapper "
            "(e.g. `python -m multi_agent_pg.agents.runner`)"
        )
    pkg = _TASK_PKG_MAP.get(name)
    if pkg is None:
        raise SystemExit(
            f"unknown --task / MAGENT_TASK value {name!r} "
            f"(known: {sorted(_TASK_PKG_MAP)})"
        )
    __import__(pkg)


def _adapter():
    from agent_core import current_adapter
    return current_adapter()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one iteration of one specialist.")
    p.add_argument(
        "--task",
        choices=sorted(_TASK_PKG_MAP),
        default=None,
        help=("Task package selector: pg / nc / cifar. Equivalent to "
              "MAGENT_TASK env. Required when invoking core.runner "
              "directly; ignored if you've already imported a task package "
              "(e.g. via the `multi_agent_pg.agents.runner` wrapper)."),
    )
    p.add_argument("--specialist", required=True,
                   choices=sorted(_adapter().all_domains))
    p.add_argument("--thinking-budget", type=int, default=None,
                   help="Override extended-thinking token budget.")
    p.add_argument("--max-turns", type=int, default=None,
                   help="Override max tool-use turns in the session.")
    p.add_argument("--no-web", action="store_true",
                   help="Disable WebSearch/WebFetch for this iteration.")
    p.add_argument("--dry-run", action="store_true",
                   help="Render the system + user message and exit without "
                        "calling the SDK. Useful for prompt debugging.")
    return p


def _print_summary(rec) -> int:
    """Human-readable end-of-iter summary. Returns shell exit code."""
    score_field = _adapter().score_field
    print(f"\n[{rec.specialist}] iter: {rec.iter_start} → {rec.iter_end}")
    if rec.error:
        print(f"  ERROR: {rec.error}")
        return 2
    print(f"  session_id: {rec.session_id or 'n/a'}")
    print(f"  tool_calls: {rec.tool_calls}")
    if rec.tool_trace:
        print(f"  trace: {' → '.join(rec.tool_trace)}")
    if rec.usage:
        print(f"  usage: {rec.usage}")
    if rec.final_row:
        r = rec.final_row
        print(f"  final: exp_{r.get('exp_id','?')} "
              f"status={r.get('status')} "
              f"{score_field}={r.get(score_field,'—')} "
              f"Δ={r.get('delta_vs_best','—')}")
        return 0 if r.get("status") in ("keep", "discard") else 1
    print("  final: no submit_trial call recorded")
    return 1


def _dry_run(specialist: str) -> int:
    """Render prompts offline for inspection. Never calls the SDK."""
    # Lazy-import .base — its top-level config import would otherwise
    # snapshot LOCAL_ROOT before task __init__ env setdefault runs.
    from .base import render_user_message
    print("=" * 72)
    print("SYSTEM PROMPT")
    print("=" * 72)
    print(_adapter().build_system_prompt(specialist))
    print("=" * 72)
    print("USER MESSAGE")
    print("=" * 72)
    print(render_user_message(specialist))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    # Register adapter BEFORE building parser (which reads adapter for
    # specialist choices) and BEFORE importing harness or .base (both of
    # which read config at module-load → freeze MAGENT_LOCAL_ROOT).
    # Task wrapper path is no-op; direct path resolves --task / MAGENT_TASK.
    _ensure_task_package_registered(argv)

    args = _build_parser().parse_args(argv)

    if args.dry_run:
        return _dry_run(args.specialist)

    # Now safe to import harness + base — task setdefault env defaults are final.
    from ..harness import blackboard, config, credentials
    from .base import DoerBase, DoerConfig
    config.ensure_dirs()
    blackboard.regenerate_markdown()    # make sure MD files reflect current TSV

    credentials.ensure_api_key()
    cls = _adapter().specialist_classes()[args.specialist]
    cfg_kwargs = {"specialist": args.specialist}
    if args.thinking_budget is not None:
        cfg_kwargs["thinking_budget"] = args.thinking_budget
    if args.max_turns is not None:
        cfg_kwargs["max_turns"] = args.max_turns
    if args.no_web:
        cfg_kwargs["enable_web"] = False

    doer = cls(cfg=DoerConfig(**cfg_kwargs))
    rec = asyncio.run(doer.run_once())
    return _print_summary(rec)


if __name__ == "__main__":
    sys.exit(main())
