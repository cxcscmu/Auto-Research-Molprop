"""Supervisor — orchestrates N specialists concurrently.

Concurrency model
─────────────────
Specialists each own ONE long-lived asyncio task on the head node. Each
task loops:

    while not should_stop():
        rec = await doer.run_once()        # blocks for the full iter
        append_audit_entry(rec)
        if rec.error is not None:
            await retry_backoff(...)       # spaced retries on SDK errors
        # else: immediately start next iter — blackboard state has changed

The supervisor itself just (a) spawns the tasks with a stagger, (b) runs
the termination watcher in parallel, (c) drops stop.flag when triggered,
(d) cancels the doer tasks once stop.flag is set.

There is NO central planner. Each doer reads the blackboard afresh every
iteration and decides independently what to try next.
"""

from __future__ import annotations
