"""Local-mode subprocess execution helper.

Used when `adapter.submission_mode == "local"`: the trial runs on the same
host as the supervisor (no SCHED at all), in a subprocess. Useful for:

  * dev iteration on a laptop / single-GPU dev box (e.g. RTX 4090) without
    burning cluster GPU budget
  * harness-side change validation (does my new submit.py not break the
    pipeline?) — local mode reproduces the full stage→preflight→exec→record
    sequence without involving SCHED
  * fast-feedback smoke for a new task fork before booking cluster time

The execution shape mirrors `timeout --signal=TERM --kill-after=N` from
the bash run_trial.sh — SIGTERM at deadline, SIGKILL N seconds later if
the child ignores SIGTERM. PYTHONUNBUFFERED is forced on so a SIGKILLed
child still flushes its last 100s of stdout to the captured log (the
classifier's kill_reason extractor depends on this).

Returns (status, exit_code) tuple where status mirrors sched.JobStatus:
  status.phase ∈ {"succeeded", "failed", "timeout"}
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LocalExecResult:
    """Result of a local subprocess execution. Shape-compatible with sched.JobStatus."""
    phase: str           # "succeeded" | "failed" | "timeout"
    exit_code: Optional[int]
    elapsed_s: float
    log_path: Optional[Path]    # path to captured stdout/stderr file


def _kill_grace_period_s(timeout_s: float) -> float:
    """Default SIGKILL grace = max(30 s, 5% of timeout). Matches PG run_trial.sh."""
    return max(30.0, 0.05 * timeout_s)


async def run_local(
    cmd: list[str] | str,
    cwd: Path,
    *,
    timeout_s: float,
    log_path: Path,
    env_overrides: Optional[dict[str, str]] = None,
    kill_grace_s: Optional[float] = None,
) -> LocalExecResult:
    """Run `cmd` as a subprocess on the local host with bounded wallclock.

    Captures both stdout and stderr to `log_path`. PYTHONUNBUFFERED=1 is
    forced into the child env so a SIGKILLed process still flushes its
    final lines.

    On timeout: send SIGTERM, wait `kill_grace_s` seconds, then SIGKILL
    the entire process group (subprocess + any torch DDP / multiprocessing
    children).
    """
    if kill_grace_s is None:
        kill_grace_s = _kill_grace_period_s(timeout_s)

    # Build child env. Start from operator env, layer overrides, force flush.
    child_env = dict(os.environ)
    if env_overrides:
        child_env.update(env_overrides)
    child_env["PYTHONUNBUFFERED"] = "1"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", buffering=1)        # line-buffered

    # New process group so we can SIGKILL the whole tree on timeout.
    is_str = isinstance(cmd, str)
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_shell(
        cmd if is_str else " ".join(_shell_quote(c) for c in cmd),
        cwd=str(cwd),
        env=child_env,
        stdout=log_fh,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,                     # → new process group
    )

    try:
        exit_code = await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        elapsed = time.monotonic() - t0
        log_fh.flush(); log_fh.close()
        if exit_code == 0:
            return LocalExecResult(phase="succeeded", exit_code=0,
                                   elapsed_s=elapsed, log_path=log_path)
        else:
            return LocalExecResult(phase="failed", exit_code=exit_code,
                                   elapsed_s=elapsed, log_path=log_path)
    except asyncio.TimeoutError:
        # SIGTERM the whole process group, wait grace, SIGKILL if still alive.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            exit_code = await asyncio.wait_for(proc.wait(), timeout=kill_grace_s)
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                exit_code = await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                exit_code = None
        elapsed = time.monotonic() - t0
        log_fh.flush(); log_fh.close()
        return LocalExecResult(phase="timeout", exit_code=exit_code,
                               elapsed_s=elapsed, log_path=log_path)


def _shell_quote(arg: str) -> str:
    """Single-quote shell-escape; safe for arbitrary string args."""
    return "'" + arg.replace("'", "'\"'\"'") + "'"


__all__ = ["run_local", "LocalExecResult"]
