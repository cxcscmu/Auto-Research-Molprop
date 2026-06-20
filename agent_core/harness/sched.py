"""Thin wrapper over the SCHED CLI — the cluster dispatcher the supervisor uses
to sync workdirs to GPU-visible SharedFS and launch the trial command.

Interface matches `multi_agent/sched_env.md` (the authoritative doc):

  sched job create --name N --type h100 --nodes K --priority P --command C
      → JSON: {"job_id": "...", "name": "...", "status": "...",
               "pool": "shared-h100", "nodes": K, "resources": "..."}

  sched job wait  N --until terminal
      → JSON: {"job_id": "...", "status": "succeeded|failed|stopped", ...}

  sched job status N
      → JSON status dict (same "status" field as `wait`)

  sched job logs  N --worker 0 --text
      → plain text stdout+stderr from one pod rank

All JSON commands go through `_run_sched_json`, which detects transient
timeouts (`{"error": "...timeout..."}`) and retries 5× with 30 s linear
backoff. Log streaming is plain text and does not retry (one-shot).

This module is NOT responsible for:

  * building the command string (caller supplies)
  * deciding when to poll (supervisor / submit_trial does that)
  * parsing classified results (harness/tracker.py does that)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import config

_LOG = logging.getLogger("multi_agent.sched")

# Defence-in-depth guards for any job we're about to stop.
#
# Observed on this cluster (2026-04-22): `sched job list` emits duplicate-name
# warnings, e.g. `apg-arch-0001` mapped to THREE distinct job_ids. That
# means job NAME is NOT a unique key on SCHED — a bare `sched job stop <name>`
# can race on an arbitrary one of the duplicates. So we always stop by
# `job_id` (the UUID-like identifier `job-<8-4-4-4-12>` returned by
# `sched job create`), and we refuse to touch any identifier whose shape we
# didn't generate ourselves.
#
# Two guards, both must pass before we invoke `sched job stop`:
#   * `_is_apg_owned_name`: the *human* name matches the active task's
#     `<prefix>-<domain[:4]>-NNNN` shape (prefix from adapter.sched_name_prefix
#     — PG="apg", CIFAR="cif", NC="nc"). Per-task prefix is what keeps
#     concurrent task swarms from cross-killing each other on shutdown.
#   * `_is_job_id`:         the *actual identifier* we pass on the CLI is
#     a fresh UUID SCHED handed us at submit time, so there is no ambiguity
#     between our job and anyone else's that happens to reuse the name.
_JOB_ID_RE = re.compile(r"^job-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# Generic shape matcher; the prefix piece is filled in lazily from adapter.
_NAME_BODY_RE = re.compile(r"^([a-z][a-z0-9]{0,3})-[a-z]{1,4}-\d{4}$")


def _is_apg_owned_name(name: str) -> bool:
    """True iff `name` looks like a job this codebase could have submitted
    UNDER THE CURRENTLY-REGISTERED TASK ADAPTER. Per-task prefix means a
    PG supervisor's shutdown chain ignores `cif-arch-0001` jobs and vice
    versa — concurrent task swarms can't cross-kill.

    Prefix resolution mirrors `harness.config:sched_job_name`:
      1. MAGENT_SCHED_NAME_PREFIX env var (set by supervisor's
         --sched-name-prefix CLI; lets two same-task supervisors run in
         parallel without cross-killing on shutdown).
      2. The active task adapter's `sched_name_prefix`.
      3. Legacy fallback "apg".
    """
    if not name:
        return False
    m = _NAME_BODY_RE.match(name)
    if m is None:
        return False
    from .config import active_sched_name_prefix
    return m.group(1) == active_sched_name_prefix()


def _is_job_id(job_id: str) -> bool:
    """True iff `job_id` looks like the `job-<uuid>` SCHED hands back."""
    return bool(job_id and _JOB_ID_RE.match(job_id))

# ── CLI primitives ───────────────────────────────────────────────────────────

_SCHED_BIN = "sched"                       # on PATH on the supervisor host
_DEFAULT_TIMEOUT_S = 60              # per-call wall limit for short commands
_WAIT_TIMEOUT_S    = 4 * 3600        # `sched job wait` blocks until terminal
_LOGS_TIMEOUT_S    = 120
_SYNC_TIMEOUT_S    = 300
_SSH_TIMEOUT_S     = 60

_RETRY_MAX_ATTEMPTS = 5
_RETRY_BACKOFF_S    = 30.0
_SCHED_CONFIG_PATH = Path.home() / ".config/sched/config.toml"


def _is_transient_timeout(payload: Any) -> bool:
    """Detect the `{"error": "…timeout…"}` shape SCHED uses for transient faults."""
    if not isinstance(payload, dict):
        return False
    err = payload.get("error")
    if err is None:
        return False
    return "timeout" in str(err).lower()


class JobNotFoundError(RuntimeError):
    """Raised by `_run_sched_json` when `sched job status/wait <name>` returns
    rc != 0 with a 'name not found' signature. Distinct from the generic
    RuntimeError so `_wait_for_job` can break out of the poll loop early —
    a truly missing job never returns on retry, and the default 4 h wait
    deadline just burns ccbox CPU for no reason.

    Subclasses RuntimeError so any pre-existing broad catches still work.
    """


# Shapes we've observed SCHED emit when the job name isn't in its records
# (either pre-submit indexing race, pod evicted, or never created):
#   {"error": "Name 'apg-arch-0019' not found"}
#   {"error": "no such job ..."}          (defensive — not observed yet)
#   {"error": "... does not exist"}       (defensive — not observed yet)
_JOB_NOT_FOUND_RE = re.compile(
    r"(?i)(name\s+['\"]?[^'\"]*['\"]?\s+not\s+found|no\s+such\s+job|does\s+not\s+exist)"
)


def _is_job_not_found(payload: Any) -> bool:
    """True iff the error payload matches a 'this job name isn't known' shape."""
    if not isinstance(payload, dict):
        return False
    err = payload.get("error")
    if err is None:
        return False
    return bool(_JOB_NOT_FOUND_RE.search(str(err)))


def _parse_sched_stdout(stdout: str) -> Any:
    """Parse SCHED stdout, accepting one or more whitespace-separated JSON values."""
    decoder = json.JSONDecoder()
    idx = 0
    payloads: list[Any] = []
    length = len(stdout)
    while idx < length:
        while idx < length and stdout[idx].isspace():
            idx += 1
        if idx >= length:
            break
        payload, idx = decoder.raw_decode(stdout, idx)
        payloads.append(payload)
    if not payloads:
        raise json.JSONDecodeError("empty sched stdout", stdout, 0)
    return payloads[-1]


def _run_sched_json(args: list[str], timeout: float = _DEFAULT_TIMEOUT_S,
                 max_attempts: int = _RETRY_MAX_ATTEMPTS,
                 backoff: float = _RETRY_BACKOFF_S) -> dict[str, Any]:
    """Invoke `sched …`, parse stdout as JSON, retry on transient timeouts.

    Retries when:
      * subprocess.TimeoutExpired / OSError (network-level failure)
      * JSON parse failure (partial / empty stdout — usually CLI disconnect)
      * JSON contains `{"error": "...timeout..."}` (SCHED transient)

    Does NOT retry on non-transient errors (non-zero exit + JSON error that
    is not "timeout") — those are surfaced as RuntimeError immediately.
    """
    last_err: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = subprocess.run(
                [_SCHED_BIN, *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            last_err = f"subprocess: {type(e).__name__}: {e}"
            if attempt < max_attempts:
                time.sleep(backoff)
                continue
            raise RuntimeError(
                f"sched {' '.join(args[:3])} failed after {max_attempts} attempts: {last_err}"
            ) from e

        # Try to parse JSON — non-JSON stdout is itself a transient signal
        # (SCHED typically returns JSON even on logical errors).
        try:
            payload = _parse_sched_stdout(result.stdout)
        except json.JSONDecodeError:
            last_err = (f"non-JSON stdout (rc={result.returncode}): "
                        f"{result.stdout[:200]!r} stderr={result.stderr[:200]!r}")
            if attempt < max_attempts:
                time.sleep(backoff)
                continue
            raise RuntimeError(
                f"sched {' '.join(args[:3])} returned non-JSON after "
                f"{max_attempts} attempts: {last_err}"
            )

        if _is_transient_timeout(payload):
            last_err = f"transient timeout: {payload.get('error')}"
            if attempt < max_attempts:
                time.sleep(backoff)
                continue
            raise RuntimeError(
                f"sched {' '.join(args[:3])} timed out after {max_attempts} attempts"
            )

        if result.returncode != 0:
            # Non-transient logical error — do NOT retry.
            if _is_job_not_found(payload):
                raise JobNotFoundError(
                    f"sched {' '.join(args[:3])} failed (rc={result.returncode}): "
                    f"{payload.get('error', payload)}"
                )
            raise RuntimeError(
                f"sched {' '.join(args[:3])} failed (rc={result.returncode}): "
                f"{payload.get('error', payload)}"
            )
        return payload

    # Unreachable (we always return or raise inside the loop), but keep mypy happy.
    raise RuntimeError(f"sched retry logic fell through: {last_err}")


def _run_sched_text(args: list[str], timeout: float = _LOGS_TIMEOUT_S) -> str:
    """Invoke `sched …` expecting plain text stdout (used for `--text` logs)."""
    result = subprocess.run(
        [_SCHED_BIN, *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sched {' '.join(args[:3])} failed (rc={result.returncode})\n"
            f"stderr: {result.stderr}"
        )
    return result.stdout


def _sched_sync_target(remote_name: str) -> tuple[str, str]:
    """Resolve a bare `sched sync :remote_name` target to `(ssh_host, path)`.

    `sched sync` creates the rsync remote from `[sync].remote_host` and
    `[sync].remote_prefix`; rsync itself cannot create missing nested parent
    directories. We mirror the same minimal expansion here so first-run pushes
    can `mkdir -p` the destination before invoking `sched sync`.
    """
    try:
        with _SCHED_CONFIG_PATH.open("rb") as f:
            cfg = tomllib.load(f)
    except OSError as e:
        raise RuntimeError(f"cannot read sched config {_SCHED_CONFIG_PATH}: {e}") from e

    sync_cfg = cfg.get("sync")
    if not isinstance(sync_cfg, dict):
        raise RuntimeError(f"sched config {_SCHED_CONFIG_PATH} has no [sync] section")
    remote_host = str(sync_cfg.get("remote_host") or "")
    remote_prefix = str(sync_cfg.get("remote_prefix") or "")
    if not remote_host:
        raise RuntimeError(f"sched config {_SCHED_CONFIG_PATH} missing [sync].remote_host")

    clean_name = remote_name.rstrip("/")
    if clean_name.startswith(("/", "~")):
        remote_path = clean_name
    elif remote_prefix:
        remote_path = f"{remote_prefix.rstrip('/')}/{clean_name}"
    else:
        remote_path = clean_name
    return remote_host, remote_path


def _quote_remote_path(path: str) -> str:
    """Shell-quote a remote path while preserving leading HOME expansion."""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME/' + path[2:].replace("\\", "\\\\").replace('"', '\\"') + '"'
    if path.startswith("$HOME/"):
        return '"$HOME/' + path[6:].replace("\\", "\\\\").replace('"', '\\"') + '"'
    return shlex.quote(path)


def _ensure_remote_sync_dir(remote_name: str) -> None:
    remote_host, remote_path = _sched_sync_target(remote_name)
    result = subprocess.run(
        ["ssh", remote_host, f"mkdir -p -- {_quote_remote_path(remote_path)}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=_SSH_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"remote mkdir failed on {remote_host}:{remote_path} "
            f"(rc={result.returncode})\nstderr: {result.stderr}"
        )


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class JobHandle:
    """Opaque reference to a submitted GPU job.

    `name` is the caller-supplied SCHED job name (all subsequent CLI calls
    look up by name, not job_id). `job_id` is SCHED's internal identifier
    surfaced by `sched job create` — useful for logs / audit trails.
    """
    name: str
    job_id: str


@dataclass(frozen=True, slots=True)
class JobStatus:
    """Snapshot of a GPU job's lifecycle state.

    `phase` is the `status` field returned by `sched job wait|status` —
    canonical values are `"succeeded" | "failed" | "stopped" |
    "pending" | "running" | "unknown"`. Exit code is not exposed by the
    SCHED CLI, so we always report None; callers infer success from phase.
    """
    name: str
    phase: str
    exit_code: Optional[int] = None


TERMINAL_PHASES = frozenset({"succeeded", "failed", "stopped"})


def is_terminal(status: JobStatus) -> bool:
    """True once the job is beyond rescue.

    `unknown` is intentionally NOT treated as terminal — SCHED sometimes
    reports it for a few seconds after a pod is scheduled.
    """
    return status.phase in TERMINAL_PHASES


# ── Public API ───────────────────────────────────────────────────────────────

def submit_gpu_job(
    name: str,
    command: str,
    *,
    nodes: int = 1,
    priority: Optional[int] = None,
    type_: str = "h100",
    cpu: Optional[int] = None,
    mem: Optional[int] = None,
) -> JobHandle:
    """Dispatch `command` to a GPU pod, returning a JobHandle.

    `command` is the full bash one-liner to run inside the pod (must include
    any `cd`/venv activation). If omitted, `priority` uses
    `config.SCHED_TASK_PRIORITY`, which defaults to 10 and can be overridden via
    `MAGENT_SCHED_TASK_PRIORITY`.
    """
    if priority is None:
        priority = config.SCHED_TASK_PRIORITY
    if type_ == "h100":
        if cpu is None:
            cpu = config.SCHED_H100_CPU
        if mem is None:
            mem = config.SCHED_H100_MEM_GIB
    args = [
        "job", "create",
        "--name", name,
        "--type", type_,
        "--nodes", str(nodes),
        "--priority", str(priority),
        "--command", command,
    ]
    if cpu is not None:
        args.extend(["--cpu", str(cpu)])
    if mem is not None:
        args.extend(["--mem", str(mem)])
    payload = _run_sched_json(args, timeout=_DEFAULT_TIMEOUT_S)
    job_id = str(payload.get("job_id") or "")
    return JobHandle(name=name, job_id=job_id)


def wait_gpu_job(job: JobHandle, timeout_s: float = _WAIT_TIMEOUT_S) -> JobStatus:
    """Block until the job reaches a terminal phase (succeeded/failed/stopped).

    Uses `sched job wait --until terminal`, which is the CLI's native blocking
    primitive — one RPC instead of an active polling loop. Retries on
    transient timeouts inside `_run_sched_json`.
    """
    try:
        payload = _run_sched_json(
            ["job", "wait", job.name, "--until", "terminal"],
            timeout=timeout_s,
        )
    except JobNotFoundError:
        # The CLI's local name→id cache can transiently drop entries once
        # other submits bump the same cache file; job_id is authoritative.
        if not job.job_id:
            raise
        payload = _run_sched_json(
            ["job", "wait", job.job_id, "--until", "terminal"],
            timeout=timeout_s,
        )
    phase = str(payload.get("status") or "unknown").lower()
    return JobStatus(name=job.name, phase=phase, exit_code=None)


def poll_gpu_job(job: JobHandle) -> JobStatus:
    """Non-blocking status probe. Prefer `wait_gpu_job` if you can block.

    Useful when the caller needs to surface progress info to a UI or
    interleave status checks with other work. `wait_gpu_job` is cheaper
    for the common 'I want the answer whenever it arrives' case.
    """
    try:
        payload = _run_sched_json(["job", "status", job.name], timeout=_DEFAULT_TIMEOUT_S)
    except JobNotFoundError:
        # Name missing from local jobs cache — fall back to the authoritative
        # job_id (the sched CLI accepts either per `sched job status --help`).
        if not job.job_id:
            raise
        payload = _run_sched_json(["job", "status", job.job_id], timeout=_DEFAULT_TIMEOUT_S)
    phase = str(payload.get("status") or "unknown").lower()
    return JobStatus(name=job.name, phase=phase, exit_code=None)


_TERMINAL_STATUSES = frozenset({
    "stopped", "cancelled", "cancelling", "succeeded", "failed",
    "completed", "killed", "expired", "terminated",
})


def _query_job_status_terminal(job_id: str, timeout_s: float = 10.0) -> str:
    """Best-effort: return the lowercase `status` field for `job_id`, or "".

    Uses `sched job status <id>` which on this cluster returns JSON-shaped
    stdout. Tries strict json.loads first, falls back to a regex pull on
    `"status": "..."`. Empty string on any error / parse failure — caller
    treats empty as "non-terminal / unverifiable".
    """
    import json
    import re
    try:
        result = subprocess.run(
            [_SCHED_BIN, "job", "status", job_id],
            check=False, capture_output=True, text=True, timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError):
        return ""
    if result.returncode != 0:
        return ""
    text = result.stdout or ""
    # Strict parse first.
    try:
        return str(json.loads(text).get("status", "")).strip().lower()
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass
    # Regex fallback for non-strict-JSON shapes.
    m = re.search(r'"status"\s*:\s*"([^"]+)"', text)
    return m.group(1).strip().lower() if m else ""


def stop_gpu_job(name: str, job_id: str, timeout_s: float = _DEFAULT_TIMEOUT_S) -> bool:
    """Ask SCHED to stop one specific job, addressing by its unique `job_id`.

    We pass `job_id` (the `job-<uuid>` returned at submit time) as the CLI
    identifier, NOT `name`, because job NAMES are not unique on this
    cluster (we've seen `apg-arch-0001` mapped to three distinct job_ids
    from prior runs). Using the UUID guarantees we stop exactly the job we
    submitted and nothing else.

    `name` is accepted purely for pattern-checking and logging — it must
    match the APG convention (`apg-<domain[:4]>-NNNN`), else we refuse.
    Used on supervisor shutdown; best-effort, swallows all errors.
    """
    if not _is_apg_owned_name(name):
        _LOG.error(
            "REFUSING to stop job name=%r: does not match APG pattern "
            "(%s). This would be a stop of something we did not submit.",
            name, _APG_JOB_NAME_RE.pattern,
        )
        return False
    if not _is_job_id(job_id):
        _LOG.error(
            "REFUSING to stop job name=%s: missing or malformed job_id=%r "
            "(expected shape %s). Name collisions on SCHED mean a bare-name "
            "stop is unsafe — refusing.",
            name, job_id, _JOB_ID_RE.pattern,
        )
        return False
    try:
        result = subprocess.run(
            [_SCHED_BIN, "job", "stop", job_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        _LOG.warning("sched job stop %s (%s): %s", name, job_id, e)
        return False
    if result.returncode == 0:
        _LOG.info("sched job stop %s (%s) ok", name, job_id)
        return True

    # rc != 0 doesn't necessarily mean failure on this cluster's sched:
    # observed 2026-04-26 — `sched job stop <id>` returns rc=1 with empty
    # stderr and empty stdout EVEN WHEN the action succeeded; verifying
    # via `sched job status <id>` afterwards shows status="stopped". So
    # treat rc!=0 as ambiguous and follow up with a status query: if the
    # job has reached any terminal phase, count as effective.
    raw_out = (result.stdout or "").strip()[:200]
    raw_err = (result.stderr or "").strip()[:200]
    actual_status = _query_job_status_terminal(job_id, timeout_s=10.0)
    if actual_status in _TERMINAL_STATUSES:
        _LOG.info(
            "sched job stop %s (%s) effective  (sched CLI returned rc=%d but "
            "follow-up status=%s — sched CLI quirk, action did succeed)",
            name, job_id, result.returncode, actual_status,
        )
        return True
    # Not terminal → genuine failure (or status check itself unavailable).
    _LOG.warning(
        "sched job stop %s (%s) rc=%d  stop_stdout=%r  stop_stderr=%r  "
        "post_status=%r  (job NOT terminal — GPU may still be running)",
        name, job_id, result.returncode, raw_out, raw_err, actual_status,
    )
    return False


# ── Active-job registry (for shutdown cleanup) ───────────────────────────────
# Thread-safe map {name → job_id} of SCHED jobs that have been submitted but
# have not yet reached a terminal phase. `tools/submit.py` registers here
# right after `submit_gpu_job` (when we have both the name and the fresh
# unique job_id) and unregisters in a `finally` once the wait returns.
# The supervisor's signal/atexit handlers iterate this map to issue
# `sched job stop <job_id>` on Ctrl+C so we don't leak cluster GPU minutes
# AND we target the exact job we submitted — not a same-named duplicate
# belonging to someone else.

_ACTIVE_JOBS: dict[str, str] = {}
_ACTIVE_JOBS_LOCK = threading.Lock()


def register_active_job(name: str, job_id: str) -> None:
    """Mark `(name, job_id)` as in-flight. Safe from any thread / coroutine.

    Safety: refuses to register if *either* the name doesn't look APG-owned
    or the job_id doesn't match the `job-<uuid>` shape SCHED hands back. A
    corrupted caller is logged loudly and silently dropped so the shutdown
    path cannot later try to stop a foreign job.
    """
    if not _is_apg_owned_name(name):
        _LOG.error(
            "REFUSING to register job name=%r in active set: does not match "
            "APG pattern (%s).",
            name, _APG_JOB_NAME_RE.pattern,
        )
        return
    if not _is_job_id(job_id):
        _LOG.error(
            "REFUSING to register job name=%s with malformed job_id=%r "
            "(expected shape %s).",
            name, job_id, _JOB_ID_RE.pattern,
        )
        return
    with _ACTIVE_JOBS_LOCK:
        _ACTIVE_JOBS[name] = job_id


def unregister_active_job(name: str) -> None:
    """Mark `name` as finished. Safe to call from any thread / coroutine."""
    with _ACTIVE_JOBS_LOCK:
        _ACTIVE_JOBS.pop(name, None)


def snapshot_active_jobs() -> list[tuple[str, str]]:
    """Return a copy of the active registry as sorted (name, job_id) pairs."""
    with _ACTIVE_JOBS_LOCK:
        return sorted(_ACTIVE_JOBS.items())


def stop_all_active_jobs() -> list[tuple[str, str]]:
    """Issue `sched job stop <job_id>` for every registered active job.

    Drains the registry atomically first so repeated calls are idempotent
    even if a second signal arrives mid-cleanup. Every entry is filtered
    through BOTH guards (`_is_apg_owned_name` + `_is_job_id`) before we
    invoke stop — so even a corrupted registry cannot end up targeting
    someone else's job. Returns the list of (name, job_id) pairs we
    actually issued stops for, for the shutdown log's audit trail.
    """
    with _ACTIVE_JOBS_LOCK:
        items = sorted(_ACTIVE_JOBS.items())
        _ACTIVE_JOBS.clear()
    tried: list[tuple[str, str]] = []
    for name, job_id in items:
        if not _is_apg_owned_name(name) or not _is_job_id(job_id):
            _LOG.error(
                "skipping malformed registry entry name=%r job_id=%r",
                name, job_id,
            )
            continue
        stop_gpu_job(name, job_id)
        tried.append((name, job_id))
    return tried


def fetch_logs(job: JobHandle, worker: int = 0) -> str:
    """Pull one rank's stdout+stderr as plain text. `worker=0` = rank 0.

    Tail limiting is NOT supported by this CLI — caller tails client-side.
    """
    return _run_sched_text(
        ["job", "logs", job.name, "--worker", str(worker), "--text"],
        timeout=_LOGS_TIMEOUT_S,
    )


# ── Notebook mode (single-GPU long-lived) ───────────────────────────────────
#
# `sched notebook` is a separate SCHED subsystem from `sched job`. It allocates a
# long-lived single-GPU pod (Jupyter/VSCode-style) and supports `sched
# notebook exec` to run a shell command in it. We use it for tasks that
# only need 1 GPU (CIFAR airbench94) so we don't waste 7/8 cards by
# allocating a whole h100 node — and avoid cluster-admin warnings about
# under-utilization.
#
# Lifecycle, contrasted with jobs:
#   * Jobs are fire-and-forget: submit → run-to-completion → terminal.
#   * Notebooks persist: create once → exec many times → stop / auto-stop.
#
# Notebook identifiers:
#   * `name`        — operator-supplied; what we keep in registry, e.g. "cif-arch-nb"
#   * `notebook_id` — UUID returned at create time; used for stop / metrics
# (Names are NOT globally unique; same as jobs, multiple notebooks could
# share a name across history. We track {name → notebook_id} per-process.)

_NB_DEFAULT_CREATE_TIMEOUT_S = 30.0
_NB_DEFAULT_EXEC_TIMEOUT_S   = 4 * 3600.0
_NB_DEFAULT_STOP_TIMEOUT_S   = 60.0


def notebook_status(name: str, timeout_s: float = _DEFAULT_TIMEOUT_S) -> str:
    """Return the notebook's current status string ("pending"/"running"/
    "stopped"/"failed" etc), or "absent" if no notebook with that name."""
    try:
        out = _run_sched_json(["notebook", "status", name], timeout=timeout_s)
    except RuntimeError as e:
        msg = str(e).lower()
        if "not found" in msg or "no such" in msg:
            return "absent"
        raise
    if isinstance(out, dict):
        return str(out.get("status", "unknown")).lower()
    return "unknown"


def notebook_create(
    name: str,
    *,
    image: str,
    type_: str = "h100",
    gpus: int = 1,
    cpu: Optional[int] = None,
    mem: Optional[int] = None,
    auto_stop: int = 43200,
    shm_size: Optional[int] = None,
    timeout_s: float = _NB_DEFAULT_CREATE_TIMEOUT_S,
) -> str:
    """Create a notebook (does NOT wait for running). Returns notebook_id.

    Caller's responsibility to call notebook_wait_running() afterwards.
    Idempotency: if a notebook with this name is already present
    (any status), this raises — use notebook_ensure() instead.
    """
    args = ["notebook", "create",
            "--name",  name,
            "--type",  type_,
            "--gpus",  str(gpus),
            "--image", image,
            "--auto-stop", str(int(auto_stop))]
    if cpu is not None:
        args += ["--cpu", str(cpu)]
    if mem is not None:
        args += ["--mem", str(mem)]
    if shm_size is not None:
        args += ["--shm-size", str(shm_size)]
    out = _run_sched_json(args, timeout=timeout_s)
    if not isinstance(out, dict) or "notebook_id" not in out:
        raise RuntimeError(f"notebook create returned unexpected payload: {out!r}")
    return str(out["notebook_id"])


def notebook_wait_running(name: str, timeout_s: float = 600.0) -> str:
    """Block until the notebook reaches `running` (or terminal). Returns
    final status. Raises on SCHED CLI failure or unrecognized status."""
    out = _run_sched_json(
        ["notebook", "wait", name],
        timeout=timeout_s,
    )
    if isinstance(out, dict):
        return str(out.get("status", "unknown")).lower()
    return "unknown"


def notebook_ensure(
    name: str,
    *,
    image: str,
    type_: str = "h100",
    gpus: int = 1,
    cpu: Optional[int] = None,
    mem: Optional[int] = None,
    auto_stop: int = 43200,
    shm_size: Optional[int] = None,
    create_timeout_s: float = _NB_DEFAULT_CREATE_TIMEOUT_S,
    wait_timeout_s: float = 600.0,
) -> str:
    """Idempotent: ensure a notebook with `name` exists and is running.

    - If absent → create + wait.
    - If running → return notebook_id (re-fetch via status if not in registry).
    - If pending → wait.
    - If stopped/failed → raise (operator must investigate; we don't auto-recreate).
    """
    st = notebook_status(name)
    if st == "absent":
        nb_id = notebook_create(
            name, image=image, type_=type_, gpus=gpus,
            cpu=cpu, mem=mem, auto_stop=auto_stop, shm_size=shm_size,
            timeout_s=create_timeout_s,
        )
        notebook_wait_running(name, timeout_s=wait_timeout_s)
        return nb_id
    if st in ("pending", "starting"):
        notebook_wait_running(name, timeout_s=wait_timeout_s)
        st = notebook_status(name)
    if st != "running":
        raise RuntimeError(
            f"notebook {name!r} status={st!r}; expected running. "
            f"Use `sched notebook delete {name}` to recreate, or investigate."
        )
    # Re-fetch notebook_id from status payload — running notebook exists,
    # but we may not have its id in this process's registry.
    detail = _run_sched_json(["notebook", "status", name])
    if not isinstance(detail, dict) or "notebook_id" not in detail:
        raise RuntimeError(f"notebook status missing notebook_id: {detail!r}")
    return str(detail["notebook_id"])


def notebook_exec(
    name: str,
    command: str,
    *,
    timeout_s: float = _NB_DEFAULT_EXEC_TIMEOUT_S,
) -> tuple[str, int]:
    """Run `command` inside the notebook via `sched notebook exec`.

    Returns (output_str, exit_code). `output_str` is the merged stdout
    (notebook exec doesn't separate stderr). `exit_code` is the child
    command's exit code (NOT the SCHED CLI's). Blocks for up to timeout_s.

    The command is sent as the literal `exec_command` positional arg (one
    string). Caller is responsible for shell-quoting if needed; typically
    you'd pass `bash <script> <args>` or `cd <dir> && bash <script>`.
    """
    out = _run_sched_json(
        ["notebook", "exec", "--timeout", str(int(timeout_s)), name, command],
        timeout=timeout_s + 60.0,           # CLI overhead margin
    )
    if not isinstance(out, dict):
        raise RuntimeError(f"notebook exec returned non-dict: {out!r}")
    output = str(out.get("output", ""))
    rc_raw = out.get("exit_code")
    try:
        rc = int(rc_raw) if rc_raw is not None else -1
    except (TypeError, ValueError):
        rc = -1
    return output, rc


def notebook_stop(
    name: str,
    notebook_id: Optional[str] = None,
    timeout_s: float = _NB_DEFAULT_STOP_TIMEOUT_S,
) -> bool:
    """Stop the notebook (graceful). Prefers `notebook_id` (globally unique)
    over `name` (not unique on this cluster — same warning as for jobs in
    sched.py:53). Falls back to name if id-stop is unsupported by the CLI.

    Returns True if the CLI call succeeded OR the notebook was already
    absent. Returns False on hard CLI error.
    """
    targets = []
    if notebook_id:
        targets.append(notebook_id)
    targets.append(name)            # always include name as fallback
    last_err = None
    for i, tgt in enumerate(targets):
        is_last = (i == len(targets) - 1)
        try:
            _run_sched_json(["notebook", "stop", tgt], timeout=timeout_s)
            return True
        except RuntimeError as e:
            msg = str(e).lower()
            absent = ("not found" in msg or "absent" in msg
                      or "already stopped" in msg)
            # Treat "absent" as success ONLY when this is the final target —
            # otherwise an id-stale "not found" would short-circuit before
            # the name-based fallback got a chance to find the live notebook.
            if absent and is_last:
                return True
            if absent:
                # Stale or unrecognized id; fall through to name target.
                continue
            last_err = e
            # Probably "unrecognized id" or transient error — try next target.
            continue
    if last_err:
        _LOG.warning("notebook_stop name=%s id=%s failed: %s",
                     name, notebook_id, last_err)
    return False


# Notebook registry — same shape + safety as job registry above.
_ACTIVE_NOTEBOOKS: dict[str, str] = {}
_ACTIVE_NOTEBOOKS_LOCK = threading.Lock()


def register_active_notebook(name: str, notebook_id: str) -> None:
    """Mark a notebook as in-flight. Used by shutdown chain to clean up."""
    if not _is_apg_owned_name_loose(name):
        _LOG.error("REFUSING to register notebook name=%r (no task prefix)", name)
        return
    with _ACTIVE_NOTEBOOKS_LOCK:
        _ACTIVE_NOTEBOOKS[name] = notebook_id


def unregister_active_notebook(name: str) -> None:
    with _ACTIVE_NOTEBOOKS_LOCK:
        _ACTIVE_NOTEBOOKS.pop(name, None)


def snapshot_active_notebooks() -> list[tuple[str, str]]:
    with _ACTIVE_NOTEBOOKS_LOCK:
        return sorted(_ACTIVE_NOTEBOOKS.items())


def stop_all_active_notebooks() -> list[tuple[str, str]]:
    """Drain the registry and stop every notebook this process owns.
    Mirrors stop_all_active_jobs(). Stops by notebook_id first (UUID,
    globally unique) and falls back to name if id is rejected. Returns
    list of (name, notebook_id) we issued stops for."""
    with _ACTIVE_NOTEBOOKS_LOCK:
        items = sorted(_ACTIVE_NOTEBOOKS.items())
        _ACTIVE_NOTEBOOKS.clear()
    tried: list[tuple[str, str]] = []
    for name, nb_id in items:
        if not _is_apg_owned_name_loose(name):
            _LOG.error("skipping malformed nb registry name=%r", name)
            continue
        notebook_stop(name, notebook_id=nb_id)
        tried.append((name, nb_id))
    return tried


def _is_apg_owned_name_loose(name: str) -> bool:
    """True iff `name` looks like one of OUR notebooks. Notebook names use
    a slightly different shape than job names — `<prefix>-<spec[:4]>-nb` —
    so we accept either the strict job form or this notebook form, both
    gated on the active prefix (env override → adapter → "apg")."""
    if not name:
        return False
    from .config import active_sched_name_prefix
    expected = active_sched_name_prefix()
    # Strict job shape OR notebook shape ("<prefix>-<spec>-nb")
    if _is_apg_owned_name(name):
        return True
    return bool(re.match(rf"^{re.escape(expected)}-[a-z]{{1,4}}-nb$", name))


def sync_workdir(local_path: str, remote_name: str) -> None:
    """Upload one local specialist workdir to the pod-visible SharedFS mirror."""
    _ensure_remote_sync_dir(remote_name)
    args = [
        "sync",
        local_path.rstrip("/") + "/",
        f":{remote_name}/",
        "-a", "--delete",
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        "--exclude=ckpt/",
        "--exclude=logs/",
        "--exclude=full_eval_results/",
    ]
    result = subprocess.run(
        [_SCHED_BIN, *args], check=False, capture_output=True, text=True, timeout=_SYNC_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sched sync failed (rc={result.returncode})\nstderr: {result.stderr}"
        )


def sync_workdir_results(remote_name: str, local_path: str) -> None:
    """Pull `full_eval_results/` from the remote workdir into the local mirror."""
    target = Path(local_path)
    target.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            _SCHED_BIN,
            "sync",
            f":{remote_name}/full_eval_results/",
            target.joinpath("full_eval_results").as_posix().rstrip("/") + "/",
            "-a",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=_SYNC_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sched sync pull failed (rc={result.returncode})\nstderr: {result.stderr}"
        )
