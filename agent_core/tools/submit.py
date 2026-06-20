"""submit_trial — task-agnostic 3-mode trial pipeline.

Flow when a specialist calls this:

  1. Stage  — copy task helper scripts + run script + (first iter only)
     baseline seed into workdir. File list / seed / run script all read
     from `adapter`.
  2. Preflight — head-side syntax_check + task-specific size_check. On
     fail, RECORD a preflight_crash / size_blocked row + return early
     (no GPU used).
  3. Execute — branch on `adapter.submission_mode`:
       "job"      → sched sync push → sched job submit → poll phases → sched pull
       "notebook" → sched notebook ensure → sched sync push → notebook exec → sched pull
       "local"    → run_local subprocess (no push, no pull)
  4. Parse + Record — shared across all modes.

PG (submission_mode default "job") goes through the EXACT event sequence
as before: submit_trial_called → stage_ok → preflight_* → sched_push_ok →
sched_submit → sched_phase × N → sched_terminal → sched_pull_* → classify_done.
The job-mode executor preserves byte-equal events.

Notebook + local modes emit `nb_*` / `local_*` events instead of `sched_*`,
so audit trails distinguish modes.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import time
from pathlib import Path
from typing import Any, Optional


def _pod_env_prefix(adapter) -> str:
    """Build a `KEY1='val1' KEY2='val2'` prefix for run_trial.sh from the
    adapter's pod_env_for_trial dict. Empty string when adapter declares
    no env (PG default).

    Quoting rule: values are shlex-quoted to defend against shell
    metachars, EXCEPT values that contain literal `$HOME` — those need
    pod-side bash expansion to resolve correctly (pod-side $HOME is the
    user's SharedFS-home on sched job pods). For those, emit unquoted so bash
    expands `$HOME` at exec time. Such values are by construction
    operator-controlled paths from task_config.py (no untrusted input)
    so the security trade-off is acceptable."""
    env = adapter.pod_env_for_trial
    if not env:
        return ""
    parts = []
    for k, v in env.items():
        if "$HOME" in v:
            parts.append(f"{k}={v}")          # let pod-side bash expand $HOME
        else:
            parts.append(f"{k}={shlex.quote(v)}")
    return " ".join(parts) + " "

from . import tool
from .code_inspect import _syntax_check_impl
from ..harness import blackboard, config, events, sched, tracker
from ..harness.local_exec import run_local


# ── Tunables (task-agnostic) ────────────────────────────────────────────────

_JOB_WAIT_TIMEOUT_S = 4 * 3600   # outer wall limit
_POLL_INTERVAL_S = 15.0
_JOB_MISSING_GRACE_POLLS = 3


# ── Adapter helpers ──────────────────────────────────────────────────────────

def _adapter():
    from agent_core import current_adapter
    return current_adapter()


# ── Shared MCP wrapper ───────────────────────────────────────────────────────

def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}


# ── Staging ──────────────────────────────────────────────────────────────────

def _stage_workdir(workdir: Path, pkg_root: Path) -> None:
    """Seed baseline source (first-iter only) + optionally seed multi-file
    editable tree (first-iter only) + refresh helper scripts every call.

    Adapter properties consulted:
      seed_file       — single editable file copied if missing
      editable_tree   — optional directory (pkg-relative) copied recursively
                        if absent in workdir (NC vendor pattern)
      stage_files     — helper scripts mtime-refreshed every call
    """
    workdir.mkdir(parents=True, exist_ok=True)

    adapter = _adapter()
    seed_file = adapter.seed_file
    target = workdir / seed_file
    if not target.exists():
        seed_src = pkg_root / seed_file
        if seed_src.is_file():
            target.write_bytes(seed_src.read_bytes())

    # Optional: copy entire editable_tree (NC vendor) on first iter only.
    # Subsequent iters preserve the agent's accumulated edits to the tree.
    tree = adapter.editable_tree
    if tree:
        tree_dst = workdir / tree
        if not tree_dst.exists():
            tree_src = pkg_root / tree
            if tree_src.is_dir():
                # shutil.copytree preserves dir structure + permissions
                shutil.copytree(tree_src, tree_dst, symlinks=False)

    # Refresh helper scripts UNCONDITIONALLY every call. Helpers are
    # harness-trusted code (run_trial.sh, run_classify.py, profile_pipeline.py)
    # — they form the boundary between agent-edited recipe and the
    # measurement / reward path. If we mtime-gated, an older stale-but-newer
    # mtime workdir copy (e.g. from a tar restore, or a prior pre-fix swarm
    # run, or an accidental agent edit) would survive and could compromise
    # measurement integrity (e.g. the v2 shell-side timing sidecar contract).
    # Cost is one filesystem write per stage_file per iter — negligible.
    for rel_src, dst_name in adapter.stage_files:
        src = pkg_root / rel_src
        if src.is_file():
            dst = workdir / dst_name
            dst.write_bytes(src.read_bytes())


def _clear_local_trial_outputs(workdir: Path) -> None:
    """Remove stale trial outputs so parsing never reads old artifacts."""
    for name in _adapter().trial_output_dirs:
        shutil.rmtree(workdir / name, ignore_errors=True)


# ── Wait for terminal ────────────────────────────────────────────────────────

async def _wait_for_job(
    job: sched.JobHandle,
    specialist: str,
    timeout_s: float = _JOB_WAIT_TIMEOUT_S,
) -> sched.JobStatus:
    """Poll `sched job status` until terminal, emitting phase-change events."""
    deadline = time.monotonic() + timeout_s
    last_phase: Optional[str] = None
    consecutive_missing = 0
    while True:
        try:
            status = await asyncio.to_thread(sched.poll_gpu_job, job)
        except sched.JobNotFoundError as e:
            consecutive_missing += 1
            if consecutive_missing >= _JOB_MISSING_GRACE_POLLS:
                events.emit(
                    specialist, "sched_job_missing",
                    job=job.name, err=str(e)[:120],
                    grace_polls=_JOB_MISSING_GRACE_POLLS,
                )
                return sched.JobStatus(name=job.name, phase="failed", exit_code=None)
            events.emit(specialist, "sched_poll_fail", job=job.name, err=str(e)[:120])
            if time.monotonic() > deadline:
                return sched.JobStatus(name=job.name, phase="failed", exit_code=None)
            await asyncio.sleep(_POLL_INTERVAL_S)
            continue
        except RuntimeError as e:
            events.emit(specialist, "sched_poll_fail", job=job.name, err=str(e)[:120])
            if time.monotonic() > deadline:
                return sched.JobStatus(name=job.name, phase="failed", exit_code=None)
            await asyncio.sleep(_POLL_INTERVAL_S)
            continue

        consecutive_missing = 0
        if status.phase != last_phase:
            events.emit(
                specialist, "sched_phase",
                job=job.name, phase=status.phase,
                prev=last_phase,
            )
            last_phase = status.phase

        if sched.is_terminal(status):
            return status
        if time.monotonic() > deadline:
            events.emit(specialist, "sched_wait_timeout", job=job.name, phase=status.phase)
            return sched.JobStatus(name=job.name, phase="failed", exit_code=None)
        await asyncio.sleep(_POLL_INTERVAL_S)


# ── Result-jsonl lookup ──────────────────────────────────────────────────────

def _find_result_jsonl(workdir: Path) -> Optional[Path]:
    """run_trial.sh + run_classify.py emit full_eval_results/<workdir-name>/run_seed0.jsonl."""
    fe = workdir / "full_eval_results"
    if not fe.is_dir():
        return None
    for sub in fe.iterdir():
        candidate = sub / "run_seed0.jsonl"
        if candidate.is_file():
            return candidate
    return None


def _find_result_log(workdir: Path) -> Optional[Path]:
    fe = workdir / "full_eval_results"
    if not fe.is_dir():
        return None
    for sub in fe.iterdir():
        candidate = sub / "run_seed0.log"
        if candidate.is_file():
            return candidate
    return None


def _double_quote_shell(path: str) -> str:
    """Double-quote a shell argument while preserving `$HOME` expansion."""
    escaped = path.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{escaped}"'


# ── Sync impl (the workhorse) ────────────────────────────────────────────────

async def _submit_trial_impl(
    specialist: str,
    hypothesis: str,
    expected_delta: str,
    parent_exp: str,
    notes: str = "",
    repo_root: Optional[str] = None,
) -> dict[str, Any]:
    """Stage → preflight → mode-dispatch → parse + record."""
    adapter = _adapter()
    if specialist not in adapter.all_domains:
        return {"ok": False, "error": f"unknown specialist {specialist!r}"}

    events.emit(specialist, "submit_trial_called",
                hypothesis=(hypothesis or "")[:80], parent_exp=parent_exp)

    workdir = config.workdir_for(specialist)
    workdir.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root) if repo_root else adapter.pkg_root

    # ── Stage + preflight (shared across modes) ─────────────────────────────
    early = await _stage_and_preflight(
        specialist, hypothesis, expected_delta, parent_exp, workdir, root,
    )
    if early is not None:
        return early

    # ── Mode dispatch ───────────────────────────────────────────────────────
    mode = adapter.submission_mode
    if mode == "job":
        exec_result = await _execute_via_job(
            specialist, parent_exp, hypothesis, expected_delta,
            workdir, adapter,
        )
    elif mode == "notebook":
        exec_result = await _execute_via_notebook(
            specialist, parent_exp, hypothesis, expected_delta,
            workdir, adapter,
        )
    elif mode == "local":
        exec_result = await _execute_via_local(
            specialist, parent_exp, hypothesis, expected_delta,
            workdir, adapter,
        )
    else:
        return {"ok": False, "error": f"unknown submission_mode {mode!r}"}

    if exec_result.get("early_return"):
        return exec_result["payload"]
    label = exec_result["label"]            # sched_job_name or notebook name or "local"
    phase = exec_result["phase"]            # "succeeded" / "failed" / "timeout"
    exit_code = exec_result["exit_code"]

    # ── Parse + record (shared) ─────────────────────────────────────────────
    return await _finalize_trial(
        specialist, hypothesis, expected_delta, parent_exp,
        workdir, label, phase, exit_code, notes, adapter,
    )


# ── Phase 2: stage + preflight (shared) ─────────────────────────────────────

async def _stage_and_preflight(
    specialist: str,
    hypothesis: str,
    expected_delta: str,
    parent_exp: str,
    workdir: Path,
    root: Path,
) -> Optional[dict[str, Any]]:
    """Run stage + syntax + size_check. Return None to continue, or an early
    result dict (payload to forward to caller) on preflight failure."""
    _stage_workdir(workdir, root)
    await asyncio.to_thread(_clear_local_trial_outputs, workdir)
    events.emit(specialist, "stage_ok", workdir=str(workdir))

    syn = _syntax_check_impl(str(workdir))
    if not syn.get("ok"):
        events.emit(specialist, "preflight_fail", reason="syntax",
                    err=(syn.get("error") or "")[:120])
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=tracker.empty_validate_row(status="preflight_crash"),
            sched_job_name="",
            workdir=workdir,
            notes=f"syntax: {syn.get('error','')[:200]}",
            keep_decision=False,
        )
        return {"ok": True, "preflight": "syntax_error", **row}

    sz = await asyncio.to_thread(_adapter().size_check, str(workdir))
    if sz.get("ok") and sz.get("verdict") == "block":
        events.emit(specialist, "preflight_block", reason="size",
                    size_bytes=sz.get("total_bytes"),
                    limit_bytes=sz.get("limit_bytes"))
        size_row = tracker.empty_validate_row(status="size_blocked")
        size_row["artifact_bytes"] = str(sz.get("total_bytes", ""))
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=size_row,
            sched_job_name="",
            workdir=workdir,
            notes=(
                f"preflight size={sz.get('total_bytes')} "
                f"limit={sz.get('limit_bytes')} "
                f"(code={sz.get('code_bytes')} "
                f"model={sz.get('model_bytes') or '?'}); "
                "no GPU time used."
            ),
            keep_decision=False,
        )
        return {"ok": True, "preflight": "size_blocked", **row}

    events.emit(specialist, "preflight_ok",
                size_bytes=sz.get("total_bytes") if sz.get("ok") else None)
    return None


# ── Phase 3a: job-mode executor (current behavior, byte-equal for PG) ───────

async def _execute_via_job(
    specialist: str,
    parent_exp: str,
    hypothesis: str,
    expected_delta: str,
    workdir: Path,
    adapter,
) -> dict[str, Any]:
    """sched sync push → sched job submit → poll → sched sync pull. Returns dict
    with either {early_return: True, payload: ...} (preflight-class fail
    that needs to be forwarded directly) or {label, phase, exit_code} on
    successful execution (parse + record happens in shared finalize)."""
    remote_name = config.remote_sync_name_for(specialist)
    remote_workdir = config.remote_workdir_for(specialist)

    # Push
    try:
        await asyncio.to_thread(sched.sync_workdir, str(workdir), remote_name)
    except Exception as e:
        events.emit(specialist, "sched_push_fail", err=str(e)[:120])
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=tracker.empty_validate_row(status="preflight_crash"),
            sched_job_name="",
            workdir=workdir,
            notes=f"sched sync push failed: {e}",
            keep_decision=False,
        )
        return {"early_return": True, "payload":
                {"ok": False, "preflight": "sched_sync_push_failed", **row}}
    events.emit(specialist, "sched_push_ok", remote=remote_name)

    # Submit
    trial_id = int(await asyncio.to_thread(tracker.next_exp_id))
    job_name = config.sched_job_name(specialist, trial_id)
    remote_workdir_sh = _double_quote_shell(remote_workdir)
    cleanup_glob = " ".join(adapter.trial_output_dirs)
    env_prefix = _pod_env_prefix(adapter)
    cmd = (
        f"cd {remote_workdir_sh} && "
        f"rm -rf {cleanup_glob} && "
        f"{env_prefix}bash {adapter.run_script} {remote_workdir_sh}"
    )
    sched_kwargs = adapter.sched_job_kwargs(specialist)
    try:
        job = await asyncio.to_thread(
            sched.submit_gpu_job,
            name=job_name,
            command=cmd,
            **sched_kwargs,
        )
    except Exception as e:
        events.emit(specialist, "sched_submit_fail", job=job_name, err=str(e)[:120])
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=tracker.empty_validate_row(status="preflight_crash"),
            sched_job_name=job_name,
            workdir=workdir,
            notes=f"sched submit failed: {e}",
            keep_decision=False,
        )
        return {"early_return": True, "payload":
                {"ok": False, "preflight": "sched_submit_failed", **row}}
    events.emit(specialist, "sched_submit", job=job_name,
                trial_id=trial_id, **sched_kwargs)
    sched.register_active_job(job_name, job.job_id)

    # Poll
    try:
        status = await _wait_for_job(job, specialist)
    finally:
        sched.unregister_active_job(job_name)
    events.emit(specialist, "sched_terminal", job=job_name, phase=status.phase)

    # Pull
    try:
        await asyncio.to_thread(sched.sync_workdir_results, remote_name, str(workdir))
    except Exception as e:
        events.emit(specialist, "sched_pull_fail", job=job_name, err=str(e)[:120])
        crash_row = tracker.empty_validate_row(status="crash")
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=crash_row,
            sched_job_name=job_name,
            workdir=workdir,
            notes=f"sched sync pull failed: {e} | sched_phase={status.phase}",
            keep_decision=False,
        )
        return {"early_return": True, "payload": {
            "ok":           False,
            "preflight":    "passed",
            "sched_phase":     status.phase,
            "sched_exit_code": status.exit_code,
            **row,
        }}
    events.emit(specialist, "sched_pull_ok", job=job_name)

    return {"label": job_name, "phase": status.phase, "exit_code": status.exit_code}


# ── Phase 3b: notebook-mode executor ────────────────────────────────────────

async def _execute_via_notebook(
    specialist: str,
    parent_exp: str,
    hypothesis: str,
    expected_delta: str,
    workdir: Path,
    adapter,
) -> dict[str, Any]:
    """Notebook ensure → sched sync push → notebook exec → sched sync pull."""
    nb_cfg = adapter.notebook_config
    if not nb_cfg or "image" not in nb_cfg:
        return {"early_return": True, "payload": {
            "ok": False,
            "error": ("submission_mode='notebook' but adapter.notebook_config "
                      "is missing 'image' (and possibly other keys). "
                      f"Got: {nb_cfg!r}"),
        }}
    nb_name = f"{config.active_sched_name_prefix()}-{specialist[:4]}-nb"
    remote_name = config.remote_sync_name_for(specialist)
    remote_workdir = config.remote_workdir_for(specialist)

    # Ensure notebook is up
    try:
        nb_id = await asyncio.to_thread(
            sched.notebook_ensure,
            nb_name,
            image=nb_cfg["image"],
            type_=nb_cfg.get("type", "h100"),
            gpus=int(nb_cfg.get("gpus", 1)),
            cpu=nb_cfg.get("cpu"),
            mem=nb_cfg.get("mem"),
            auto_stop=int(nb_cfg.get("auto_stop", 43200)),
            shm_size=nb_cfg.get("shm_size"),
        )
    except Exception as e:
        events.emit(specialist, "nb_ensure_fail", nb=nb_name, err=str(e)[:160])
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=tracker.empty_validate_row(status="preflight_crash"),
            sched_job_name=nb_name, workdir=workdir,
            notes=f"notebook ensure failed: {e}",
            keep_decision=False,
        )
        return {"early_return": True, "payload":
                {"ok": False, "preflight": "nb_ensure_failed", **row}}
    # Register IDEMPOTENTLY — same (nb_name, nb_id) across many trials. The
    # registry holds it for the supervisor's whole lifetime so the shutdown
    # chain can stop it on Ctrl+C / deadline-exit. Don't unregister after
    # exec — that would orphan the notebook for 12 h until auto_stop fires.
    sched.register_active_notebook(nb_name, nb_id)
    events.emit(specialist, "nb_ensure_ok", nb=nb_name, notebook_id=nb_id)

    # Push (notebook reads workdir from SharedFS, same as job mode)
    try:
        await asyncio.to_thread(sched.sync_workdir, str(workdir), remote_name)
    except Exception as e:
        events.emit(specialist, "sched_push_fail", err=str(e)[:120])
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=tracker.empty_validate_row(status="preflight_crash"),
            sched_job_name=nb_name, workdir=workdir,
            notes=f"sched sync push failed: {e}",
            keep_decision=False,
        )
        return {"early_return": True, "payload":
                {"ok": False, "preflight": "sched_sync_push_failed", **row}}
    events.emit(specialist, "sched_push_ok", remote=remote_name)

    # Exec
    trial_id = int(await asyncio.to_thread(tracker.next_exp_id))
    remote_workdir_sh = _double_quote_shell(remote_workdir)
    cleanup_glob = " ".join(adapter.trial_output_dirs)
    env_prefix = _pod_env_prefix(adapter)
    cmd = (
        f"cd {remote_workdir_sh} && "
        f"rm -rf {cleanup_glob} && "
        f"{env_prefix}bash {adapter.run_script} {remote_workdir_sh}"
    )
    events.emit(specialist, "nb_exec_start", nb=nb_name, trial_id=trial_id)
    try:
        output, exit_code = await asyncio.to_thread(sched.notebook_exec, nb_name, cmd)
    except Exception as e:
        events.emit(specialist, "nb_exec_fail", nb=nb_name, err=str(e)[:160])
        # Notebook stays registered — shutdown chain will stop it.
        # Don't lose the cause: write any partial output we may have.
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=tracker.empty_validate_row(status="crash"),
            sched_job_name=nb_name, workdir=workdir,
            notes=f"notebook exec failed: {e}",
            keep_decision=False,
        )
        return {"early_return": True, "payload":
                {"ok": False, "nb_exec": "failed", **row}}

    # ALWAYS persist the exec output to workdir for crash forensics. Even
    # on success this is useful (training stdout that's not in result log).
    # _finalize_trial fallbacks to this when full_eval_results is missing.
    nb_log_path = workdir / "notebook_exec.log"
    try:
        nb_log_path.write_text(output or "", encoding="utf-8", errors="replace")
    except OSError:
        pass

    phase = "succeeded" if exit_code == 0 else "failed"
    events.emit(specialist, "nb_exec_done", nb=nb_name, exit_code=exit_code,
                phase=phase)

    # Pull
    try:
        await asyncio.to_thread(sched.sync_workdir_results, remote_name, str(workdir))
    except Exception as e:
        events.emit(specialist, "sched_pull_fail", job=nb_name, err=str(e)[:120])
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=tracker.empty_validate_row(status="crash"),
            sched_job_name=nb_name, workdir=workdir,
            notes=f"sched sync pull failed: {e} | nb_phase={phase}",
            keep_decision=False,
        )
        return {"early_return": True, "payload": {
            "ok": False, "preflight": "passed",
            "sched_phase": phase, "sched_exit_code": exit_code, **row,
        }}
    events.emit(specialist, "sched_pull_ok", job=nb_name)

    return {"label": nb_name, "phase": phase, "exit_code": exit_code}


# ── Phase 3c: local-mode executor ───────────────────────────────────────────

async def _execute_via_local(
    specialist: str,
    parent_exp: str,
    hypothesis: str,
    expected_delta: str,
    workdir: Path,
    adapter,
) -> dict[str, Any]:
    """Run trial as subprocess on the launcher host. No push, no pull —
    workdir IS local. Useful for dev / smoke / no-cluster runs."""
    local_cfg = adapter.local_config
    cleanup_glob = adapter.trial_output_dirs
    # Pre-clean trial output dirs in the local workdir.
    for name in cleanup_glob:
        shutil.rmtree(workdir / name, ignore_errors=True)

    cmd = ["bash", adapter.run_script, str(workdir)]
    timeout_s = float(local_cfg.get("timeout_s", 7200))
    env_overrides = dict(adapter.pod_env_for_trial)   # task-side env first
    if "cuda_visible_devices" in local_cfg:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(local_cfg["cuda_visible_devices"])
    env_overrides["WORKDIR"] = str(workdir)

    label = f"local-{specialist[:4]}"
    log_path = workdir / "local_run.log"
    events.emit(specialist, "local_exec_start", label=label,
                cmd=" ".join(cmd), timeout_s=int(timeout_s))
    try:
        result = await run_local(
            cmd, cwd=workdir, timeout_s=timeout_s, log_path=log_path,
            env_overrides=env_overrides,
        )
    except Exception as e:
        events.emit(specialist, "local_exec_fail", label=label, err=str(e)[:160])
        row = await asyncio.to_thread(
            blackboard.record_trial,
            specialist=specialist, domain=specialist,
            parent_exp=parent_exp, hypothesis=hypothesis,
            expected_delta=expected_delta,
            validate_row=tracker.empty_validate_row(status="crash"),
            sched_job_name=label, workdir=workdir,
            notes=f"local exec exception: {e}",
            keep_decision=False,
        )
        return {"early_return": True, "payload":
                {"ok": False, "local_exec": "failed", **row}}
    events.emit(specialist, "local_exec_done", label=label,
                phase=result.phase, exit_code=result.exit_code,
                elapsed_s=int(result.elapsed_s))
    return {"label": label, "phase": result.phase, "exit_code": result.exit_code}


# ── Phase 4: parse + record (shared) ────────────────────────────────────────

async def _finalize_trial(
    specialist: str,
    hypothesis: str,
    expected_delta: str,
    parent_exp: str,
    workdir: Path,
    label: str,                 # sched_job_name / nb_name / "local-…"
    phase: str,                 # "succeeded" / "failed" / "timeout"
    exit_code: Optional[int],
    notes: str,
    adapter,
) -> dict[str, Any]:
    """Parse run_seed*.jsonl + record_trial + emit classify_done.

    Crash-excerpt log priority (first-found wins):
      1. <workdir>/full_eval_results/*/run_seed0.log       (run_trial.sh wrote it)
      2. <workdir>/notebook_exec.log                       (notebook mode early fail)
      3. <workdir>/local_run.log                           (local mode early fail)
    Without (2)+(3) fallback, notebook/local early failures (rsync error,
    missing torchvision, vendor missing) would lose all forensic trail
    because run_trial.sh never wrote a result-dir log.
    """
    def _resolve_excerpt_log() -> Optional[Path]:
        """Find the best log for crash-excerpt extraction, in priority order."""
        primary = _find_result_log(workdir)
        if primary is not None and primary.is_file() and primary.stat().st_size > 0:
            return primary
        # Fallbacks: mode-specific outer logs (only present when mode-exec
        # ran; missing for failed-before-exec paths, which is fine).
        for fallback in ("notebook_exec.log", "local_run.log"):
            p = workdir / fallback
            if p.is_file() and p.stat().st_size > 0:
                return p
        return primary       # may still be None

    jsonl = _find_result_jsonl(workdir)
    if jsonl is None:
        log_path = _resolve_excerpt_log()
        excerpt = ""
        if log_path is not None:
            excerpt = (await asyncio.to_thread(tracker.extract_crash_excerpt, log_path)) or ""
        validate_row = tracker.empty_validate_row(status="crash")
        notes_final = notes or ""
        if excerpt:
            notes_final = f"{notes_final} | {excerpt}" if notes_final else excerpt
    else:
        validate_row = await asyncio.to_thread(tracker.parse_validate_result, jsonl)
        notes_final = notes
        if validate_row["status"] in ("crash", "preflight_crash"):
            log_path = _resolve_excerpt_log()
            if log_path is not None:
                excerpt = await asyncio.to_thread(tracker.extract_crash_excerpt, log_path)
                if excerpt:
                    notes_final = f"{notes_final} | {excerpt}" if notes_final else excerpt

    row = await asyncio.to_thread(
        blackboard.record_trial,
        specialist=specialist, domain=specialist,
        parent_exp=parent_exp, hypothesis=hypothesis,
        expected_delta=expected_delta,
        validate_row=validate_row,
        sched_job_name=label,
        workdir=workdir,
        notes=notes_final,
    )

    score_field = adapter.score_field
    events.emit(specialist, "classify_done",
                exp_id=row.get("exp_id"),
                status=row.get("status"),
                **{score_field: row.get(score_field)},
                delta=row.get("delta_vs_best"))

    return {
        "ok":           True,
        "preflight":    "passed",
        "sched_phase":     phase,
        "sched_exit_code": exit_code,
        **row,
    }


# ── Async @tool wrapper (SDK-facing) ─────────────────────────────────────────

@tool(
    "submit_trial",
    (
        "Submit this specialist's current train_gpt.py to a real 8×H100 "
        "evaluation via SCHED. Runs a local syntax + size preflight first — "
        "failures are recorded WITHOUT burning GPU time. On success, blocks "
        "until the job finishes, then writes a row to the blackboard TSV and "
        "returns {exp_id, status, val_bpb, delta_vs_best, artifact_bytes, "
        "train_s, eval_s, total_s, snapshot_path, sched_job_name, notes}. "
        "Status is one of: keep | discard | crash | size_blocked | "
        "preflight_crash | eval_budget_overrun | train_budget_overrun."
    ),
    {
        "type": "object",
        "properties": {
            "specialist":     {"type": "string",
                               "description": "Your domain key, e.g. 'arch'."},
            "hypothesis":     {"type": "string",
                               "description": "One-sentence description of what you changed."},
            "expected_delta": {"type": "string",
                               "description": "Signed estimate, e.g. '-0.002'."},
            "parent_exp":     {"type": "string",
                               "description": "exp_id you rooted from (usually best.json)."},
            "notes":          {"type": "string",
                               "description": "Optional free-form rationale.",
                               "default": ""},
        },
        "required": ["specialist", "hypothesis", "expected_delta", "parent_exp"],
    },
)
async def submit_trial(args: dict[str, Any]) -> dict[str, Any]:
    result = await _submit_trial_impl(
        specialist=args["specialist"],
        hypothesis=args["hypothesis"],
        expected_delta=args["expected_delta"],
        parent_exp=args["parent_exp"],
        notes=args.get("notes", ""),
    )
    return _mcp(result)
