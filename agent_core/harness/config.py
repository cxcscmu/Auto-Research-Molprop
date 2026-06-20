"""Path constants and system-wide knobs for the multi-agent harness — task-agnostic.

The laptop-supervisor topology uses two distinct filesystem concepts:

  * local disk      — opened directly by Python for blackboard + workdirs
  * remote SharedFS     — referenced only as strings in `sched sync` targets and
                      `sched job create --command` arguments

Nothing in this module touches the filesystem at import time *except* loading
`<task_pkg>/swarm_config.json` (and that's done lazily on first read of
SCHED_TASK_PRIORITY / MODEL_DEFAULT). Call `ensure_dirs()` once from the
supervisor before any specialist starts.

Task-specific extension
───────────────────────
This module reads task-specific knobs (PKG_ROOT, all_domains, baseline
filename) via `agent_core.current_adapter()`. Each task package
(`multi_agent_pg`, `multi_agent_nc`, `multi_agent_cifar`) registers a
`TaskAdapter` on import so this module can resolve those.

PG-specific path defaults (VENV_PATH, DATA_SRC) and the task's specialist
tuple (DOER_DOMAINS / ANALYST_DOMAINS / ALL_DOMAINS) live in the task
package's own `harness/config.py`, which re-exports from this module
plus adds the task constants.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    """Read an integer environment override, with a clear error on invalid input."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


# ── Filesystem roots ──────────────────────────────────────────────────────────

# Local root — opened directly by Python. Holds the authoritative blackboard,
# local snapshots, and the specialists' editable workdirs. Default name is
# task-agnostic ("remote_dev"); operators can override per machine.
LOCAL_ROOT = Path(os.environ.get(
    "MAGENT_LOCAL_ROOT",
    str(Path.home() / "remote_dev"),
)).expanduser()

# Remote SharedFS root as seen from the pod. This is intentionally a plain string,
# not a Path: laptop-side Python never opens it. The default uses `$HOME` so
# the pod's shell expands it at job runtime.
REMOTE_ROOT = os.environ.get(
    "MAGENT_REMOTE_ROOT",
    "$REMOTE_ROOT",
)

# SCHED sync target prefix — the `X` in `sched sync :X/ ...`, resolved by SCHED as a
# path under the pod-side SharedFS tree.
REMOTE_SYNC_PREFIX = os.environ.get(
    "MAGENT_REMOTE_SYNC_PREFIX",
    "remote_dev",
)

# VENV_PATH and DATA_SRC are TASK-SPECIFIC defaults — defined in the task
# package's own harness/config.py (e.g. multi_agent_pg/harness/config.py
# defaults VENV_PATH to ~/auto-parameter-golf/venv).

# ── Blackboard layout ─────────────────────────────────────────────────────────

BLACKBOARD_DIR = LOCAL_ROOT / "blackboard"
WORKDIRS_ROOT  = LOCAL_ROOT / "workdirs"
SNAPSHOTS_DIR  = BLACKBOARD_DIR / "snapshots"
LOCKS_DIR      = BLACKBOARD_DIR / "locks"

LEADERBOARD_MD = BLACKBOARD_DIR / "LEADERBOARD.md"
KNOWLEDGE_MD   = BLACKBOARD_DIR / "KNOWLEDGE.md"
TREE_TSV       = BLACKBOARD_DIR / "tree.tsv"
RESULTS_TSV    = BLACKBOARD_DIR / "results.tsv"
BEST_JSON      = BLACKBOARD_DIR / "best.json"
STOP_FLAG      = BLACKBOARD_DIR / "stop.flag"

# ── SCHED job naming ─────────────────────────────────────────────────────────────

SCHED_NAME_PREFIX = "apg"            # PG default; tasks override via adapter.sched_name_prefix
#
# Each task picks its own short prefix (≤ 4 chars) so dashboards / `sched job
# list` distinguish tasks at a glance and shutdown chains can't cross-kill.
# PG keeps "apg" for back-compat with existing job_ids in production.
# CIFAR uses "cif", NC uses "nc". `sched_job_name(...)` reads adapter at call
# time; this constant is only the fallback if no adapter is registered.

# ── Swarm config (file-driven, editable without exports) ─────────────────────
#
# `<task_pkg>/swarm_config.json` is the canonical place to tune per-specialist
# knobs that would otherwise need a fleet of env vars. Used for the
# per-specialist SCHED priority tier and Claude model selection. The file lives
# next to train_gpt.py and is committed with the package so a deployment can
# edit it in-place. Resolution order for each knob:
#   1. swarm_config.json value, if present
#   2. matching MAGENT_* env var (back-compat)
#   3. hard default
# Missing file: silent (all defaults). File present but unparseable: logged
# warning + treated as empty, never abort — supervisor must keep running.


def _load_swarm_config() -> dict:
    """Read swarm_config.json from the active task package's pkg_root.

    Lazy import of current_adapter so this module can be imported before
    a task adapter is registered (early supervisor bootstrap, tests).
    """
    from agent_core import current_adapter
    try:
        path = current_adapter().pkg_root / "swarm_config.json"
    except RuntimeError:
        # No adapter registered — return empty config (caller falls back to
        # MAGENT_* env vars and hard defaults).
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        logging.getLogger(__name__).warning(
            "swarm_config.json present at %s but unreadable (%s); "
            "falling back to env vars / hard defaults",
            path, e,
        )
        return {}
    return data if isinstance(data, dict) else {}


# Lazy swarm_config loading — adapter may not be registered when this module
# first imports. We compute on first access and cache; if the adapter isn't
# registered yet, _load_swarm_config returns {} and we KEEP it uncached so a
# later access (after adapter registration) re-evaluates against the file.

_SWARM_CFG_CACHE: "dict | None" = None


def _swarm_cfg() -> dict:
    """Lazy + adapter-aware load of swarm_config.json.

    First call after a task adapter is registered loads from
    `current_adapter().pkg_root / "swarm_config.json"` and caches.
    Calls before adapter registration return {} but DON'T cache, so a later
    call (post-registration) actually finds the file.
    """
    global _SWARM_CFG_CACHE
    if _SWARM_CFG_CACHE is not None:
        return _SWARM_CFG_CACHE
    from agent_core import _active_adapter
    if _active_adapter is None:
        # Adapter not yet registered — return empty WITHOUT caching, so the
        # next call (post-registration) succeeds.
        return {}
    cfg = _load_swarm_config()
    _SWARM_CFG_CACHE = cfg
    return cfg


SCHED_PRIORITY_PROD = 10


def sched_priority_for(specialist: str) -> int:
    """Return the effective SCHED priority for a given specialist key.

    Lookup: swarm_config.json overrides → swarm_config.json default →
    `MAGENT_SCHED_TASK_PRIORITY` env → 10.
    """
    cfg = _swarm_cfg()
    sched_cfg = cfg.get("sched_priority", {}) if isinstance(cfg.get("sched_priority", {}), dict) else {}
    overrides = {
        str(k): int(v)
        for k, v in (sched_cfg.get("overrides") or {}).items()
    }
    if specialist in overrides:
        return overrides[specialist]
    return int(sched_cfg.get("default", _env_int("MAGENT_SCHED_TASK_PRIORITY", 10)))


def sched_type_for(specialist: str) -> str:
    """Return the effective SCHED pool type for a given specialist (e.g.
    "h100", "h200"). Mirrors sched_priority_for / model_for shape: per-spec
    overrides → default → "h100".

    Used to flip an entire task swarm between H100 and H200 by editing
    `swarm_config.json` without touching code or env. Both calibrate +
    runtime submit paths read this same config file.
    """
    cfg = _swarm_cfg()
    type_cfg = cfg.get("sched_type", {}) if isinstance(cfg.get("sched_type", {}), dict) else {}
    overrides = {
        str(k): str(v)
        for k, v in (type_cfg.get("overrides") or {}).items()
    }
    if specialist in overrides:
        return overrides[specialist]
    return str(type_cfg.get("default", "h100"))


def model_for(specialist: str) -> str:
    """Return the effective Claude model id for a given specialist.

    Lookup: swarm_config.json overrides → swarm_config.json default →
    "claude-sonnet-4-6".
    """
    cfg = _swarm_cfg()
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    overrides = {
        str(k): str(v)
        for k, v in (model_cfg.get("overrides") or {}).items()
    }
    if specialist in overrides:
        return overrides[specialist]
    return str(model_cfg.get("default", "claude-sonnet-4-6"))


# Module-level back-compat: legacy callers do `config.SCHED_TASK_PRIORITY` etc.
# Resolve via __getattr__ so the lookup goes through _swarm_cfg() at access
# time, which works even when this module was imported before the adapter
# was registered.

_LAZY_SWARM_ATTRS = frozenset({
    "SCHED_TASK_PRIORITY", "SCHED_PRIORITY_EXPLORE", "SCHED_PRIORITY_OVERRIDES",
    "MODEL_DEFAULT", "MODEL_OVERRIDES",
    "_SWARM_CFG", "_SCHED_PRIO_CFG", "_MODEL_CFG",
})


def _lazy_attr(name: str):
    cfg = _swarm_cfg()
    sched_cfg = cfg.get("sched_priority", {}) if isinstance(cfg.get("sched_priority", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    if name == "_SWARM_CFG":
        return cfg
    if name == "_SCHED_PRIO_CFG":
        return sched_cfg
    if name == "_MODEL_CFG":
        return model_cfg
    if name == "SCHED_TASK_PRIORITY":
        return int(sched_cfg.get("default", _env_int("MAGENT_SCHED_TASK_PRIORITY", 10)))
    if name == "SCHED_PRIORITY_EXPLORE":
        return int(sched_cfg.get("default", _env_int("MAGENT_SCHED_TASK_PRIORITY", 10)))
    if name == "SCHED_PRIORITY_OVERRIDES":
        return {str(k): int(v) for k, v in (sched_cfg.get("overrides") or {}).items()}
    if name == "MODEL_DEFAULT":
        return str(model_cfg.get("default", "claude-sonnet-4-6"))
    if name == "MODEL_OVERRIDES":
        return {str(k): str(v) for k, v in (model_cfg.get("overrides") or {}).items()}
    raise AttributeError(name)


def __getattr__(name: str):
    """Module-level lazy accessor for swarm_config-derived constants."""
    if name in _LAZY_SWARM_ATTRS:
        return _lazy_attr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── bwrap sandbox probe ─────────────────────────────────────────────────────
#
# Resolution for "should sandbox be disabled?":
#   1. MAGENT_DISABLE_SANDBOX=1/true/yes/on  → disable (skip probe + container check)
#   2. MAGENT_DISABLE_SANDBOX=0/false/no/off → enable  (skip probe + container check)
#   3. unset                                  → probe + container detection;
#      auto-disable if EITHER probe fails OR running inside LXC / container
#
# Why two-signal: the bare bwrap probe (T5 pattern) detects whether basic
# unprivileged-userns + mount works. But the SDK's bundled CLI uses a
# pivot_root + nested proc-mount pattern (`--bind / /newroot --proc
# /newroot/proc --chdir /newroot`) that hits a DIFFERENT kernel code path.
# In LXC, observed empirically (2026-04-25): bare bwrap probe passes, but
# SDK's pivot+proc-mount fails INTERMITTENTLY with `Can't mount proc on
# /newroot/proc: Operation not permitted` — sometimes succeeds, sometimes
# EPERM under multi-agent contention. Auto-disabling on container detection
# is the only reliable way to avoid this flake; operator can force-enable
# via MAGENT_DISABLE_SANDBOX=0 if they've verified their LXC config works.

_sandbox_decision: "bool | None" = None

# Container virt types where SDK's nested-userns proc-mount is known/likely
# flaky. Bare-metal / KVM / VMware / Xen do NOT need this guard — they have
# full kernel namespace capability.
_CONTAINER_VIRT_DISABLE = frozenset({
    "lxc", "lxc-libvirt", "docker", "podman",
    "openvz", "wsl", "rkt", "systemd-nspawn",
    "container-other",
})


def _bwrap_pivot_proc_works() -> bool:
    """Probe whether bwrap can do its standard sandbox setup.

    Tests: `bwrap --bind / / --dev /dev --proc /proc --tmpfs /tmp
    /usr/bin/true` — the typical mount profile the SDK bundled CLI
    needs (host fs, fresh /dev, fresh /proc, fresh /tmp). Empirically
    matches the SDK's behaviour better than the earlier
    `--bind / /newroot --chdir /newroot` pattern, which gave false
    negatives because `/bin/true` lookup failed at the new root view
    independent of whether proc-mount actually worked.

    Returns True iff the 5-second self-test exits 0. Anything else
    (bwrap missing, timeout, non-zero exit, OSError) returns False —
    the safe default is "assume sandbox is broken, disable it".
    """
    import shutil
    import subprocess
    log = logging.getLogger(__name__)
    bwrap_path = shutil.which("bwrap")
    if not bwrap_path:
        log.warning("[bwrap probe] bwrap binary NOT FOUND in PATH")
        return False
    cmd = [
        "bwrap",
        "--bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "/usr/bin/true",
    ]
    log.info("[bwrap probe] testing: %s  (binary=%s)", " ".join(cmd), bwrap_path)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=5, text=True)
    except subprocess.TimeoutExpired:
        log.warning("[bwrap probe] result: TIMEOUT after 5s")
        return False
    except OSError as e:
        log.warning("[bwrap probe] result: OSError — %s", e)
        return False
    if r.returncode == 0:
        log.info("[bwrap probe] result: OK  (rc=0, sandbox available)")
        return True
    stderr_lines = [line for line in (r.stderr or "").splitlines() if line.strip()]
    tail = "  |  ".join(stderr_lines[-2:])[:240] if stderr_lines else "(no stderr)"
    log.warning("[bwrap probe] result: FAILED  (rc=%d)  stderr: %s",
                r.returncode, tail)
    return False


def _detect_container_virt() -> str:
    """Return the systemd-detect-virt value, or "" if undetectable.

    Uses systemd-detect-virt when available; falls back to checking
    /proc/1/cgroup for "lxc" or "docker" tokens. Empty string means
    "could not determine" (defaults to bare-metal / safe-to-enable).
    """
    import shutil
    import subprocess
    if shutil.which("systemd-detect-virt"):
        try:
            r = subprocess.run(
                ["systemd-detect-virt"],
                capture_output=True, timeout=2, text=True,
            )
            return (r.stdout or "").strip().lower()
        except (subprocess.TimeoutExpired, OSError):
            pass
    # Fallback: cgroup membership often reveals lxc/docker
    try:
        with open("/proc/1/cgroup") as f:
            cg = f.read().lower()
        for token in ("lxc", "docker", "podman", "kubepods"):
            if token in cg:
                return token
    except OSError:
        pass
    return ""


def should_disable_sandbox() -> bool:
    """Return True if SDK sandbox should be off for this run.

    Cached per process — probe + detection run at most once. Operator
    can force either direction via MAGENT_DISABLE_SANDBOX env var;
    absent that, two-signal logic: bwrap probe AND container-virt check
    both must pass to enable.
    """
    global _sandbox_decision
    if _sandbox_decision is not None:
        return _sandbox_decision

    log = logging.getLogger(__name__)
    env = os.environ.get("MAGENT_DISABLE_SANDBOX", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        _sandbox_decision = True
        log.warning(
            "[sandbox] DECISION: DISABLED  (operator forced via MAGENT_DISABLE_SANDBOX=%s)",
            env,
        )
        return _sandbox_decision
    if env in ("0", "false", "no", "off"):
        _sandbox_decision = False
        log.info(
            "[sandbox] DECISION: ENABLED  (operator forced via MAGENT_DISABLE_SANDBOX=%s)",
            env,
        )
        return _sandbox_decision

    # Auto: probe + container detection. Both must pass to enable.
    works = _bwrap_pivot_proc_works()
    virt = _detect_container_virt()
    in_container = virt in _CONTAINER_VIRT_DISABLE
    log.info(
        "[container detect] systemd-detect-virt → %r  (problematic-container=%s)",
        virt or "(unknown)", in_container,
    )

    if not works:
        _sandbox_decision = True
        log.warning(
            "[sandbox] DECISION: DISABLED  (auto, bwrap probe failed). "
            "agents/hooks.block_bash_writes hook is the compensating control. "
            "Set MAGENT_DISABLE_SANDBOX=0 to force-enable.",
        )
    elif in_container:
        _sandbox_decision = True
        log.warning(
            "[sandbox] DECISION: DISABLED  (auto, bwrap probe OK BUT running "
            "inside %r where the SDK's pivot_root + nested proc-mount is "
            "intermittently denied by kernel under multi-agent load — "
            "observed 2026-04-25: probe passes, SDK call hits EPERM "
            "'Cant mount proc on /newroot/proc' under contention). "
            "agents/hooks.block_bash_writes hook is the compensating control. "
            "Set MAGENT_DISABLE_SANDBOX=0 to force-enable if you've verified "
            "the SDK actually works in your specific container config.",
            virt,
        )
    else:
        _sandbox_decision = False
        log.info(
            "[sandbox] DECISION: ENABLED  (auto, probe OK + non-container "
            "environment %r)",
            virt or "(unknown)",
        )
    return _sandbox_decision


# Some 8xH100 nodes in this project expose less than sched's generic defaults.
# Keep the submit path explicit so jobs are schedulable across the smaller nodes.
SCHED_H100_CPU = _env_int("MAGENT_SCHED_H100_CPU", 118)
SCHED_H100_MEM_GIB = _env_int("MAGENT_SCHED_H100_MEM_GIB", 1580)


def active_sched_name_prefix() -> str:
    """Return the SCHED name prefix that all naming + ownership checks must use.

    Single source of truth so sched_job_name (job creation),
    sched._is_apg_owned_name / _is_apg_owned_name_loose (shutdown chain),
    and tools/submit.py (notebook naming) can't drift.

    Resolution order:
      1. MAGENT_SCHED_NAME_PREFIX env var (set by supervisor's
         --sched-name-prefix CLI, see __main__:_apply_remote_isolation_from_argv).
         Lets two supervisors of the same task run in parallel without
         colliding on job names — pass --sched-name-prefix apga to one and
         apgb to the other so SCHED sees them as distinct and SIGINT
         cleanup regexes don't cross-kill.
      2. The active task adapter's `sched_name_prefix` (PG="apg",
         CIFAR="cif", NC="nc").
      3. Module-level SCHED_NAME_PREFIX="apg" fallback when no adapter
         is registered.
    """
    override = os.environ.get("MAGENT_SCHED_NAME_PREFIX", "").strip()
    if override:
        return override
    try:
        from agent_core import current_adapter
        return current_adapter().sched_name_prefix
    except (RuntimeError, AttributeError):
        return SCHED_NAME_PREFIX


def sched_job_name(domain: str, trial_id: int) -> str:
    """Return `<prefix>-<domain[:4]>-NNNN` — fits SCHED's 30-char name limit.

    `trial_id` is the monotonic counter from blackboard.next_exp_id();
    zero-padded to four digits so the name sorts lexically.
    Prefix comes from `active_sched_name_prefix()`.
    """
    if trial_id < 0 or trial_id > 9999:
        raise ValueError(f"trial_id {trial_id} out of [0, 9999]")
    return f"{active_sched_name_prefix()}-{domain[:4]}-{trial_id:04d}"


# ── Session model ────────────────────────────────────────────────────────────

DOER_THINKING_BUDGET_TOKENS   = 8_000
ANALYST_THINKING_BUDGET_TOKENS = 4_000

# Termination: stop when wall-clock exceeds DEADLINE OR when no new score
# improvement for NO_IMPROVEMENT_GRACE_S. OR-semantics, not AND.
DEADLINE_HOURS              = 48
NO_IMPROVEMENT_GRACE_S      = 4 * 3600

# Run-trial subprocess soft limit — larger than single_agent because run_trial
# does train + pack + eval/TTT, each with its own official budget.
TRIAL_WALL_BUDGET_S         = 2_100


def ensure_dirs() -> None:
    """Create blackboard scaffolding. Idempotent; safe to call at every startup."""
    for d in (BLACKBOARD_DIR, WORKDIRS_ROOT, SNAPSHOTS_DIR, LOCKS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _all_domains() -> tuple[str, ...]:
    """Resolve all_domains via the active task adapter."""
    from agent_core import current_adapter
    return current_adapter().all_domains


def workdir_for(domain: str) -> Path:
    """Return (but do not create) the specialist's private workdir."""
    all_doms = _all_domains()
    if domain not in all_doms:
        raise ValueError(f"unknown domain {domain!r}, must be one of {all_doms}")
    return WORKDIRS_ROOT / f"workdir_{domain}"


def remote_workdir_for(domain: str) -> str:
    """Return the pod-visible absolute workdir path for one specialist."""
    all_doms = _all_domains()
    if domain not in all_doms:
        raise ValueError(f"unknown domain {domain!r}, must be one of {all_doms}")
    return f"{REMOTE_ROOT}/workdirs/workdir_{domain}"


def remote_sync_name_for(domain: str) -> str:
    """Return the SCHED sync target name for one specialist workdir."""
    all_doms = _all_domains()
    if domain not in all_doms:
        raise ValueError(f"unknown domain {domain!r}, must be one of {all_doms}")
    return f"{REMOTE_SYNC_PREFIX}/workdirs/workdir_{domain}"
