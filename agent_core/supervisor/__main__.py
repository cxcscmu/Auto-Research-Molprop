"""CLI entrypoint for the supervisor.

Usage (head node):

    # Default: full swarm — all 9 doers + meta analyst. Per-specialist SCHED
    # priority comes from multi_agent/swarm_config.json (arch/opt/quant/meta
    # at 10, the other six at 9 out of the box).
    python -m multi_agent.supervisor \
        --deadline-hours 48 \
        --no-improvement-hours 4

    # Narrower subset (debugging, or to match past runs):
    python -m multi_agent.supervisor \
        --specialists arch,opt,quant,meta

Reads the blackboard, bootstraps a baseline row if none exists, spawns one
coroutine per specialist with a launch stagger, runs until termination,
prints an end-of-run summary.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import logging
import math
import os
import signal
import sys
from dataclasses import asdict
from typing import Optional


def _finite_float(s: str) -> float:
    """argparse type: float, but reject nan/inf so the CLI override path
    can't poison best.json / results.tsv. Mirrors the calibrate_baseline
    helper's check; both layers are kept on purpose."""
    v = float(s)
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(
            f"baseline score must be finite, got {s!r}"
        )
    return v

# Harness imports are DEFERRED. Reason: core.harness.config reads
# MAGENT_LOCAL_ROOT / MAGENT_REMOTE_ROOT / MAGENT_REMOTE_SYNC_PREFIX at
# module-import time, which freezes those constants. nc/cifar __init__.py
# uses os.environ.setdefault to install task-specific defaults; if we
# imported config here, those setdefaults would arrive too late and the
# direct path (`MAGENT_TASK=cifar python -m agent_core.supervisor`)
# would write into PG's blackboard. Imports happen inside main() after
# _ensure_task_package_registered.

_LOG = logging.getLogger("multi_agent.supervisor.main")
_SHUTDOWN_COUNT = 0


# ── Task package selection ──────────────────────────────────────────────────
#
# When core.supervisor is invoked directly (`python -m agent_core.supervisor`)
# we need to import the task package BEFORE building argparse defaults
# (those default values come from the active adapter). Resolution order:
#
#   1. `--task <name>` CLI flag (if present in argv before parser construction)
#   2. `MAGENT_TASK` env var
#   3. None — caller already imported a task package (e.g. wrapper at
#      multi_agent_pg/supervisor/__main__.py imports multi_agent_pg before
#      forwarding to core.main)
#
# `_ensure_task_package_registered` resolves this once at the top of main().

_TASK_PKG_MAP = {
    "pg":     "multi_agent_pg",
    "nc":     "multi_agent_nc",
    "cifar":  "multi_agent_cifar",
}


def _peek_task_arg(argv: Optional[list[str]]) -> Optional[str]:
    """Return the value of `--task X` from argv (or `--task=X`), without
    consuming it from argparse. None if absent."""
    args = argv if argv is not None else sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--task" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--task="):
            return a.split("=", 1)[1]
    return None


def _peek_state_root_arg(argv: Optional[list[str]]) -> Optional[str]:
    """Return the value of `--state-root X` from argv (or `--state-root=X`),
    without consuming it from argparse. None if absent.

    Same pre-parse pattern as `_peek_task_arg`. Must run BEFORE
    `_ensure_task_package_registered` so the task package's __init__.py
    `os.environ.setdefault('MAGENT_LOCAL_ROOT', ...)` becomes a no-op
    (we want CLI value to win).
    """
    args = argv if argv is not None else sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--state-root" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--state-root="):
            return a.split("=", 1)[1]
    return None


def _apply_state_root_from_argv(argv: Optional[list[str]]) -> None:
    """Pre-parse --state-root from argv and overwrite MAGENT_LOCAL_ROOT.

    Uses os.environ[k] = v (not setdefault) so the CLI value wins over
    any prior shell env. Idempotent: re-applying the same value is a
    no-op. Called from main() before any harness import.
    """
    val = _peek_state_root_arg(argv)
    if val:
        os.environ["MAGENT_LOCAL_ROOT"] = os.path.expanduser(val)


def _peek_named_arg(argv: Optional[list[str]], name: str) -> Optional[str]:
    """Generic argv pre-peek for `--<name> X` and `--<name>=X` forms."""
    args = argv if argv is not None else sys.argv[1:]
    for i, a in enumerate(args):
        if a == f"--{name}" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


def _apply_remote_isolation_from_argv(argv: Optional[list[str]]) -> None:
    """Pre-parse --remote-root / --remote-sync-prefix / --sched-name-prefix
    and write them to MAGENT_REMOTE_ROOT / MAGENT_REMOTE_SYNC_PREFIX /
    MAGENT_SCHED_NAME_PREFIX. Same pre-parse pattern as state-root.

    Why this exists: --state-root only isolates ccbox-LOCAL paths
    (blackboard + local workdirs). The actual GPU runs happen via
    `sched sync` to SharedFS, with paths derived from MAGENT_REMOTE_ROOT and
    MAGENT_REMOTE_SYNC_PREFIX. Two PG supervisors running in parallel
    with isolated --state-root but DEFAULT remote/sync env would BOTH
    sync to `$REMOTE_ROOT/workdirs/workdir_<spec>/` on SharedFS,
    overwriting each other and corrupting trial results. Same for
    SCHED job name prefix: two PG runs both default to "apg-", causing
    same-named jobs and shutdown-chain cross-kills.

    Operator must pass all three (or rely on shell env) for true A/B
    isolation. This helper just plumbs the CLI flags into env vars
    early so harness.config picks them up at import time.
    """
    rr = _peek_named_arg(argv, "remote-root")
    if rr:
        os.environ["MAGENT_REMOTE_ROOT"] = rr
    rsp = _peek_named_arg(argv, "remote-sync-prefix")
    if rsp:
        os.environ["MAGENT_REMOTE_SYNC_PREFIX"] = rsp
    schedp = _peek_named_arg(argv, "sched-name-prefix")
    if schedp:
        os.environ["MAGENT_SCHED_NAME_PREFIX"] = schedp


def _ensure_task_package_registered(argv: Optional[list[str]]) -> None:
    """Import the active task package so `register_task_adapter` runs.

    No-op if an adapter is already registered (e.g. caller imported
    multi_agent_pg first). Otherwise resolves: --task / MAGENT_TASK /
    raise.
    """
    from agent_core import _active_adapter
    if _active_adapter is not None:
        return
    name = _peek_task_arg(argv) or os.environ.get("MAGENT_TASK", "").strip().lower()
    if not name:
        raise SystemExit(
            "no task package registered. Set MAGENT_TASK=pg|nc|cifar, "
            "pass --task <name>, or invoke via a task wrapper "
            "(e.g. `python -m multi_agent_pg.supervisor`)"
        )
    pkg = _TASK_PKG_MAP.get(name)
    if pkg is None:
        raise SystemExit(
            f"unknown --task / MAGENT_TASK value {name!r} "
            f"(known: {sorted(_TASK_PKG_MAP)})"
        )
    __import__(pkg)   # triggers register_task_adapter inside the task pkg


def _cleanup_active_sched_jobs() -> None:
    """Best-effort: stop every SCHED resource (job + notebook) this process
    has registered as in-flight. Called from SIGINT/SIGTERM handlers and
    from atexit so Ctrl+C / kill / normal exit all converge on a clean
    cluster state. Safe to call repeatedly — registries drain atomically.

    Job stops address by `job_id` (not name); notebook stops by name
    (notebook ids are kept in registry too but `sched notebook stop` takes
    name).
    """
    from ..harness import sched

    # ── Jobs (sched.submit_gpu_job-spawned) ─────────────────────────────────────
    job_pairs = sched.snapshot_active_jobs()
    if job_pairs:
        _LOG.warning(
            "shutdown: stopping %d active SCHED job(s): %s",
            len(job_pairs),
            ", ".join(f"{n}[{jid}]" for n, jid in job_pairs),
        )
        stopped_jobs = sched.stop_all_active_jobs()
        if stopped_jobs:
            _LOG.warning(
                "shutdown: sched job stop issued for: %s",
                ", ".join(f"{n}[{jid}]" for n, jid in stopped_jobs),
            )
        else:
            _LOG.warning("shutdown: sched job stop issued for: (none)")

    # ── Notebooks (sched.notebook_ensure-spawned) ───────────────────────────────
    nb_pairs = sched.snapshot_active_notebooks()
    if nb_pairs:
        _LOG.warning(
            "shutdown: stopping %d active SCHED notebook(s): %s",
            len(nb_pairs),
            ", ".join(f"{n}[{nbid}]" for n, nbid in nb_pairs),
        )
        stopped_nbs = sched.stop_all_active_notebooks()
        if stopped_nbs:
            _LOG.warning(
                "shutdown: sched notebook stop issued for: %s",
                ", ".join(f"{n}[{nbid}]" for n, nbid in stopped_nbs),
            )

    if not job_pairs and not nb_pairs:
        # Nothing to clean — common when supervisor exits normally
        # after run.deadline, all in-flight already drained.
        return


def _signal_handler(signum: int, _frame) -> None:
    """SIGINT/SIGTERM handler: stop in-flight SCHED jobs, write stop.flag,
    then re-raise KeyboardInterrupt so asyncio unwinds the coroutine tree.
    A second signal bypasses cleanup and force-exits (useful if the first
    cleanup is itself hanging).
    """
    global _SHUTDOWN_COUNT
    _SHUTDOWN_COUNT += 1
    sig_name = signal.Signals(signum).name
    if _SHUTDOWN_COUNT >= 2:
        sys.stderr.write(
            f"\n[shutdown] second {sig_name} received — force exit without "
            f"further sched cleanup\n"
        )
        sys.stderr.flush()
        os._exit(130)

    sys.stderr.write(
        f"\n[shutdown] caught {sig_name}; stopping active GPU jobs before "
        f"exit (press {sig_name} again to force-exit)…\n"
    )
    sys.stderr.flush()

    # Drop a stop.flag so the supervisor's own "should I launch another iter"
    # gates see it immediately (harmless if the flag already exists).
    try:
        from ..harness import config
        config.STOP_FLAG.write_text(f"{sig_name} received", encoding="utf-8")
    except OSError:
        pass

    _cleanup_active_sched_jobs()
    # Hand control back to the main thread by raising — KeyboardInterrupt
    # propagates through asyncio.run and unwinds all coroutines.
    raise KeyboardInterrupt(f"{sig_name} received")


def _install_shutdown_hooks() -> None:
    """Install SIGINT + SIGTERM handlers and an atexit fallback.

    SIGINT (Ctrl+C from the tmux pane) and SIGTERM (systemd / kill) both
    run `_signal_handler`. atexit is a last-resort catch-all for the case
    where the process exits via an unhandled exception, sys.exit, or normal
    completion — it re-issues sched stop on anything still in the registry.
    """
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_cleanup_active_sched_jobs)


def _all_domains() -> tuple[str, ...]:
    from agent_core import current_adapter
    return current_adapter().all_domains


def _build_parser() -> argparse.ArgumentParser:
    # All harness imports must be deferred until task is registered (see top).
    from agent_core import current_adapter
    from ..harness import config
    from . import core
    adapter = current_adapter()

    p = argparse.ArgumentParser(description="Run the multi-agent swarm supervisor.")
    p.add_argument(
        "--task",
        choices=sorted(_TASK_PKG_MAP),
        default=None,
        help=(
            "Task package selector: pg (Parameter Golf) / nc (nanochat) / "
            "cifar (cifar-airbench). Equivalent to MAGENT_TASK env. "
            "Required when invoking core.supervisor directly; ignored if "
            "you've already imported a task package (e.g. via the "
            "`multi_agent_pg.supervisor` wrapper)."
        ),
    )
    _doer_count    = len(adapter.doer_domains)
    _analyst_count = len(adapter.analyst_domains)
    _all_count     = _doer_count + _analyst_count
    _analyst_str   = (f" + {_analyst_count} analyst{'s' if _analyst_count != 1 else ''}"
                      if _analyst_count else " (no analysts)")
    p.add_argument(
        "--specialists",
        default=",".join(_all_domains()),
        help=(
            f"Comma-separated specialist keys (subset of ALL_DOMAINS = "
            f"DOER_DOMAINS + ANALYST_DOMAINS). Default is the full "
            f"{_all_count}-agent swarm ({_doer_count} doers{_analyst_str}); "
            f"per-specialist SCHED priority comes from "
            f"{adapter.pkg_root.name}/swarm_config.json. Pass a narrower "
            f"CSV to debug a subset."
        ),
    )
    p.add_argument(
        "--max-trials", type=int, default=None, metavar="N",
        help=(
            "PRIMARY termination: stop after N completed trial evaluations "
            "(counts all statuses: OK + CRASH + DISCARD). When set, "
            "no-improvement grace is disabled so the swarm always exhausts "
            "its full budget — essential for fair ablation comparisons. "
            "--deadline-hours becomes a safety net only (default 168h when "
            "--max-trials is set). Example: --max-trials 100"
        ),
    )
    p.add_argument("--deadline-hours", type=float, default=None,
                   help=(
                       "Wall-clock safety net. Default: 168h when --max-trials "
                       "is set (safety net only), 48h otherwise (primary stop)."
                   ))
    p.add_argument("--no-improvement-hours", type=float,
                   default=float(config.NO_IMPROVEMENT_GRACE_S) / 3600.0,
                   help=(
                       "Stop if no `keep` for this many hours. Ignored when "
                       "--max-trials is set (swarm runs to full budget)."
                   ))
    p.add_argument("--launch-stagger-s", type=float,
                   default=core.LAUNCH_STAGGER_S)
    p.add_argument(
        "--state-root", type=str, default=None, metavar="PATH",
        help=(
            "Override MAGENT_LOCAL_ROOT for this run (blackboard + workdirs "
            "live under PATH/blackboard and PATH/workdirs). The flag is "
            "pre-parsed before harness imports, so it wins over both shell "
            "env and the task package's setdefault. Pass a fresh path per "
            "run when running A/B ablations side-by-side. "
            "Equivalent to MAGENT_LOCAL_ROOT=PATH in the shell. "
            "NOTE: this only isolates ccbox-LOCAL paths. To run two "
            "supervisors of the same task in parallel, ALSO pass "
            "--remote-root, --remote-sync-prefix, and --sched-name-prefix."
        ),
    )
    p.add_argument(
        "--remote-root", type=str, default=None, metavar="PATH",
        help=(
            "Override MAGENT_REMOTE_ROOT — the SharedFS-side absolute workdir "
            "path that the GPU pod sees. Default '$REMOTE_ROOT'. "
            "Required when running two same-task supervisors in parallel; "
            "without an override, both supervisors sync workdirs to the "
            "same SharedFS path and overwrite each other's train_gpt.py "
            "between trials, corrupting results. Use a literal '$HOME' "
            "(single-quoted) so the pod expands it at job runtime."
        ),
    )
    p.add_argument(
        "--remote-sync-prefix", type=str, default=None, metavar="PREFIX",
        help=(
            "Override MAGENT_REMOTE_SYNC_PREFIX — the prefix used by "
            "`sched sync :PREFIX/...`. Default 'remote_dev'. Same parallel-"
            "isolation rationale as --remote-root: two supervisors with "
            "the same prefix race each other on the same SharedFS rsync "
            "target."
        ),
    )
    p.add_argument(
        "--sched-name-prefix", type=str, default=None, metavar="STR",
        help=(
            "Override the SCHED job-name prefix (normally task-adapter "
            "default: 'apg' for PG, 'cif' for CIFAR, 'nc' for NC). "
            "Sets MAGENT_SCHED_NAME_PREFIX in the env so harness.config:"
            "sched_job_name() picks it up. Job names become "
            "<PREFIX>-<dom[:4]>-NNNN; keep PREFIX <=4 chars to leave "
            "room within SCHED's 30-char limit. Required for two "
            "same-task supervisors in parallel: without unique "
            "prefixes, the supervisor's SIGINT cleanup regex cross-"
            "kills the other run's jobs."
        ),
    )
    p.add_argument(
        "--max-turns", type=int, default=None, metavar="N",
        help=(
            "Override DoerConfig.max_turns for every specialist this run "
            "(default 200). Lower values (e.g. 50) discourage in-session "
            "multi-submit. Used in the no-lineage ablation to align "
            "session shape across the A/B pair (both runs should pass "
            "the same --max-turns to keep session shape constant)."
        ),
    )
    p.add_argument(
        "--no-lineage", action="store_true",
        help=(
            "No-lineage ablation: blank LEADERBOARD/KNOWLEDGE/Recent "
            "Activity/Saturation in the per-iteration prompt, drop "
            "read_snapshot/diff_snapshots tools, and deny Bash reads of "
            "blackboard files via the block_bash_blackboard PreToolUse "
            "hook. Sets MAGENT_NO_LINEAGE=1 in the process environment so "
            "downstream readers (agents/base.py, agents/hooks.py) see it "
            "without further plumbing. The current-best exp_id and score "
            "remain visible (needed for rebase_to). Use a fresh "
            "--state-root for the ablation run so its blackboard is "
            "isolated from the lineage-on baseline."
        ),
    )
    # metavar uses the flag stem (e.g. BASELINE_BPB / BASELINE_ACCURACY)
    # so --help reads the same as before for any task that keeps PG's flag.
    _flag_metavar = adapter.baseline_score_flag.lstrip("-").replace("-", "_").upper()
    p.add_argument(adapter.baseline_score_flag, type=_finite_float, dest="baseline_score",
                   metavar=_flag_metavar,
                   help=f"Seed blackboard with this {adapter.score_field} as the "
                        f"baseline if no rows exist yet. Default is "
                        f"{adapter.baseline_score_default:.4f} — "
                        f"{adapter.bootstrap_hypothesis[:80]}…")
    p.add_argument(
        "--reset-stale-workdirs", action="store_true",
        help=(
            f"Before launch, delete any workdir_<spec>/{adapter.baseline_filename} "
            f"whose sha256 differs from the package-root baseline "
            f"({adapter.pkg_root.name}/{adapter.baseline_filename}). Re-seed "
            f"happens on each specialist's first iter via _stage_workdir. "
            f"Use this after a baseline migration when specialists should "
            f"abandon their prior-era edits. Without this flag, a hash "
            f"mismatch is REPORTED at startup but not acted on."
        ),
    )
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def _maybe_bootstrap(baseline_score: Optional[float]) -> None:
    """Seed the blackboard on first run. Idempotent.

    For tasks with `requires_calibrated_baseline=True` (NC, CIFAR), an
    empty blackboard with no explicit `--baseline-*` flag is a hard
    error: the placeholder default would pollute early-iter
    `delta_vs_best`. Operator must first run
    `python -m multi_agent_<task>.calibrate_baseline --score X.XXX`.
    """
    from ..harness import blackboard
    if blackboard.read_best() is not None:
        return
    from agent_core import current_adapter
    adapter = current_adapter()
    score_field = adapter.score_field

    if baseline_score is None and adapter.requires_calibrated_baseline:
        pkg = adapter.pkg_root.name
        raise SystemExit(
            f"refusing to cold-start with placeholder "
            f"{score_field}={adapter.baseline_score_default:.4f}.\n"
            f"  task '{pkg}' requires a calibrated baseline; either:\n"
            f"    (a) run the unedited baseline ≥1 time, then\n"
            f"        python -m {pkg}.calibrate_baseline --score X.XXXX [--score Y.YYYY ...]\n"
            f"    (b) pass {adapter.baseline_score_flag} X.XXXX to supervisor "
            f"if you have a trusted score from a prior pod and accept the "
            f"single-source-of-truth risk."
        )

    score_value = baseline_score if baseline_score is not None else adapter.baseline_score_default
    if not math.isfinite(score_value):
        raise SystemExit(
            f"refusing to bootstrap with non-finite {score_field}={score_value!r}; "
            f"check {adapter.baseline_score_flag} or "
            f"{adapter.pkg_root.name}/task_config.py:baseline_score_default"
        )
    blackboard.bootstrap_from_baseline({
        score_field:      f"{score_value:.6f}",
        "hypothesis":     adapter.bootstrap_hypothesis,
        "snapshot_path":  "",
    })


def _print_summary(s: core.RunSummary) -> None:
    print()
    print("=" * 72)
    print(f"Supervisor run complete")
    print("=" * 72)
    print(f"  started      : {s.started_iso}")
    print(f"  ended        : {s.ended_iso}")
    print(f"  elapsed      : {s.elapsed_s:.0f} s ({s.elapsed_s/3600:.2f} h)")
    print(f"  stop reason  : {s.stop_reason}")
    print(f"  specialists  : {', '.join(s.specialists)}")
    print(f"  iters/spec   :")
    for k, v in sorted(s.iters_per_spec.items()):
        print(f"    {k:<6} {v}")
    if s.final_best:
        from agent_core import current_adapter
        score_field = current_adapter().score_field
        print(f"  final best   : exp_{s.final_best.get('exp_id','?')} "
              f"{score_field}={s.final_best.get(score_field,'?')} "
              f"({s.final_best.get('specialist','?')})")
    else:
        print(f"  final best   : (none — no VALID runs)")
    print("=" * 72)


def main(argv: Optional[list[str]] = None) -> int:
    # `--state-root`, `--remote-root`, `--remote-sync-prefix`, and
    # `--sched-name-prefix` are pre-parsed BEFORE the task package is
    # imported, because each task package's __init__.py uses
    # os.environ.setdefault to install MAGENT_LOCAL_ROOT /
    # MAGENT_REMOTE_ROOT / MAGENT_REMOTE_SYNC_PREFIX defaults.
    # setdefault is a no-op if the env is already set; by writing the
    # CLI values here, we ensure the CLI wins over per-task defaults.
    # MAGENT_SCHED_NAME_PREFIX has no setdefault path; this is the only
    # injection point so it must run before harness.config imports.
    _apply_state_root_from_argv(argv)
    _apply_remote_isolation_from_argv(argv)

    # Register a task adapter BEFORE building the parser or importing
    # any harness module. Parser defaults (specialists, baseline flag) and
    # config.LOCAL_ROOT both depend on env vars the task package may
    # setdefault on import.
    _ensure_task_package_registered(argv)

    # Now safe to import harness — env defaults are final.
    from ..harness import config, credentials
    from . import core

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    specialists = [s.strip() for s in args.specialists.split(",") if s.strip()]
    if not specialists:
        print("error: --specialists cannot be empty", file=sys.stderr)
        return 2

    # `--no-lineage` flips a process-wide env var that downstream readers
    # consume (agents/base.py:_no_lineage_active, agents/hooks.py:
    # block_bash_blackboard). Setting it here (after argparse) ensures
    # every specialist coroutine spawned below sees the same value,
    # without each having to take a separate config field.
    if args.no_lineage:
        os.environ["MAGENT_NO_LINEAGE"] = "1"

    # `--max-turns` is plumbed via DoerConfig overrides (one per spec),
    # threaded into core.run as `doer_cfg_overrides`. Other DoerConfig
    # fields (model, thinking_budget, enable_web) keep their defaults —
    # __post_init__ resolves model from swarm_config.json.
    overrides: Optional[dict[str, "core.DoerConfig"]] = None
    if args.max_turns is not None:
        from ..agents.base import DoerConfig
        overrides = {
            s: DoerConfig(specialist=s, max_turns=args.max_turns)
            for s in specialists
        }

    credentials.ensure_api_key()
    config.ensure_dirs()
    _maybe_bootstrap(args.baseline_score)

    _install_shutdown_hooks()

    # Resolve deadline_hours: explicit value > mode-based default.
    # When max_trials is set, use a generous safety net (7 days) unless
    # the operator explicitly passed --deadline-hours.
    if args.deadline_hours is not None:
        deadline_hours = args.deadline_hours
    elif args.max_trials is not None:
        deadline_hours = 168.0   # 7 days — safety net only
    else:
        deadline_hours = float(config.DEADLINE_HOURS)  # 48h primary

    try:
        summary = asyncio.run(core.run(
            specialists,
            deadline_hours=deadline_hours,
            no_improvement_grace_s=args.no_improvement_hours * 3600.0,
            max_trials=args.max_trials,
            launch_stagger_s=args.launch_stagger_s,
            doer_cfg_overrides=overrides,
            reset_stale_workdirs=args.reset_stale_workdirs,
        ))
    except KeyboardInterrupt:
        # Signal handler already stopped in-flight SCHED jobs; atexit may fire
        # again but it's idempotent (empty registry = no-op).
        print("\n[shutdown] supervisor exited via signal", file=sys.stderr)
        return 130
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
