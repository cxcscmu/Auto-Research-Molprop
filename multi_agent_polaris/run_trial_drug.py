"""run_trial_drug.py — local trial runner for Drug Discovery.

Staged into specialist workdir by the harness (stage_files); NOT inside
pipeline/ so the agent cannot modify it. This file is the trust boundary:
it controls all TDC data access and enforces test-label isolation.

Flow:
  1. Load TDC ADMET BenchmarkGroup from HARNESS_TDC_DATA_DIR.
  2. For each of 22 tasks (or 1 task in SMOKE_TEST mode):
     a. Get train_val split from TDC.
     b. Scaffold-split train_val → internal train (80%) + internal val (20%).
     c. Write train/val CSVs to harness-private temp dir (NOT workdir).
     d. Write test SMILES-only CSV to harness-private temp dir (no Y col).
     e. Call: python experiment.py --task T --mode fit  --train ... --val ...
     f. Call: python experiment.py --task T --mode predict --input ... --output ...
     g. Read predictions; compute per-task internal val metric for diagnostics.
  3. Compute aggregate_score = mean normalised improvement over baseline.
  4. Write result JSONL to full_eval_results/<workdir_name>/run_seed0.jsonl.

Test leakage guard:
  - Test labels (Y) are loaded into a harness-private variable and never
    written to any file accessible from the workdir.
  - The test CSV passed to experiment.py contains only Drug and Drug_ID cols.
  - group.evaluate() (which touches test labels) is NEVER called in this loop;
    it is reserved for calibrate_baseline.py and final paper reporting only.

Aggregate score formula (higher = better):
  For classification tasks (AUROC): norm_i = (auroc_i - base_auroc_i) / base_auroc_i
  For regression tasks (MAE, lower=better): norm_i = (base_mae_i - mae_i) / base_mae_i
  aggregate_score = mean(norm_i) across tasks with valid base score.

Baseline scores are read from HARNESS_BASELINE_SCORES (JSON file written by
calibrate_baseline.py). If missing, aggregate_score = mean raw val metric
(bootstrapping mode used by calibrate_baseline itself).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Suppress noisy RDKit C++ warnings (e.g. "not removing hydrogen atom without
# neighbors") that clutter progress output. These are benign parsing artefacts.
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

# ── Project-root bootstrap (benchmark-data provider import) ────────────────────
# This file is staged into WORKDIR and run from there (run_trial.sh does
# `cd "$WORKDIR"`), so the project root is NOT on sys.path. HARNESS_PKG_ROOT
# (injected by task_config.pod_env_for_trial) points at the real benchmark package
# source dir; its parent is the project root holding agent_core/ + the benchmark
# packages. Prepend it so we can import the BenchmarkDataProvider.
_PKG_ROOT_ENV = os.environ.get("HARNESS_PKG_ROOT", "")
if _PKG_ROOT_ENV:
    _proj_root = str(Path(_PKG_ROOT_ENV).resolve().parent)
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)


# ── Constants ────────────────────────────────────────────────────────────────

SMOKE_TEST    = os.environ.get("SMOKE_TEST", "0") == "1"
SINGLE_TASK   = os.environ.get("HARNESS_SINGLE_TASK", "")   # run one task only
TDC_DATA_DIR  = os.environ.get("HARNESS_TDC_DATA_DIR",
                               os.path.expanduser("~/drug_dev/tdc_data"))
BASELINE_FILE = os.environ.get("HARNESS_BASELINE_SCORES", "")
WORKDIR       = Path(os.environ.get("HARNESS_WORKDIR",
                     os.environ.get("WORKDIR", "."))).resolve()
OUT_DIR       = WORKDIR / "full_eval_results" / WORKDIR.name
WALL_LIMIT_S  = int(os.environ.get("HARNESS_WALL_LIMIT_S", "3600"))  # fallback matches run_trial.sh

# ── Ablation mode ─────────────────────────────────────────────────────────────
# HARNESS_ABLATION_MODE controls which pipeline files are frozen during eval.
#   "joint"        (default) — no files frozen, full search space
#   "feature_only" — models.py + calibration.py frozen at PKG_ROOT baseline
#   "model_only"   — features.py frozen at PKG_ROOT baseline
#
# Enforcement is two-layered per subprocess call:
#   1. FREEZE  — overwrite frozen files with PKG_ROOT baseline before subprocess
#   2. VERIFY  — hash-check frozen files after subprocess; abort task if tampered
ABLATION_MODE = os.environ.get("HARNESS_ABLATION_MODE", "feature_only").strip().lower()
# HARNESS_PKG_ROOT must be set by pod_env_for_trial (task_config.py) to the
# real multi_agent_drug/ source directory.  run_trial.sh does `cd "$WORKDIR"`
# before exec-ing this script, so Path(__file__) would resolve to WORKDIR —
# NOT the source tree — making the fallback wrong for ablation freeze.
# The fallback is kept only for calibrate_baseline (ABLATION_MODE=joint,
# freeze never runs) and local smoke tests where freeze is also joint.
PKG_ROOT = Path(os.environ.get("HARNESS_PKG_ROOT",
                               str(Path(__file__).resolve().parent)))

_VALID_ABLATION_MODES = {"joint", "feature_only", "model_only", "data_only"}
if ABLATION_MODE not in _VALID_ABLATION_MODES:
    raise ValueError(
        f"HARNESS_ABLATION_MODE={ABLATION_MODE!r} is invalid. "
        f"Must be one of: {_VALID_ABLATION_MODES}"
    )

# Files frozen (always restored to PKG_ROOT baseline) per ablation mode.
_ABLATION_FROZEN: dict[str, list[str]] = {
    "joint":        [],
    "feature_only": [
        "pipeline/models.py",
        "pipeline/calibration.py",
        "pipeline/pipeline.py",
        "experiment.py",
    ],
    "model_only": [
        "pipeline/features.py",
        "pipeline/pipeline.py",
        "experiment.py",
    ],
    # data_only: everything frozen; only external_data/{task}.csv can vary.
    "data_only": [
        "pipeline/features.py",
        "pipeline/models.py",
        "pipeline/calibration.py",
        "pipeline/pipeline.py",
        "experiment.py",
    ],
}

# External data limits (data_only mode)
MAX_EXTERNAL_ROWS_PER_TASK = 5_000   # rows per task per trial
MAX_EXTERNAL_CSV_BYTES     = 2 * 1024 * 1024   # 2 MB

# Leakage-safe augmentation thresholds (data_only mode). See _merge_external_data.
# L2 same-source rejection: if more than this fraction of the TDC *test* set for a
# task is present in the raw external source (by InChIKey skeleton), the source
# overlaps the benchmark's own origin → reject the whole file (not just dedup it).
# 5% tolerates a few incidental common drugs while killing same-source databases
# (observed exploit overlap rates: half_life 20%, vdss 22%, ames 64%).
SAME_SOURCE_TEST_OVERLAP = 0.05
# L3 analog filter: drop external molecules whose ECFP4 Tanimoto to ANY test
# molecule is >= this — close analogs carry test-set information.
ANALOG_TANIMOTO          = 0.90

# Scaffold split ratio for internal train/val from TDC's train_val.
VAL_SPLIT_FRAC = 0.20
SPLIT_SEED     = 42


# ── TDC helpers ──────────────────────────────────────────────────────────────

def _load_data_provider():
    """Construct the benchmark-data provider selected by MAGENT_TASK.

    The staged runner is a standalone subprocess (no in-memory adapter), so it builds
    its own provider. Defaults to 'drug' so any legacy call without the env still runs
    TDC. Falls back to a direct file-path import (bypassing the package __init__) if the
    normal package import fails.
    """
    task = os.environ.get("MAGENT_TASK", "drug").strip().lower()
    _MODMAP = {
        "drug":    ("multi_agent_drug.benchmark_data",    "TdcAdmetDataProvider"),
        "molnet":  ("multi_agent_molnet.benchmark_data",  "MolNetDataProvider"),
        "polaris": ("multi_agent_polaris.benchmark_data", "PolarisDataProvider"),
    }
    if task not in _MODMAP:
        raise SystemExit(f"run_trial_drug: unknown MAGENT_TASK={task!r} (known: {sorted(_MODMAP)})")
    mod_name, cls_name = _MODMAP[task]
    try:
        import importlib
        mod = importlib.import_module(mod_name)
    except Exception:
        import importlib.util
        if not _PKG_ROOT_ENV:
            raise SystemExit("run_trial_drug: HARNESS_PKG_ROOT unset; cannot locate benchmark_data")
        bd_path = Path(_PKG_ROOT_ENV) / "benchmark_data.py"
        spec = importlib.util.spec_from_file_location(mod_name, bd_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return getattr(mod, cls_name)()


# Benchmark-data provider (group loader / metric / task type / isolation knobs).
# Selected by MAGENT_TASK; defaults to TDC. Construction is cheap (no data load).
PROVIDER = _load_data_provider()


def _scaffold_split(df, val_frac: float = VAL_SPLIT_FRAC, seed: int = SPLIT_SEED):
    """Random scaffold split of a DataFrame. Returns (train_df, val_df).

    Grouping/shuffle/greedy-fill core lives in agent_core.harness.splits (shared with
    the MolNet loader's outer split). large=train, small=val — byte-identical to the
    former inline implementation for the same (df, val_frac, seed).
    """
    from agent_core.harness.splits import scaffold_partition_indices
    train_idx, val_idx = scaffold_partition_indices(df["Drug"], val_frac, seed)
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


# Metric computation, metric direction, per-task metric lookup, and baseline
# normalisation now live in the benchmark-data provider (PROVIDER, defined above;
# see agent_core.benchmark_data) so the runner stays benchmark-agnostic. TDC values
# are unchanged: TdcAdmetDataProvider inherits the verbatim implementations.


# ── Trial runner ─────────────────────────────────────────────────────────────

def _probe_unshare(tdc_data_dir: str, empty_dir: str) -> bool:
    """Return True if unshare mount-namespace isolation is available on this host."""
    if not tdc_data_dir or not os.path.isdir(tdc_data_dir):
        return False
    try:
        shell_cmd = (
            f"mount --bind {shlex.quote(empty_dir)} {shlex.quote(tdc_data_dir)} && "
            f"ls {shlex.quote(tdc_data_dir)} | wc -l"
        )
        r = subprocess.run(
            ["unshare", "--mount", "--user", "--map-root-user", "bash", "-c", shell_cmd],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0 and r.stdout.strip() == "0"
    except Exception:
        return False


def _run_subprocess(
    cmd: list[str],
    timeout_s: float,
    tdc_block_dir: Path | None = None,
    tdc_data_dir: str | None = None,
    empty_dir: str | None = None,
) -> tuple[int, str]:
    """Run agent subprocess with multiple isolation layers.

    Layer 1 (filesystem, strongest): unshare --mount --user --map-root-user
      bind-mounts an empty directory over TDC_DATA_DIR inside a private
      mount namespace. The agent subprocess sees an empty directory in place
      of the TDC cache — test.csv files are invisible at the OS level.
      Active when tdc_data_dir + empty_dir are provided and unshare is usable.

    Layer 2 (import): tdc_block_dir prepended to PYTHONPATH so `import tdc`
      raises ImportError in the agent subprocess (kept as defence-in-depth).

    Layer 3 (env): harness-only env vars stripped (HARNESS_TDC_DATA_DIR etc.)
      so agent can't trivially discover the TDC data path.
    """
    try:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Layer 3: strip ALL HARNESS_* env vars so the agent subprocess
        # cannot discover TDC data paths, ablation mode, pkg root, or any
        # other harness-side configuration.
        for key in list(env):
            if key.startswith("HARNESS_"):
                del env[key]
        # Layer 2: PYTHONPATH stub.
        if tdc_block_dir is not None:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (str(tdc_block_dir) + os.pathsep + existing
                                 if existing else str(tdc_block_dir))

        # Layer 1: mount namespace to hide TDC data directory.
        if tdc_data_dir and empty_dir and os.path.isdir(tdc_data_dir):
            inner = " ".join(shlex.quote(str(c)) for c in cmd)
            shell_cmd = (
                f"mount --bind {shlex.quote(empty_dir)} {shlex.quote(tdc_data_dir)} && "
                f"exec {inner}"
            )
            actual_cmd = [
                "unshare", "--mount", "--user", "--map-root-user",
                "bash", "-c", shell_cmd,
            ]
        else:
            actual_cmd = list(cmd)

        result = subprocess.run(
            actual_cmd, capture_output=True, text=True, timeout=timeout_s, env=env,
        )
        return result.returncode, (result.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as exc:
        return 1, str(exc)


def _ablation_freeze(workdir: Path) -> dict[str, str]:
    """Overwrite frozen pipeline files with PKG_ROOT baseline and return hashes.

    Called immediately BEFORE every agent subprocess (fit and predict) so that
    even if the agent edited a frozen file in a previous iteration, the eval
    always uses the original baseline version.

    Returns {relative_path: md5_hex} for post-run integrity verification.
    Raises RuntimeError if a baseline source file is missing (config error).
    """
    import hashlib
    import shutil

    frozen = _ABLATION_FROZEN.get(ABLATION_MODE, [])
    hashes: dict[str, str] = {}
    for rel in frozen:
        src = PKG_ROOT / rel
        dst = workdir / rel
        if not src.is_file():
            raise RuntimeError(
                f"[ablation:{ABLATION_MODE}] baseline source missing: {src}\n"
                f"Cannot enforce {ABLATION_MODE} isolation without this file."
            )
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[rel] = hashlib.md5(dst.read_bytes()).hexdigest()
    return hashes


def _ablation_verify(workdir: Path, expected: dict[str, str]) -> list[str]:
    """Verify frozen files were not modified by the subprocess.

    Called immediately AFTER every agent subprocess. Returns list of files
    whose content changed (should always be empty for a clean run).
    A non-empty list means the pipeline code attempted to overwrite a frozen
    file during execution — this trial is marked as tampered and discarded.
    """
    import hashlib

    tampered: list[str] = []
    for rel, expected_hash in expected.items():
        dst = workdir / rel
        if dst.is_file():
            actual = hashlib.md5(dst.read_bytes()).hexdigest()
            if actual != expected_hash:
                tampered.append(rel)
        else:
            tampered.append(f"{rel}(deleted)")
    return tampered


# ── Task-level result cache (feature_only mode) ──────────────────────────────
#
# When ABLATION_MODE == "feature_only", only pipeline/features.py can change.
# If the agent only modified task-conditional blocks in get_task_features(),
# tasks whose blocks are unchanged don't need to re-run.
#
# Security design:
#   - Cache stored in harness-private blackboard/task_cache/ (NOT in WORKDIR).
#     Agents cannot write outside their WORKDIR, so they cannot tamper with
#     cached val_metric values.
#   - global_hash = SHA256 of features.py with recognized task blocks blanked.
#     Any change to featurize(), imports, helpers, or constants invalidates
#     ALL cached tasks.
#   - Parser only recognises strict `if task_name in ('t1', 't2'):` with string
#     literal tuples.  Any other pattern (==, variable, set, list, multi-line)
#     causes _UNRECOGNIZED sentinel → full re-run (truly conservative).
#   - Cache stores val_metric (not norm_improvement); norm is recomputed from
#     current baseline_scores.json on every load, so recalibration is safe.
#   - baseline_hash in cache key: if baseline_scores.json changes, cache stales.

import hashlib as _hashlib
import json as _json_cache
import re as _re_cache

_CACHE_VERSION  = 2
_UNRECOGNIZED   = "_unrecognized_"
_MISSING        = "_missing_"
_VALID_TASK_RE  = _re_cache.compile(r'^[a-z][a-z0-9_]+$')

# Strict pattern: only `if task_name in ('t1', 't2'):` with round-bracket tuple
_STRICT_BLOCK_RE = _re_cache.compile(
    r"if\s+task_name\s+in\s+\(([^)]+)\)\s*:", _re_cache.MULTILINE
)
# Any other `task_name` condition pattern that we do NOT recognise
_ANY_TASK_COND_RE = _re_cache.compile(
    r"\btask_name\b\s*(==|!=|in\s+\[|in\s+\{|in\s+[a-zA-Z_])", _re_cache.MULTILINE
)


def _cache_path(workdir: Path) -> Path:
    """Return harness-private cache path (outside agent-editable WORKDIR).

    WORKDIR = .../drug_dev/workdirs/workdir_fphs
    Cache   → .../drug_dev/blackboard/task_cache/workdir_fphs.json
    """
    local_root = workdir.parent.parent   # drug_dev/
    return local_root / "blackboard" / "task_cache" / f"{workdir.name}.json"


def _extract_func_body(source: str, func_name: str) -> str:
    """Return source lines of a top-level function.

    Returns _MISSING sentinel (not empty string) when function is absent,
    so callers can distinguish 'function not found' from 'empty function'.
    """
    lines   = source.splitlines()
    in_func = False
    body: list[str] = []
    for line in lines:
        if not in_func:
            if f"def {func_name}(" in line:
                in_func = True
                body.append(line)
        else:
            if (line and not line[0].isspace()
                    and not line.startswith('#')
                    and line.strip()
                    and (line.strip().startswith('def ')
                         or line.strip().startswith('class '))):
                break
            body.append(line)
    return '\n'.join(body) if body else _MISSING


def _compute_feature_hashes(workdir: Path) -> tuple[str, dict[str, str]]:
    """Return (global_hash, {task_name: block_hash}) from workdir's features.py.

    global_hash
        SHA256 of features.py with ALL recognized task-conditional blocks blanked
        out.  Covers featurize(), imports, helpers, constants, and the skeleton of
        get_task_features().  ANY unrecognized change here → full re-run.

    task_hashes
        Per-task SHA256 of each strictly-recognized block.  Only used when
        global_hash is unchanged.

    Returns ('_error_', {}) on read/parse error.
    Returns ('_unrecognized_', {}) if any unrecognized task_name condition is found
        in get_task_features() → caller must re-run all tasks.
    """
    features_path = workdir / "pipeline" / "features.py"
    if not features_path.is_file():
        return "_error_", {}
    try:
        src = features_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "_error_", {}

    # ── 1. Extract get_task_features() body ──────────────────────────────────
    task_func_src = _extract_func_body(src, "get_task_features")
    if task_func_src == _MISSING:
        return "_error_", {}

    # ── 2. Parse ONLY strict `if task_name in ('t1', 't2'):` blocks ──────────
    #    First find which lines contain unrecognized task_name conditions.
    task_func_lines = task_func_src.splitlines()
    recognized_spans: list[tuple[int, int, list[str], str]] = []  # (i_start, i_end, names, block_src)

    i = 0
    while i < len(task_func_lines):
        line = task_func_lines[i]
        stripped = line.lstrip()

        # Skip blank and comment lines
        if not stripped or stripped.startswith('#'):
            i += 1
            continue

        # Check for any task_name condition on this line
        any_cond = _ANY_TASK_COND_RE.search(line)
        strict_m = _STRICT_BLOCK_RE.search(line)

        if any_cond and not strict_m:
            # Unrecognized pattern (==, list, set, variable) → bail out entirely
            return _UNRECOGNIZED, {}

        if strict_m:
            # Recognized strict block: collect its lines
            raw_names = _re_cache.findall(r"['\"]([^'\"]+)['\"]", strict_m.group(1))
            names = [n for n in raw_names if _VALID_TASK_RE.match(n)]
            if not names:
                i += 1
                continue
            block_lines = [line]
            base_indent = len(line) - len(stripped)
            j = i + 1
            while j < len(task_func_lines):
                bl = task_func_lines[j]
                if bl.strip() == '':
                    block_lines.append(bl); j += 1; continue
                cur_indent = len(bl) - len(bl.lstrip())
                if cur_indent > base_indent:
                    block_lines.append(bl); j += 1
                else:
                    break
            block_src = '\n'.join(block_lines)
            recognized_spans.append((i, j, names, block_src))
            i = j
            continue

        i += 1

    # ── 3. global_hash: features.py with recognized blocks blanked ────────────
    #    We blank the blocks inside get_task_features(); the rest of features.py
    #    (featurize, descriptors, fingerprints, imports, helpers) is hashed as-is.
    src_for_global = src
    # Locate the task blocks in the full source and replace with blank lines
    # (simple approach: replace the block text with equivalent number of blank lines)
    for _, _, _, block_src in recognized_spans:
        # Replace first occurrence of block_src in src_for_global
        n_lines = block_src.count('\n') + 1
        src_for_global = src_for_global.replace(block_src, '\n' * (n_lines - 1), 1)

    global_hash = _hashlib.sha256(src_for_global.encode()).hexdigest()[:16]

    # ── 4. Per-task hashes ────────────────────────────────────────────────────
    task_hashes: dict[str, str] = {}
    for _, _, names, block_src in recognized_spans:
        block_hash = _hashlib.sha256(block_src.encode()).hexdigest()[:12]
        for name in names:
            task_hashes[name] = block_hash

    return global_hash, task_hashes


def _baseline_hash() -> str:
    """SHA256 of current baseline_scores.json (first 12 chars)."""
    if BASELINE_FILE and Path(BASELINE_FILE).is_file():
        return _hashlib.sha256(Path(BASELINE_FILE).read_bytes()).hexdigest()[:12]
    return "_no_baseline_"


def _load_task_cache(workdir: Path) -> dict | None:
    """Load harness-private task cache. Returns None if missing/invalid."""
    path = _cache_path(workdir)
    if not path.is_file():
        return None
    try:
        data = _json_cache.loads(path.read_text())
        if data.get("version") != _CACHE_VERSION:
            return None
        # Defence-in-depth: never reuse a cache written under a different
        # ablation mode (path isolation already separates state roots, but
        # don't rely on it alone).
        if data.get("ablation_mode") != ABLATION_MODE:
            return None
        return data
    except Exception:
        return None


def _save_task_cache(workdir: Path, global_hash: str,
                     task_hashes: dict[str, str],
                     task_metrics: dict[str, dict]) -> None:
    """Persist cache to harness-private location (not inside WORKDIR)."""
    path = _cache_path(workdir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json_cache.dumps({
            "version":        _CACHE_VERSION,
            "ablation_mode":  ABLATION_MODE,
            "global_hash":    global_hash,
            "baseline_hash":  _baseline_hash(),
            "task_hashes":    task_hashes,
            # Store val_metric (not norm_improvement) so norm can be recomputed
            # if baseline_scores.json is recalibrated.
            "task_metrics":   task_metrics,
        }, indent=2))
    except Exception:
        pass   # cache write failure is non-fatal


def _resolve_tasks_to_run(
    all_tasks: list[str],
    workdir: Path,
    baseline_scores: dict,
) -> tuple[list[str], dict[str, dict]]:
    """Decide which tasks need fresh evaluation vs can reuse cached results.

    Returns (tasks_to_run, cached_results).
    Conservative: any ambiguity → tasks_to_run == all_tasks, cached_results == {}.
    Only active in feature_only mode.
    """
    if ABLATION_MODE != "feature_only":
        return list(all_tasks), {}

    global_hash, task_hashes = _compute_feature_hashes(workdir)

    if global_hash in ("_error_", _UNRECOGNIZED):
        reason = "parse error" if global_hash == "_error_" else "unrecognized task_name pattern"
        print(f"[cache] {reason} → re-run all 22 tasks", flush=True)
        return list(all_tasks), {}

    cache = _load_task_cache(workdir)
    if cache is None:
        return list(all_tasks), {}

    # Invalidate if global code changed
    if cache.get("global_hash") != global_hash:
        print("[cache] global features changed → re-run all 22 tasks", flush=True)
        return list(all_tasks), {}

    # Invalidate if baseline_scores.json was recalibrated
    if cache.get("baseline_hash") != _baseline_hash():
        print("[cache] baseline_scores.json changed → re-run all 22 tasks", flush=True)
        return list(all_tasks), {}

    # Per-task: compare block hashes, recompute norm from current baseline
    cached_results: dict[str, dict] = {}
    tasks_to_run:   list[str] = []

    for task in all_tasks:
        cur_hash  = task_hashes.get(task, "_empty_")
        prev_hash = cache.get("task_hashes", {}).get(task, "_empty_")
        cached_m  = cache.get("task_metrics", {}).get(task)

        if (cur_hash == prev_hash
                and cached_m is not None
                and cached_m.get("status") == "ok"
                and cached_m.get("val_metric") is not None):
            # Recompute norm from current baseline (safe against recalibration)
            val_metric = cached_m["val_metric"]
            base       = baseline_scores.get(task, {}).get("metric")
            norm       = PROVIDER.normalise(val_metric, base, PROVIDER.task_metric(task)) if base else None
            cached_results[task] = {
                **cached_m,
                "norm_improvement": norm,
                "status": "ok",
            }
        else:
            tasks_to_run.append(task)

    n_cached = len(cached_results)
    if n_cached:
        print(
            f"[cache] {n_cached} tasks reused from cache, "
            f"{len(tasks_to_run)} tasks to re-run: "
            f"{tasks_to_run}",
            flush=True,
        )
    return tasks_to_run, cached_results


# ── data_only external data support ──────────────────────────────────────────

_STD_TOOLS: dict = {}


def _get_std_tools():
    """Lazily build (and cache) RDKit standardizers: desalt + neutralize."""
    if not _STD_TOOLS:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        _STD_TOOLS["lfc"]  = rdMolStandardize.LargestFragmentChooser()
        _STD_TOOLS["unch"] = rdMolStandardize.Uncharger()
    return _STD_TOOLS


def _std_keys(mol):
    """Standardized identity keys for leakage-safe dedup.

    Pipeline: keep largest fragment (strip salts/solvents) → neutralize charges
    → standard InChIKey. Returns (full_inchikey, skeleton). The skeleton is the
    first InChIKey block (connectivity layer) — invariant to stereo / charge /
    isotope / salt, so it matches "the same compound" across representations.
    (None, None) on failure → caller treats the molecule as unverifiable.
    """
    try:
        from rdkit.Chem.inchi import MolToInchiKey
        t = _get_std_tools()
        m = t["unch"].uncharge(t["lfc"].choose(mol))
        ik = MolToInchiKey(m)
        if not ik:
            return None, None
        return ik, ik.split("-")[0]
    except Exception:
        return None, None


def _ecfp(mol):
    """ECFP4 (Morgan r=2, 2048-bit) for analog (Tanimoto) filtering."""
    from rdkit.Chem import AllChem
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def _build_test_dedup_index(group, tasks) -> dict:
    """Per-task index of TDC test molecules for leakage-safe augmentation.

    Returns {task: {'ikeys': set, 'skeletons': set, 'fps': [fp], 'n': int}}.
      ikeys     — standardized full InChIKeys (L1 identity dedup)
      skeletons — InChIKey skeletons (L2 same-source overlap rate)
      fps       — ECFP4 fingerprints (L3 analog filter)
    Built once per trial, only for tasks the agent actually augmented.
    """
    from rdkit import Chem

    result: dict = {}
    for task in tasks:
        ikeys: set = set(); skels: set = set(); fps: list = []
        try:
            test_df = group.get(task)["test"]
            for smi in test_df["Drug"]:
                mol = Chem.MolFromSmiles(str(smi))
                if not mol:
                    continue
                ik, sk = _std_keys(mol)
                if ik:
                    ikeys.add(ik); skels.add(sk)
                try:
                    fps.append(_ecfp(mol))
                except Exception:
                    pass
        except Exception:
            pass
        result[task] = {"ikeys": ikeys, "skeletons": skels, "fps": fps, "n": len(fps)}
    return result


def _merge_external_data(
    train_df,
    val_df,
    task_name: str,
    workdir: Path,
    test_index: dict,
    task_type: str,
) -> tuple[object, dict]:
    """Load external_data/{task_name}.csv → leakage-safe filter → merge into train.

    Three harness-side layers (the agent cannot bypass them):
      L1  identity dedup     — drop external molecules whose STANDARDIZED InChIKey
                               (desalt + neutralize, so salt/charge variants match)
                               equals any test / val / train molecule.
      L2  same-source reject — if > SAME_SOURCE_TEST_OVERLAP of the TDC *test* set
                               (by InChIKey skeleton) appears in the raw external
                               source, the source overlaps the benchmark's own
                               origin → REJECT the whole file (not just dedup it).
                               This is what stops an agent re-importing the
                               benchmark's source database (e.g. Obach/Lombardo via
                               PKSmart, Hansen via an "ames" copy).
      L3  analog filter      — drop survivors with ECFP4 Tanimoto >= ANALOG_TANIMOTO
                               to any test molecule (close analogs leak too).

    audit['verdict'] ∈ {no_external, rejected_same_source, accepted_empty,
    accepted, error} and audit['agent_note'] explain the outcome so the agent
    (reading per_task[task]['data_aug']) learns NOT to resubmit a rejected source.

    Returns (augmented_train_df, audit). On error: (original train_df, audit).
    """
    import pandas as pd
    from rdkit import Chem

    ext_path = workdir / "external_data" / f"{task_name}.csv"
    audit: dict = {"task": task_name, "source": "",
                   "verdict": "no_external", "agent_note": ""}

    if not ext_path.is_file():
        return train_df, audit

    try:
        raw_text = ext_path.read_text(encoding="utf-8", errors="replace")

        # Optional #source: comment on first line
        lines = raw_text.splitlines()
        if lines and lines[0].startswith("#source:"):
            audit["source"] = lines[0][8:].strip()
            csv_text = "\n".join(lines[1:])
        else:
            csv_text = raw_text

        import io
        ext_df = pd.read_csv(io.StringIO(csv_text))

        # Format validation
        if "Drug" not in ext_df.columns or "Y" not in ext_df.columns:
            audit["verdict"] = "error"
            audit["error"] = "missing Drug or Y column"
            audit["agent_note"] = "CSV must have 'Drug' and 'Y' columns."
            return train_df, audit

        audit["external_rows_raw"] = len(ext_df)

        # Size cap
        if len(ext_df) > MAX_EXTERNAL_ROWS_PER_TASK:
            ext_df = ext_df.sample(MAX_EXTERNAL_ROWS_PER_TASK, random_state=42)
            audit["truncated_to"] = MAX_EXTERNAL_ROWS_PER_TASK

        # Y validation
        ext_df["Y"] = pd.to_numeric(ext_df["Y"], errors="coerce")
        ext_df = ext_df.dropna(subset=["Y"])
        if task_type == "classification":
            ext_df = ext_df[ext_df["Y"].isin([0, 1, 0.0, 1.0])]
        else:
            q01, q99 = ext_df["Y"].quantile(0.01), ext_df["Y"].quantile(0.99)
            ext_df = ext_df[(ext_df["Y"] >= q01) & (ext_df["Y"] <= q99)]

        audit["valid_smiles_before_dedup"] = len(ext_df)

        test_ikeys = test_index.get("ikeys", set())
        test_skels = test_index.get("skeletons", set())
        test_fps   = test_index.get("fps", [])
        test_n     = test_index.get("n", 0)

        # Standardized identity keys for val / train (for L1 dedup).
        def _keyset(df) -> set:
            ks: set = set()
            for smi in df["Drug"]:
                mol = Chem.MolFromSmiles(str(smi))
                if mol:
                    ik, _ = _std_keys(mol)
                    if ik:
                        ks.add(ik)
            return ks

        val_keys   = _keyset(val_df)
        train_keys = _keyset(train_df)

        # Parse external molecules once → (row, mol, std_ikey, skeleton).
        ext_mols = []
        invalid_smiles = 0
        for _, row in ext_df.iterrows():
            mol = Chem.MolFromSmiles(str(row["Drug"]))
            if mol is None:
                invalid_smiles += 1
                continue
            ik, sk = _std_keys(mol)
            ext_mols.append((row, mol, ik, sk))
        audit["invalid_smiles"] = invalid_smiles

        # ── L2: same-source rejection (raw external skeletons vs test) ────────
        if test_n > 0:
            ext_skels = {sk for _, _, _, sk in ext_mols if sk}
            overlap   = len(ext_skels & test_skels)
            rate      = overlap / test_n
            audit["test_skeleton_overlap"] = overlap
            audit["test_overlap_rate"]     = round(rate, 4)
            if rate > SAME_SOURCE_TEST_OVERLAP:
                audit["verdict"]    = "rejected_same_source"
                audit["merged_rows"] = 0
                # A source judged to be the benchmark's own/sibling data must not
                # be carried into later trials: remove the file so it is neither
                # re-merged nor snapshotted into the lineage. This makes the
                # rejection a one-time event (no repeated re-rejection of the
                # same file every trial) and frees the agent to try another source.
                removed = False
                try:
                    ext_path.unlink()
                    removed = True
                except Exception:
                    pass
                audit["removed_file"] = removed
                audit["agent_note"] = (
                    f"REJECTED — same-source leakage: {rate:.0%} of the TDC test "
                    f"molecules for '{task_name}' are present in this external "
                    f"source (InChIKey-skeleton match). It is the benchmark's own "
                    f"or a sibling source. The file has been REMOVED and will not "
                    f"be carried to the next trial. Do NOT resubmit this dataset or "
                    f"another copy of it; find a GENUINELY INDEPENDENT assay "
                    f"(different lab/database, not the original {task_name} source).")
                print(f"  [data_aug] {task_name}: REJECTED same-source "
                      f"(test overlap {rate:.0%}, removed={removed}, "
                      f"src='{audit['source'][:50]}')", flush=True)
                return train_df, audit

        # ── L1: standardized identity dedup vs test / val / train ─────────────
        dropped_test = dropped_val = dropped_train_dup = unverifiable = 0
        survivors = []   # (row, mol)
        for row, mol, ik, _ in ext_mols:
            if ik is None:
                unverifiable += 1            # cannot standardize → drop (safety)
                continue
            if ik in test_ikeys:
                dropped_test += 1
                continue
            if ik in val_keys:
                dropped_val += 1
                continue
            if ik in train_keys:
                dropped_train_dup += 1
                continue
            survivors.append((row, mol))

        # ── L3: analog filter (ECFP4 Tanimoto vs test) ────────────────────────
        dropped_analog = 0
        safe_rows = []
        if test_fps:
            from rdkit import DataStructs
            for row, mol in survivors:
                try:
                    sims = DataStructs.BulkTanimotoSimilarity(_ecfp(mol), test_fps)
                except Exception:
                    sims = []
                if sims and max(sims) >= ANALOG_TANIMOTO:
                    dropped_analog += 1
                    continue
                safe_rows.append(row)
        else:
            safe_rows = [row for row, _ in survivors]

        audit.update({
            "dropped_test_exact":   dropped_test,
            "dropped_val_exact":    dropped_val,
            "dropped_train_dup":    dropped_train_dup,
            "dropped_analog":       dropped_analog,
            "unverifiable_dropped": unverifiable,
            "merged_rows":          len(safe_rows),
        })

        if not safe_rows:
            audit["verdict"]    = "accepted_empty"
            audit["agent_note"] = (
                f"No net rows added: everything was a duplicate or close analog "
                f"of existing data (test {dropped_test} / val {dropped_val} / "
                f"train {dropped_train_dup} / analog {dropped_analog}). This source "
                f"largely overlaps known data — try a more independent dataset.")
            return train_df, audit

        safe_df = pd.DataFrame(safe_rows)[["Drug", "Y"]].reset_index(drop=True)
        merged  = pd.concat([train_df, safe_df], ignore_index=True)
        audit["verdict"]    = "accepted"
        audit["agent_note"] = (
            f"Accepted +{len(safe_rows)} rows after leakage filtering "
            f"(dropped test {dropped_test} / val {dropped_val} / train "
            f"{dropped_train_dup} / analog {dropped_analog}).")
        print(
            f"  [data_aug] {task_name}: +{len(safe_rows)} rows "
            f"(train {len(train_df)}→{len(merged)}, dropped test={dropped_test} "
            f"val={dropped_val} dup={dropped_train_dup} analog={dropped_analog})",
            flush=True,
        )
        return merged, audit

    except Exception as exc:
        audit["verdict"] = "error"
        audit["error"] = str(exc)[:300]
        audit["agent_note"] = f"External data processing error: {str(exc)[:150]}"
        return train_df, audit


def _scan_for_blocked_import(workdir: Path, modules: tuple[str, ...]) -> str | None:
    """Return first offending line if a blocked benchmark import is found in any
    agent-editable file.

    Scans experiment.py (root seed) + pipeline/**/*.py (editable tree) for
    `import <m>` / `from <m>` over `modules` (('tdc',) for TDC; ('tdc','deepchem')
    for MolNet). Dynamic imports (exec/importlib) are not caught here; the runtime
    PYTHONPATH blocker (_make_import_block_dir) provides defence-in-depth.
    """
    scan_targets: list[Path] = []
    root_exp = workdir / "experiment.py"
    if root_exp.is_file():
        scan_targets.append(root_exp)
    pipeline_dir = workdir / "pipeline"
    if pipeline_dir.is_dir():
        scan_targets.extend(sorted(pipeline_dir.rglob("*.py")))

    for py_file in scan_targets:
        try:
            lines = py_file.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for m in modules:
                if f"import {m}" in stripped or f"from {m}" in stripped:
                    return f"{py_file.name}:{i}: {stripped}"
    return None


def _make_import_block_dir(tmpdir: Path, modules: tuple[str, ...]) -> Path:
    """Create blocking stub packages that raise ImportError when imported.

    Place this at the front of PYTHONPATH in agent subprocesses so that any attempt
    to import a blocked benchmark module (e.g. tdc, deepchem) — including dynamic
    imports — hits the stub instead of the real installation.
    """
    block_root = tmpdir / "_import_block"
    for m in modules:
        pkg = block_root / m
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text(
            f'raise ImportError("{m} import blocked in agent subprocess")\n'
        )
    return block_root


def run_trial() -> dict:
    """Run full ADMET trial. Returns result dict for JSONL."""
    import pandas as pd
    import numpy as np

    t0 = time.monotonic()

    # ── Ablation mode announcement ────────────────────────────────────────────
    frozen_files = _ABLATION_FROZEN.get(ABLATION_MODE, [])
    if ABLATION_MODE != "joint":
        print(
            f"[ablation:{ABLATION_MODE}] ACTIVE — "
            f"frozen files (restored from PKG_ROOT before every subprocess): "
            f"{frozen_files}",
            flush=True,
        )
    else:
        print(f"[ablation:joint] no file restrictions", flush=True)

    # ── Static scan: reject pipeline code that imports tdc directly ───────────
    leakage_hit = _scan_for_blocked_import(WORKDIR, PROVIDER.isolation_block_modules())
    if leakage_hit:
        return {
            "status":          "CRASH",
            "aggregate_score": None,
            "n_tasks_ok":      0,
            "elapsed_s":       time.monotonic() - t0,
            "ablation_mode":   ABLATION_MODE,
            "kill_reason":     f"leakage_blocked: tdc import in agent code: {leakage_hit}",
            "per_task":        {},
        }

    # Harness uses HARNESS_PYTHON (full venv with PyTDC).
    # Agent subprocesses use AGENT_PYTHON (stripped venv, NO PyTDC).
    # Hard isolation: TDC literally not installed in agent venv.
    agent_python  = os.environ.get("AGENT_PYTHON") or sys.executable
    experiment_py = str(WORKDIR / "experiment.py")

    group     = PROVIDER.load_group()
    if SINGLE_TASK:
        all_tasks = [SINGLE_TASK]
    elif SMOKE_TEST:
        all_tasks = [list(group.dataset_names)[0]]
    else:
        all_tasks = list(group.dataset_names)

    baseline_scores: dict = {}
    if BASELINE_FILE and Path(BASELINE_FILE).is_file():
        with open(BASELINE_FILE) as f:
            baseline_scores = json.load(f)

    per_task: dict[str, dict] = {}
    # Pre-initialise every task to 0.0 improvement so failures and skips
    # automatically count as "same as baseline" without any append in each
    # failure branch. Successes overwrite their entry at the end of the loop.
    task_norm: dict[str, float] = {t: 0.0 for t in all_tasks}

    with tempfile.TemporaryDirectory(prefix="drug_trial_") as tmpdir:
        tmp = Path(tmpdir)

        # Isolation setup (done once per trial, shared across all tasks).
        tdc_block_dir = _make_import_block_dir(tmp, PROVIDER.isolation_block_modules())   # Layer 2: PYTHONPATH stub
        empty_dir     = str(tmp / "_empty")         # Layer 1: bind-mount target
        os.makedirs(empty_dir, exist_ok=True)
        tdc_data_dir  = PROVIDER.data_dir()         # Layer 1 target = provider's data dir

        # Probe unshare availability once. On WSL2 this succeeds; on systems
        # without unprivileged user namespaces it falls back gracefully.
        use_unshare = _probe_unshare(tdc_data_dir, empty_dir)
        ns_kwargs   = dict(tdc_data_dir=tdc_data_dir, empty_dir=empty_dir) \
                      if use_unshare else {}

        # ── external data: pre-build test dedup index (once per trial) ───────
        # Runs in data_only AND joint (joint includes the daugm specialist, so
        # its external data must be merged + leakage-filtered too). Only for
        # tasks actually augmented — avoids fingerprinting all 22 test sets.
        test_index: dict = {}
        if ABLATION_MODE in ("data_only", "joint"):
            aug_tasks = [t for t in all_tasks
                         if (WORKDIR / "external_data" / f"{t}.csv").is_file()]
            if aug_tasks:
                print(f"[data_aug] building leakage-safe test index for "
                      f"{len(aug_tasks)} augmented task(s): {aug_tasks}", flush=True)
                test_index = _build_test_dedup_index(group, aug_tasks)

        # ── Task-level cache resolution ──────────────────────────────────────
        tasks_to_run, cached_results = _resolve_tasks_to_run(
            all_tasks, WORKDIR, baseline_scores
        )

        # Inject cached results directly into task_norm and per_task so they
        # count toward aggregate_score without re-running.
        for task_name, cached_r in cached_results.items():
            per_task[task_name]   = cached_r
            norm_val = cached_r.get("norm_improvement")
            task_norm[task_name]  = norm_val if norm_val is not None else 0.0

        n_tasks   = len(all_tasks)
        n_to_run  = len(tasks_to_run)
        print(
            f"[run_trial_drug] {n_to_run}/{n_tasks} tasks to run  "
            f"({n_tasks - n_to_run} from cache)  "
            f"(wall_limit={WALL_LIMIT_S}s, smoke={SMOKE_TEST})",
            flush=True,
        )

        # Track results for cache update at end of trial
        new_task_results: dict[str, dict] = {}

        for task_idx, task_name in enumerate(tasks_to_run, 1):
            t_task = time.monotonic()
            print(f"[{task_idx:2d}/{n_tasks}] {task_name} ...", end="  ", flush=True)

            if time.monotonic() - t0 > WALL_LIMIT_S - 60:
                per_task[task_name] = {"status": "skipped", "reason": "wall_limit"}
                # task_norm[task_name] stays 0.0 — no append needed
                print("SKIPPED (wall_limit)", flush=True)
                continue

            try:
                benchmark    = group.get(task_name)
                train_val_df = benchmark["train_val"]
                test_full_df = benchmark["test"]   # Y stays harness-private

                # Scaffold split train_val → internal train / val.
                train_df, val_df = _scaffold_split(train_val_df)

                # task_type is a benchmark-data property; pass it to the agent
                # subprocess (which has no benchmark package on its path) via env so
                # build_model picks Regressor vs Classifier correctly for ANY benchmark.
                # For TDC this equals the old get_task_type(task) value (zero change).
                os.environ["MAGENT_TASK_TYPE"] = PROVIDER.task_type(task_name)

                # ── merge external data (leakage-safe filter) — data_only + joint ──
                ext_audit: dict = {}
                if ABLATION_MODE in ("data_only", "joint"):
                    _tt = PROVIDER.task_type(task_name)
                    train_df, ext_audit = _merge_external_data(
                        train_df, val_df, task_name, WORKDIR,
                        test_index.get(task_name, {}), _tt,
                    )

                # Fix 1: val Y NEVER touches disk.
                # train_csv has Y (needed for fit); val_x_csv has NO Y column.
                # val_df["Y"] lives only in harness memory for metric computation.
                train_csv    = tmp / f"{task_name}_train.csv"
                val_x_cols   = [c for c in val_df.columns if c != "Y"]
                val_x_csv    = tmp / f"{task_name}_val_x.csv"
                val_pred_csv = tmp / f"{task_name}_val_pred.csv"
                train_df.to_csv(train_csv, index=False)
                val_df[val_x_cols].to_csv(val_x_csv, index=False)
                # val_df["Y"] kept in Python variable — no CSV written

                # ── ABLATION FREEZE (fit) ──────────────────────────────────
                # Restore frozen files to PKG_ROOT baseline BEFORE fit so that
                # any agent edits to frozen files are silently discarded.
                freeze_hashes: dict[str, str] = {}
                if ABLATION_MODE != "joint":
                    freeze_hashes = _ablation_freeze(WORKDIR)

                # FIT: train (with Y) + val_x (NO Y).
                # Early stopping uses an internal holdout carved from train
                # inside fit_model(); agent never sees val Y.
                rc_fit, err_fit = _run_subprocess(
                    [agent_python, experiment_py,
                     "--task", task_name, "--mode", "fit",
                     "--train", str(train_csv),
                     "--val-x", str(val_x_csv),
                     "--model-dir", str(WORKDIR)],
                    timeout_s=min(600, WALL_LIMIT_S - (time.monotonic() - t0)),
                    tdc_block_dir=tdc_block_dir,
                    **ns_kwargs,
                )

                # ── ABLATION VERIFY (post-fit) ─────────────────────────────
                # Confirm fit subprocess did not overwrite any frozen file.
                if freeze_hashes:
                    tampered = _ablation_verify(WORKDIR, freeze_hashes)
                    if tampered:
                        per_task[task_name] = {
                            "status": "crash",
                            "phase":  "ablation_tamper_fit",
                            "reason": f"frozen files modified during fit: {tampered}",
                        }
                        print(
                            f"CRASH(ablation_tamper) fit modified frozen "
                            f"files: {tampered}  [{time.monotonic()-t_task:.1f}s]",
                            flush=True,
                        )
                        continue

                if rc_fit != 0:
                    per_task[task_name] = {"status": "crash", "phase": "fit",
                                           "rc": rc_fit, "err": err_fit[-500:]}
                    print(f"CRASH(fit) rc={rc_fit}  [{time.monotonic()-t_task:.1f}s]\n"
                          f"    {err_fit[-300:].strip()}", flush=True)
                    continue  # task_norm[task_name] stays 0.0

                # ── ABLATION FREEZE (predict) ──────────────────────────────
                # Restore again before predict — fit subprocess might have
                # written back to frozen files as a side-effect.
                if ABLATION_MODE != "joint":
                    freeze_hashes = _ablation_freeze(WORKDIR)

                # PREDICT on val_x (no Y) — harness evaluates against val Y.
                rc_pred, err_pred = _run_subprocess(
                    [agent_python, experiment_py,
                     "--task", task_name, "--mode", "predict",
                     "--input", str(val_x_csv), "--output", str(val_pred_csv),
                     "--model-dir", str(WORKDIR)],
                    timeout_s=min(120, WALL_LIMIT_S - (time.monotonic() - t0)),
                    tdc_block_dir=tdc_block_dir,
                    **ns_kwargs,
                )

                # ── ABLATION VERIFY (post-predict) ─────────────────────────
                if freeze_hashes:
                    tampered = _ablation_verify(WORKDIR, freeze_hashes)
                    if tampered:
                        per_task[task_name] = {
                            "status": "crash",
                            "phase":  "ablation_tamper_predict",
                            "reason": f"frozen files modified during predict: {tampered}",
                        }
                        print(
                            f"CRASH(ablation_tamper) predict modified frozen "
                            f"files: {tampered}  [{time.monotonic()-t_task:.1f}s]",
                            flush=True,
                        )
                        continue

                if rc_pred != 0:
                    per_task[task_name] = {"status": "crash", "phase": "predict",
                                           "rc": rc_pred, "err": err_pred[-500:]}
                    print(f"CRASH(predict) rc={rc_pred}  [{time.monotonic()-t_task:.1f}s]\n"
                          f"    {err_pred[-300:].strip()}", flush=True)
                    continue  # task_norm[task_name] stays 0.0

                # Compute reward using harness-held val labels.
                pred_df   = pd.read_csv(val_pred_csv)
                y_pred    = pred_df["Y"].values
                y_val     = val_df["Y"].values   # never written to disk

                task_type   = PROVIDER.task_type(task_name)
                metric_name = PROVIDER.task_metric(task_name)
                val_metric  = PROVIDER.compute_metric(y_val, y_pred, metric_name)

                base = baseline_scores.get(task_name, {}).get("metric")
                norm = PROVIDER.normalise(val_metric, base, metric_name) if val_metric is not None else None
                task_norm[task_name] = norm if norm is not None else 0.0

                metric_str = f"{val_metric:.4f}" if val_metric is not None else "None"
                print(f"ok  {task_type[:3]}  metric={metric_str}  [{time.monotonic()-t_task:.1f}s]",
                      flush=True)

                result_entry = {
                    "status":           "ok",
                    "task_type":        task_type,
                    "val_metric":       val_metric,
                    "norm_improvement": norm,
                    "n_train":          len(train_df),
                    "n_val":            len(val_df),
                    "n_test":           len(test_full_df),
                }
                if ext_audit:
                    result_entry["data_aug"] = ext_audit
                per_task[task_name]          = result_entry
                new_task_results[task_name]  = result_entry   # eligible for cache

            except Exception as exc:
                per_task[task_name] = {"status": "crash", "reason": str(exc)[:500]}
                print(f"CRASH(exception) {str(exc)[:60]}  [{time.monotonic()-t_task:.1f}s]", flush=True)
                # task_norm[task_name] stays 0.0
                # Do NOT cache crash results

    # ── Update task cache with newly computed ok results ─────────────────────
    if new_task_results and ABLATION_MODE == "feature_only":
        global_hash, task_hashes = _compute_feature_hashes(WORKDIR)
        if global_hash not in ("_error_", _UNRECOGNIZED):
            # Build task_metrics: store val_metric (NOT norm_improvement).
            # norm is recomputed from current baseline on every cache load,
            # so recalibrating baseline_scores.json doesn't stale the cache.
            def _to_metric_entry(r: dict) -> dict:
                return {k: v for k, v in r.items()
                        if k not in ("norm_improvement",)}
            # Merge cached + fresh; fresh overwrites cached
            merged_metrics: dict[str, dict] = {
                k: _to_metric_entry(v)
                for k, v in {**cached_results, **new_task_results}.items()
                if v.get("status") == "ok" and v.get("val_metric") is not None
            }
            _save_task_cache(WORKDIR, global_hash, task_hashes, merged_metrics)

    elapsed = time.monotonic() - t0
    n_ok    = sum(1 for v in per_task.values() if v.get("status") == "ok")

    if n_ok == 0:
        first_err = next((v.get("err") or v.get("reason", "unknown")
                          for v in per_task.values()), "no tasks completed")
        return {
            "status":          "CRASH",
            "aggregate_score": None,
            "n_tasks_ok":      0,
            "elapsed_s":       elapsed,
            "ablation_mode":   ABLATION_MODE,
            "kill_reason":     f"0/{len(all_tasks)} tasks completed: {first_err}",
            "per_task":        per_task,
        }

    # Aggregate over ALL tasks: successes use real norm, failures use 0.0.
    all_norms = list(task_norm.values())
    aggregate = float(np.mean(all_norms))
    n_failed  = len(all_tasks) - n_ok
    return {
        "status":          "OK",
        "aggregate_score": aggregate,
        "n_tasks_ok":      n_ok,
        "n_tasks_failed":  n_failed,
        "elapsed_s":       elapsed,
        "ablation_mode":   ABLATION_MODE,
        "kill_reason":     (f"{n_failed} tasks failed (counted as 0 improvement)"
                            if n_failed else ""),
        "per_task":        per_task,
    }


# ── Entry point ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Drug Discovery trial runner")
    p.add_argument("--out", type=Path, help="Output JSONL path (default: auto)")
    args = p.parse_args(argv)

    out_path = args.out or (OUT_DIR / "run_seed0.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_trial()

    with out_path.open("w") as f:
        json.dump(result, f)
        f.write("\n")

    status = result.get("status", "CRASH")
    score  = result.get("aggregate_score")
    print(f"[run_trial_drug] status={status} aggregate_score={score} "
          f"n_ok={result.get('n_tasks_ok')} elapsed={result.get('elapsed_s'):.1f}s")
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
