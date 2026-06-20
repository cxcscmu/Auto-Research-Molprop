"""Shared blackboard — task-agnostic shared blackboard.

Concurrency model
─────────────────
Specialists each run inside their own Claude SDK session on the laptop
supervisor. They all read/write the SAME blackboard files on local disk.
Every write path that touches the TSV or regenerates an MD file must
hold `blackboard_lock()` — a filelock-backed critical section. Reads do
not lock (worst-case stale view is fine; each write is atomic-rename to
keep readers consistent).

Snapshots
─────────
Every `keep` trial's full workdir is frozen under blackboard/snapshots/
<exp_id>_<domain>/. We retain ALL keeps because snapshots are small and
let future specialists rebase onto the exact winning source.

Stop flag
─────────
A supervisor process checks DEADLINE and no-improvement separately and
writes `stop.flag` when either triggers. Specialists must consult
`should_stop()` at the top of every iteration and exit gracefully.

Task-specific bits
──────────────────
The primary score field name (`val_bpb` for PG) and the baseline source
filename (`train_gpt.py` for PG) are read from the active task adapter.
Other task-specific touches (file copies on snapshot, etc.) currently
default to PG conventions; nc/cifar forks override at the task package
level if needed.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import shutil
import time
from pathlib import Path
from typing import Iterator, Optional

import filelock

from . import config, tracker

# ── Lock primitives ──────────────────────────────────────────────────────────

_LOCK_FILE = "blackboard.lock"                 # lives under LOCKS_DIR
_LOCK_ACQUIRE_TIMEOUT_S = 120                  # one critical section should never exceed this


@contextlib.contextmanager
def blackboard_lock() -> Iterator[None]:
    """Exclusive filelock on the blackboard. Use around every mutation."""
    config.LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = config.LOCKS_DIR / _LOCK_FILE
    lock = filelock.FileLock(str(lock_path), timeout=_LOCK_ACQUIRE_TIMEOUT_S)
    with lock:
        yield


# ── Adapter helpers ──────────────────────────────────────────────────────────

def _adapter():
    from agent_core import current_adapter
    return current_adapter()


def _score_field() -> str:
    return _adapter().score_field


def _score_lower_is_better() -> bool:
    return _adapter().score_lower_is_better


def _baseline_filename() -> str:
    return _adapter().baseline_filename


# ── Stop-flag API ────────────────────────────────────────────────────────────

def should_stop() -> bool:
    """True when the supervisor (or a human) has placed stop.flag."""
    return config.STOP_FLAG.exists()


def request_stop(reason: str) -> None:
    """Drop stop.flag atomically with a one-line reason string."""
    config.STOP_FLAG.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.STOP_FLAG.with_suffix(".tmp")
    tmp.write_text(
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  {reason}\n",
        encoding="utf-8",
    )
    tmp.replace(config.STOP_FLAG)


# ── best.json ────────────────────────────────────────────────────────────────

def _read_best() -> dict:
    if not config.BEST_JSON.exists():
        return {}
    try:
        return json.loads(config.BEST_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_best() -> Optional[dict]:
    """Return the cached best.json, or None if no keep has ever happened."""
    data = _read_best()
    return data or None


def _write_best(payload: dict) -> None:
    tmp = config.BEST_JSON.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(config.BEST_JSON)


# ── Snapshots ────────────────────────────────────────────────────────────────

def _snapshot_experiment(exp_id: str, domain: str, workdir: Path) -> Path:
    """Freeze the contents of a specialist workdir into blackboard/snapshots/.

    Copies baseline source + (multi-file tasks) the editable_tree
    recursively + helper scripts + eval outputs. Excludes runtime
    artifacts under `trial_output_dirs` (already classified into the
    eval/ sub-dir below).

    Returns the snapshot dir relative to BLACKBOARD_DIR.
    """
    snap_dir = config.SNAPSHOTS_DIR / f"{exp_id}_{domain}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    adapter = _adapter()
    baseline = adapter.baseline_filename
    for rel in (baseline, "pack_submission.py"):
        src = workdir / rel
        if src.is_file():
            shutil.copy2(src, snap_dir / rel)

    # Multi-file tasks (NC v2-B): snapshot the entire editable tree so
    # rebase_to / diff_snapshots / read_snapshot can reconstruct the
    # full editable surface that produced this exp_id's score.
    tree = adapter.editable_tree
    if tree:
        tree_src = workdir / tree
        tree_dst = snap_dir / tree
        if tree_src.is_dir():
            # Replace if exists (rare; defensive against stale half-copy)
            if tree_dst.exists():
                shutil.rmtree(tree_dst)
            shutil.copytree(tree_src, tree_dst, symlinks=False)

    # Packed code blob (PG: ckpt/train_gpt_packed.py). Filename pattern
    # mirrors the baseline name; non-PG forks may need to override.
    packed_name = baseline.replace(".py", "_packed.py")
    packed = workdir / "ckpt" / packed_name
    if packed.is_file():
        shutil.copy2(packed, snap_dir / packed_name)

    # run_trial.sh writes into full_eval_results/<workdir-name>/
    fe_root = workdir / "full_eval_results"
    if fe_root.is_dir():
        for sub in fe_root.iterdir():
            if not sub.is_dir():
                continue
            dest = snap_dir / "eval"
            dest.mkdir(exist_ok=True)
            for name in ("run_seed0.jsonl", "run_seed0.log",
                         "summary.json", "result.json"):
                src = sub / name
                if src.is_file():
                    shutil.copy2(src, dest / name)
            break   # run_trial.sh only writes one subdir per run

    # Task-adapter extra directories (e.g. external_data/ in data_only mode).
    for extra in getattr(adapter, "snapshot_extra_dirs", []):
        extra_src = workdir / extra
        extra_dst = snap_dir / extra
        if extra_src.is_dir():
            if extra_dst.exists():
                shutil.rmtree(extra_dst)
            shutil.copytree(extra_src, extra_dst, symlinks=False)

    return snap_dir.relative_to(config.BLACKBOARD_DIR)


# ── Main commit path ─────────────────────────────────────────────────────────

def record_trial(
    *,
    specialist: str,
    domain: str,
    parent_exp: str,
    hypothesis: str,
    expected_delta: str,
    validate_row: dict,
    sched_job_name: str,
    workdir: Path,
    notes: str = "",
    keep_decision: Optional[bool] = None,
) -> dict:
    """Serialise a finished trial → TSV + best.json + snapshot (if keep)."""
    score_field = _score_field()
    lower_is_better = _score_lower_is_better()
    with blackboard_lock():
        exp_id = tracker.next_exp_id()
        best = _read_best()
        baseline_exp = best.get("exp_id", "") if best else parent_exp
        best_score = best.get(score_field) if best else None

        coarse = validate_row.get("status", "crash")
        score_str = validate_row.get(score_field, "")
        delta_str = ""
        final_status = coarse

        if coarse == "keep":
            try:
                new_score = float(score_str)
                if best_score is not None:
                    delta = new_score - float(best_score)
                    delta_str = f"{delta:+.6f}"
                    is_better = (delta < 0) if lower_is_better else (delta > 0)
                else:
                    is_better = True
                if keep_decision is None:
                    final_status = "keep" if is_better else "discard"
                else:
                    final_status = "keep" if keep_decision else "discard"
            except (TypeError, ValueError):
                final_status = "discard"

        # Always-snapshot of baseline source for EVERY iter.
        try:
            tg = workdir / _baseline_filename()
            if tg.is_file():
                all_dir = config.SNAPSHOTS_DIR / "all"
                all_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tg, all_dir / f"{exp_id}.py")
        except OSError:
            pass

        snapshot_rel = ""
        if final_status == "keep":
            rel = _snapshot_experiment(exp_id, domain, workdir)
            snapshot_rel = str(rel)

        row = {
            "exp_id":         exp_id,
            "timestamp":      datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "specialist":     specialist,
            "parent_exp":     parent_exp,
            "baseline_exp":   baseline_exp,
            "domain":         domain,
            "hypothesis":     hypothesis,
            "expected_delta": expected_delta,
            "status":         final_status,
            score_field:      score_str,
            "delta_vs_best":  delta_str,
            "artifact_bytes": str(validate_row.get("artifact_bytes", "")),
            "train_s":        validate_row.get("train_s", ""),
            "eval_s":         validate_row.get("eval_s", ""),
            "total_s":        validate_row.get("total_s", ""),
            "sched_job_name":    sched_job_name,
            "snapshot_path":  snapshot_rel,
            "notes":          notes,
        }
        # Forward task-specific validate_row fields (e.g. CIFAR's accuracy /
        # acc_std / n_seeds) into the row so they actually land in the TSV.
        # Pre-existing keys above (status, score, train_s, etc.) are NOT
        # overwritten — they're already authoritatively set. tracker's
        # extrasaction="ignore" then drops anything not in tsv_fields, so
        # this is safe to over-include.
        for k, v in validate_row.items():
            if k not in row:
                row[k] = v
        tracker.append_result(row)

        if final_status == "keep":
            try:
                new_score = float(score_str)
                if best_score is None:
                    is_new_best = True
                else:
                    bs = float(best_score)
                    is_new_best = (new_score < bs) if lower_is_better else (new_score > bs)
                if is_new_best:
                    _write_best({
                        "exp_id":        exp_id,
                        "specialist":    specialist,
                        "domain":        domain,
                        "hypothesis":    hypothesis,
                        score_field:     new_score,
                        "delta_vs_prev": delta_str,
                        "timestamp":     row["timestamp"],
                        "snapshot_path": snapshot_rel,
                    })
            except (TypeError, ValueError):
                pass

        regenerate_markdown()
        return row


# ── MD regeneration ──────────────────────────────────────────────────────────

_MAX_LEADERBOARD_ROWS = 20


def regenerate_markdown() -> None:
    """Rewrite LEADERBOARD.md, KNOWLEDGE.md, and tree.tsv."""
    rows = tracker.read_results()
    _write_leaderboard(rows)
    _write_tree_tsv(rows)
    _write_knowledge(rows)


def _write_leaderboard(rows: list[dict]) -> None:
    score_field = _score_field()
    kept = [r for r in rows if r.get("status") == "keep"]
    lower_is_better = _score_lower_is_better()
    sentinel = float("inf") if lower_is_better else float("-inf")
    kept.sort(
        key=lambda r: float(r.get(score_field, sentinel) or sentinel),
        reverse=not lower_is_better,
    )
    best = _read_best()

    lines: list[str] = [
        "# Leaderboard",
        "*Auto-generated by harness/blackboard.py — do not edit by hand.*",
        "",
    ]
    if best:
        lines += [
            "## Current Best",
            f"- **exp_id**: {best.get('exp_id', '?')}",
            f"- **{score_field}**: {best.get(score_field, '?')}",
            f"- **specialist**: {best.get('specialist', '?')}",
            f"- **hypothesis**: {best.get('hypothesis', '')[:120]}",
            f"- **snapshot**: `{best.get('snapshot_path', '')}`",
            "",
        ]
    else:
        lines += ["## Current Best", "*(no VALID runs yet)*", ""]

    lines += [
        f"## Top {_MAX_LEADERBOARD_ROWS} Kept Trials",
        "",
        f"| exp | {score_field} | Δ | specialist | hypothesis |",
        "|-----|---------|---|------------|------------|",
    ]
    for r in kept[:_MAX_LEADERBOARD_ROWS]:
        hyp = (r.get("hypothesis", "") or "").replace("|", "∣")
        if len(hyp) > 100:
            hyp = hyp[:97] + "..."
        lines.append(
            f"| {r.get('exp_id', '?')} "
            f"| {r.get(score_field, '')} "
            f"| {r.get('delta_vs_best', '') or '—'} "
            f"| {r.get('specialist', '')} "
            f"| {hyp} |"
        )
    if not kept:
        lines.append("| *(none yet)* | | | | |")

    _atomic_write(config.LEADERBOARD_MD, "\n".join(lines) + "\n")


def _write_knowledge(rows: list[dict]) -> None:
    """Regenerate KNOWLEDGE.md — current-best lineage + recent activity."""
    score_field = _score_field()
    preserved = ""
    if config.KNOWLEDGE_MD.exists():
        existing = config.KNOWLEDGE_MD.read_text(encoding="utf-8")
        marker = "## Key Insights"
        idx = existing.find("\n" + marker)
        if idx != -1:
            preserved = existing[idx + 1:].rstrip()

    lines: list[str] = [
        "# Research Knowledge Base",
        "*Auto-generated — the `## Key Insights` section at the bottom is preserved.*",
        "",
        "## Research Tree",
        f"Full tree is at `{config.TREE_TSV}` (TSV; columns: "
        f"`exp_id, parent_exp, depth, path, specialist, status, {score_field}, "
        "delta_vs_best, hypothesis`; preorder-sorted so siblings are "
        "adjacent). Slice it from Bash, e.g.:",
        "",
        "```",
        "# subtree rooted at exp_042 (all descendants + the node itself)",
        f"awk -F '\\t' 'NR==1 || $4 ~ /(^|\\/)042(\\/|$)/' {config.TREE_TSV}",
        "# direct children of exp_042",
        f"awk -F '\\t' 'NR==1 || $2==\"042\"' {config.TREE_TSV}",
        f"# only kept trials, sorted by {score_field}",
        f"awk -F '\\t' 'NR==1 || $6==\"keep\"' {config.TREE_TSV} | sort -t $'\\t' -k7,7g",
        "```",
        "",
    ]
    lines += _render_best_lineage(rows)
    lines += [
        "",
        "## Recent Activity (last 30)",
        "",
        f"| exp | specialist | status | {score_field} | hypothesis |",
        "|-----|------------|--------|---------|------------|",
    ]
    visible_rows = [r for r in rows if r.get("status") != "harness_abort"]
    for r in list(reversed(visible_rows))[:30]:
        hyp = (r.get("hypothesis", "") or "").replace("|", "∣")
        if len(hyp) > 80:
            hyp = hyp[:77] + "..."
        lines.append(
            f"| {r.get('exp_id', '?')} "
            f"| {r.get('specialist', '')} "
            f"| {r.get('status', '')} "
            f"| {r.get(score_field, '') or '—'} "
            f"| {hyp} |"
        )
    if not rows:
        lines.append("| *(none yet)* | | | | |")
    lines.append("")

    if preserved:
        lines.append(preserved)
    else:
        lines += [
            "## Key Insights",
            "*(Add manually — this section is preserved across regenerations.)*",
            "",
        ]

    _atomic_write(config.KNOWLEDGE_MD, "\n".join(lines) + "\n")


def _build_tree_index(rows: list[dict]) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Return (exp_map, children) indexed by exp_id. Children are sorted."""
    children: dict[str, list[str]] = {}
    exp_map: dict[str, dict] = {}
    for r in rows:
        eid = r.get("exp_id", "")
        if not eid:
            continue
        exp_map[eid] = r
        children.setdefault(r.get("parent_exp", "") or "", []).append(eid)
    for kids in children.values():
        kids.sort()
    return exp_map, children


def _render_best_lineage(rows: list[dict]) -> list[str]:
    """Render root→best ancestor chain plus best's direct children."""
    adapter = _adapter()
    score_field = adapter.score_field
    short = adapter.score_short_label
    exp_map, children = _build_tree_index(rows)
    if not exp_map:
        return ["```", "(no experiments yet)", "```"]

    best = _read_best()
    best_exp = best.get("exp_id") if best else None
    if not best_exp or best_exp not in exp_map:
        return ["```", "(no VALID trials yet — see tree.tsv for full state)", "```"]

    chain: list[str] = []
    cur: Optional[str] = best_exp
    visited: set[str] = set()
    while cur and cur in exp_map and cur not in visited:
        visited.add(cur)
        chain.append(cur)
        cur = exp_map[cur].get("parent_exp") or None
    chain.reverse()

    def line(eid: str, marker: str = "") -> str:
        r = exp_map.get(eid, {})
        bpb = r.get(score_field, "") or "—"
        delta = r.get("delta_vs_best", "")
        delta_str = f", Δ={delta}" if delta and delta != "—" else ""
        spec = r.get("specialist", "?")
        status = r.get("status", "?")
        hyp = (r.get("hypothesis", "") or "")[:100]
        return f"exp_{eid} [{spec}, {status}, {short}={bpb}{delta_str}] {hyp}{marker}"

    out = ["**Current-best lineage** (root → best):", "```"]
    for i, eid in enumerate(chain):
        prefix = "   " * i + ("└─ " if i > 0 else "")
        marker = "  ← BEST" if eid == best_exp else ""
        out.append(f"{prefix}{line(eid, marker)}")
    kids = children.get(best_exp, [])
    if kids:
        out.append("")
        out.append(f"**Best's direct children ({len(kids)}):**")
        for kid in kids:
            out.append(f"  • {line(kid)}")
    out.append("```")
    return out


def _write_tree_tsv(rows: list[dict]) -> None:
    """Emit tree.tsv — one row per experiment, preorder-sorted by lineage."""
    score_field = _score_field()
    exp_map, children = _build_tree_index(rows)

    header = "\t".join([
        "exp_id", "parent_exp", "depth", "path",
        "specialist", "status", score_field, "delta_vs_best", "hypothesis",
    ])
    out_lines: list[str] = [header]

    def walk(eid: str, depth: int, path: str) -> None:
        r = exp_map.get(eid, {})
        hyp = (r.get("hypothesis", "") or "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
        out_lines.append("\t".join([
            eid,
            r.get("parent_exp", "") or "",
            str(depth),
            path,
            r.get("specialist", "") or "",
            r.get("status", "") or "",
            r.get(score_field, "") or "",
            r.get("delta_vs_best", "") or "",
            hyp,
        ]))
        for kid in children.get(eid, []):
            walk(kid, depth + 1, f"{path}/{kid}")

    roots = sorted(eid for eid, r in exp_map.items()
                   if not r.get("parent_exp") or r["parent_exp"] not in exp_map)
    for root in roots:
        walk(root, 0, root)

    _atomic_write(config.TREE_TSV, "\n".join(out_lines) + "\n")


def _atomic_write(path: Path, content: str) -> None:
    """Write via tmpfile + rename so partial writes never surface to readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


# ── Bootstrap helper ─────────────────────────────────────────────────────────

def bootstrap_from_baseline(baseline_row: dict) -> None:
    """Seed the blackboard with a synthetic `baseline` row."""
    adapter = _adapter()
    score_field = adapter.score_field
    with blackboard_lock():
        rows = tracker.read_results()
        if any(r.get("status") == "baseline" for r in rows):
            return
        row = {
            "exp_id":         "000",
            "timestamp":      datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "specialist":     "baseline",
            "parent_exp":     "",
            "baseline_exp":   "",
            "domain":         "baseline",
            "hypothesis":     baseline_row.get("hypothesis", adapter.bootstrap_hypothesis),
            "expected_delta": "0",
            "status":         "baseline",
            score_field:      baseline_row[score_field],
            "delta_vs_best":  "",
            "artifact_bytes": baseline_row.get("artifact_bytes", ""),
            "train_s":        baseline_row.get("train_s", ""),
            "eval_s":         baseline_row.get("eval_s", ""),
            "total_s":        baseline_row.get("total_s", ""),
            "sched_job_name":    "",
            "snapshot_path":  baseline_row.get("snapshot_path", ""),
            "notes":          adapter.baseline_note,
        }
        tracker.append_result(row)
        _write_best({
            "exp_id":        "000",
            "specialist":    "baseline",
            "domain":        "baseline",
            "hypothesis":    row["hypothesis"],
            score_field:     float(row[score_field]),
            "delta_vs_prev": "",
            "timestamp":     row["timestamp"],
            "snapshot_path": row["snapshot_path"],
        })
        regenerate_markdown()


# ── Live-status helper for the supervisor ────────────────────────────────────

def wait_for_lock_clear(timeout_s: float = 5.0) -> bool:
    """Sanity probe used at startup: try to acquire+release the lock."""
    lock_path = config.LOCKS_DIR / _LOCK_FILE
    lock = filelock.FileLock(str(lock_path), timeout=timeout_s)
    t0 = time.monotonic()
    try:
        with lock:
            pass
        return True
    except filelock.Timeout:
        return False
    finally:
        _ = time.monotonic() - t0     # noqa: intentional side-effect-free timer
