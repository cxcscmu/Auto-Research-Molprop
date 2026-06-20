"""Workdir / snapshot navigation tools — task-agnostic.

A specialist never writes to `snapshots/` directly — that's the
blackboard's job. It only *reads* historical snapshots (to see what a
sibling specialist's winning edit looked like) and *rebases* its own
workdir onto a past snapshot (to fork from exp_042 instead of the
current best).

Like the `code_inspect` tools, each entry point is split into a plain
sync `_<name>_impl(...)` (for direct head-side calls) and an async
`@tool`-decorated MCP wrapper (for SDK tool-use).

The baseline filename (e.g. "train_gpt.py" for PG) is read from the
active task adapter so the same helpers work for any task.
"""

from __future__ import annotations

import difflib
import json
import shutil
from pathlib import Path
from typing import Any, Optional

from . import tool
from ..harness import config, tracker


# ── Shared MCP wrapper ───────────────────────────────────────────────────────

def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}


def _baseline_filename() -> str:
    from agent_core import current_adapter
    return current_adapter().baseline_filename


def _score_field() -> str:
    from agent_core import current_adapter
    return current_adapter().score_field


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_snapshot_dir(exp_id: str) -> Path | None:
    """Map exp_id → blackboard/snapshots/<exp_id>_<domain>/.

    The snapshot dir is named `<exp_id>_<domain>`; we glob by prefix so
    the caller doesn't need to pass the domain. Returns None on miss.
    """
    matches = list(config.SNAPSHOTS_DIR.glob(f"{exp_id}_*"))
    if not matches:
        return None
    # Deterministic pick — shouldn't matter because blackboard enforces
    # exp_id uniqueness, but sort for reproducibility anyway.
    return sorted(matches)[0]


# ── read_snapshot ────────────────────────────────────────────────────────────

_MAX_SNIPPET_BYTES = 200_000


def _read_snapshot_impl(exp_id: str, path: Optional[str] = None) -> dict[str, Any]:
    """Return the snapshotted source text + meta for exp_id.

    `path`: optional snapshot-relative path. Default = baseline_filename
    (single-file behavior preserved for PG/CIFAR). Multi-file tasks
    (NC v2-B) pass e.g. "vendor/nanochat/nanochat/gpt.py".

    Path is restricted to the snapshot's editable surface: the baseline
    file or anything inside the adapter's editable_tree. Absolute paths
    and parent-directory traversal (`..`) are rejected — this is a
    model-callable MCP tool and must not leak file reads outside the
    snapshot.
    """
    snap = _find_snapshot_dir(exp_id)
    if snap is None:
        return {"ok": False, "error": f"no snapshot found for exp_{exp_id}"}

    baseline = _baseline_filename()
    rel = path or baseline

    # Path-bounds check before touching the filesystem.
    if Path(rel).is_absolute():
        return {"ok": False,
                "error": f"absolute paths are rejected: {rel}"}
    # Reject explicit `..` segments AND any rel that resolves outside snap.
    parts = Path(rel).parts
    if any(p == ".." for p in parts):
        return {"ok": False,
                "error": f"parent-directory traversal not allowed: {rel}"}

    from agent_core import current_adapter
    a = current_adapter()
    tree = a.editable_tree
    allowed = (rel == baseline) or (tree and (parts[:1] == (tree,) or rel.startswith(f"{tree}/")))
    if not allowed:
        scope = baseline if not tree else f"{baseline} or {tree}/**"
        return {"ok": False,
                "error": f"path must be inside editable surface ({scope}); got {rel!r}"}

    code_path = snap / rel
    # Resolve and re-verify containment to defend against symlinks.
    try:
        resolved = code_path.resolve()
        snap_resolved = snap.resolve()
        resolved.relative_to(snap_resolved)
    except (OSError, ValueError):
        return {"ok": False,
                "error": f"resolved path escapes snapshot dir: {rel}"}

    if not code_path.is_file():
        return {"ok": False, "error": f"{rel} missing inside {snap}"}

    try:
        raw = code_path.read_bytes()
    except OSError as e:
        return {"ok": False, "error": f"read failed: {e}"}

    truncated = False
    if len(raw) > _MAX_SNIPPET_BYTES:
        raw = raw[:_MAX_SNIPPET_BYTES]
        truncated = True
    content = raw.decode("utf-8", errors="replace")
    if truncated:
        content += "\n\n# … (truncated; snapshot larger than read limit)\n"

    # Cross-reference the TSV row so the agent sees what it produced.
    score_field = _score_field()
    meta: dict[str, Any] = {}
    for r in tracker.read_results():
        if r.get("exp_id") == exp_id:
            meta = {
                "specialist":     r.get("specialist", ""),
                "hypothesis":     r.get("hypothesis", ""),
                score_field:      r.get(score_field, ""),
                "delta_vs_best":  r.get("delta_vs_best", ""),
                "artifact_bytes": r.get("artifact_bytes", ""),
                "status":         r.get("status", ""),
            }
            break

    return {
        "ok":        True,
        "exp_id":    exp_id,
        "path":      str(code_path),
        "rel":       rel,
        "content":   content,
        "truncated": truncated,
        "meta":      meta,
    }


@tool(
    "read_snapshot",
    (
        "Fetch the train_gpt.py source from a past kept experiment. "
        "Use this to study what a sibling specialist (or your own earlier "
        "keep) actually did — the snapshot is frozen at keep-time so you "
        "see the exact code that produced the reported val_bpb. "
        "Returns {ok, exp_id, content, meta}; content is truncated to "
        "~200 KB with a marker. "
        "Optional `path` arg lets multi-file tasks (NC) read any file "
        "inside the snapshot tree (e.g. 'vendor/nanochat/nanochat/gpt.py')."
    ),
    {
        "type": "object",
        "properties": {
            "exp_id": {
                "type": "string",
                "description": "Zero-padded exp_id, e.g. '042'.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Optional snapshot-relative file path. Defaults to the "
                    "task baseline (PG: train_gpt.py, NC: experiment.py, "
                    "CIFAR: airbench94_muon.py). Multi-file tasks pass "
                    "e.g. 'vendor/nanochat/scripts/base_train.py'."
                ),
            },
        },
        "required": ["exp_id"],
    },
)
async def read_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    return _mcp(_read_snapshot_impl(args["exp_id"], args.get("path")))


# ── rebase_to ────────────────────────────────────────────────────────────────

def _rebase_to_impl(exp_id: str, workdir: str) -> dict[str, Any]:
    """Replace workdir's editable surface with the snapshot of exp_id.

    Single-file tasks (PG, CIFAR): copies snapshot/<baseline> over
    workdir/<baseline>.
    Multi-file tasks (NC v2-B): copies snapshot/<baseline> AND
    snapshot/<editable_tree>/ over workdir/* (replacing the existing
    tree with the snapshot's). Idempotent.
    """
    snap = _find_snapshot_dir(exp_id)
    if snap is None:
        return {"ok": False, "error": f"no snapshot found for exp_{exp_id}"}
    from agent_core import current_adapter
    a = current_adapter()
    baseline = a.baseline_filename
    src = snap / baseline
    if not src.is_file():
        return {"ok": False, "error": f"{baseline} missing inside {snap}"}

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    dst = wd / baseline
    shutil.copy2(src, dst)
    bytes_written = dst.stat().st_size
    files_written = 1

    # Multi-file: also replace editable_tree.
    tree = a.editable_tree
    if tree:
        tree_src = snap / tree
        tree_dst = wd / tree
        if tree_src.is_dir():
            # Replace the whole tree with the snapshot's version.
            if tree_dst.exists():
                shutil.rmtree(tree_dst)
            shutil.copytree(tree_src, tree_dst, symlinks=False)
            for f in tree_dst.rglob("*"):
                if f.is_file():
                    bytes_written += f.stat().st_size
                    files_written += 1

    # Task-adapter extra directories (e.g. external_data/ in data_only mode).
    for extra in getattr(a, "snapshot_extra_dirs", []):
        extra_src = snap / extra
        extra_dst = wd / extra
        if extra_src.is_dir():
            if extra_dst.exists():
                shutil.rmtree(extra_dst)
            shutil.copytree(extra_src, extra_dst, symlinks=False)
            for f in extra_dst.rglob("*"):
                if f.is_file():
                    bytes_written += f.stat().st_size
                    files_written += 1
        else:
            # Snapshot has no external_data → clear workdir's copy too
            if extra_dst.exists():
                shutil.rmtree(extra_dst)

    return {
        "ok":            True,
        "parent_exp":    exp_id,
        "workdir":       str(wd),
        "bytes_written": bytes_written,
        "files_written": files_written,
    }


@tool(
    "rebase_to",
    (
        "Copy the snapshot of exp_id's train_gpt.py into your own workdir, "
        "overwriting whatever is currently there. Use this to fork from a "
        "non-best parent (e.g. to revive a discarded branch). Returns "
        "{ok, parent_exp, workdir, bytes_written}."
    ),
    {
        "type": "object",
        "properties": {
            "exp_id":  {"type": "string"},
            "workdir": {
                "type": "string",
                "description": "Your own workdir_<domain>/ path.",
            },
        },
        "required": ["exp_id", "workdir"],
    },
)
async def rebase_to(args: dict[str, Any]) -> dict[str, Any]:
    return _mcp(_rebase_to_impl(args["exp_id"], args["workdir"]))


# ── diff_snapshots ───────────────────────────────────────────────────────────

_MAX_DIFF_BYTES = 8_000
_MAX_DIFF_LINES = 300


def _diff_snapshots_impl(exp_a: str, exp_b: str) -> dict[str, Any]:
    """Unified diff between two snapshots' editable surfaces.

    Single-file tasks: just diff <baseline_filename>.
    Multi-file tasks (NC v2-B): walk both editable trees, emit diffs for
    every file present in either snapshot. Output capped at 300 lines /
    8 KB total across all files.
    """
    if exp_a == exp_b:
        return {
            "ok":        True,
            "exp_a":     exp_a,
            "exp_b":     exp_b,
            "diff":      "",
            "truncated": False,
            "note":      "exp_a == exp_b; no diff",
        }

    from agent_core import current_adapter
    a = current_adapter()
    baseline = a.baseline_filename
    snap_a = _find_snapshot_dir(exp_a)
    if snap_a is None:
        return {"ok": False, "error": f"no snapshot found for exp_{exp_a}"}
    snap_b = _find_snapshot_dir(exp_b)
    if snap_b is None:
        return {"ok": False, "error": f"no snapshot found for exp_{exp_b}"}

    # Build the union set of relative paths to diff.
    rels: list[str] = [baseline]
    tree = a.editable_tree
    if tree:
        # Union of .py files across both snapshots' editable_tree.
        seen: set[str] = set()
        for snap in (snap_a, snap_b):
            tree_root = snap / tree
            if tree_root.is_dir():
                for f in tree_root.rglob("*.py"):
                    if f.is_file():
                        seen.add(str(f.relative_to(snap)))
        rels.extend(sorted(seen))

    chunks: list[str] = []
    total_bytes = 0
    total_lines = 0
    truncated = False
    files_diffed = 0
    files_changed = 0

    for rel in rels:
        if truncated:
            break
        code_a = snap_a / rel
        code_b = snap_b / rel
        try:
            text_a = code_a.read_text(encoding="utf-8", errors="replace") if code_a.is_file() else ""
            text_b = code_b.read_text(encoding="utf-8", errors="replace") if code_b.is_file() else ""
        except OSError:
            continue
        if text_a == text_b:
            files_diffed += 1
            continue
        lines_a = text_a.splitlines(keepends=True)
        lines_b = text_b.splitlines(keepends=True)
        files_changed += 1
        files_diffed += 1
        for line in difflib.unified_diff(
            lines_a, lines_b,
            fromfile=f"exp_{exp_a}/{rel}",
            tofile=f"exp_{exp_b}/{rel}",
            n=3,
        ):
            total_lines += 1
            total_bytes += len(line.encode("utf-8"))
            if total_lines > _MAX_DIFF_LINES or total_bytes > _MAX_DIFF_BYTES:
                truncated = True
                break
            chunks.append(line)

    diff_text = "".join(chunks)
    if truncated:
        diff_text += (
            f"\n… (truncated; diff exceeded "
            f"{_MAX_DIFF_LINES} lines / {_MAX_DIFF_BYTES} bytes after "
            f"{files_diffed} files — use read_snapshot on specific paths "
            f"for the full content)\n"
        )

    score_field = _score_field()
    rows_by_id = {r.get("exp_id"): r for r in tracker.read_results()}

    def _meta(exp_id: str) -> dict[str, Any]:
        r = rows_by_id.get(exp_id) or {}
        return {
            "specialist":    r.get("specialist", ""),
            "hypothesis":    r.get("hypothesis", ""),
            score_field:     r.get(score_field, ""),
            "delta_vs_best": r.get("delta_vs_best", ""),
            "status":        r.get("status", ""),
        }

    return {
        "ok":            True,
        "exp_a":         exp_a,
        "exp_b":         exp_b,
        "diff":          diff_text,
        "truncated":     truncated,
        "files_diffed":  files_diffed,
        "files_changed": files_changed,
        "meta_a":        _meta(exp_a),
        "meta_b":        _meta(exp_b),
    }


@tool(
    "diff_snapshots",
    (
        "Unified diff of train_gpt.py between two past experiments' "
        "snapshots. Use this when you want to see exactly what changed "
        "between exp_a and exp_b without reading both files in full. "
        "Output is truncated to ~300 lines / ~8 KB with a marker; for "
        "larger diffs, call read_snapshot on each side. Returns "
        "{ok, exp_a, exp_b, diff, truncated, meta_a, meta_b}."
    ),
    {
        "type": "object",
        "properties": {
            "exp_a": {"type": "string", "description": "First exp_id (zero-padded, e.g. '042')."},
            "exp_b": {"type": "string", "description": "Second exp_id (zero-padded)."},
        },
        "required": ["exp_a", "exp_b"],
    },
)
async def diff_snapshots(args: dict[str, Any]) -> dict[str, Any]:
    return _mcp(_diff_snapshots_impl(args["exp_a"], args["exp_b"]))
