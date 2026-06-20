"""Read-only access to extracted PR source files — task-agnostic.

The offline `scripts/build_pr_library/extract_sources.py` pipeline
reconstructs each PR's source tree from the cached unified diff and
writes it under `<task_pkg>/knowledge/pr_library/<tier>/src/pr_XXXX/<orig/path>`,
with a sibling `_manifest.json` listing every extractable file.

This module exposes ONE SDK tool — `read_pr_source` — that fetches a
single file's contents for a given PR number. Modified files (not
fully reconstructable from diff alone) are stored as
`<path>.diff.patch` and returned with `is_patch=True`.

Per-file cap is 40 KB / ~10 K tokens. `offset` + `limit` (in lines)
let the agent paginate larger sources.

Knowledge-base location is read from the active task adapter
(`current_adapter().knowledge_dir`), same as `pr_library.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import tool


_TIERS = ("L1", "L2", "L3", "L_future")
_DEFAULT_BYTE_CAP = 40_000
_DEFAULT_LINE_CAP = 1_000


def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}


def _pr_library_dir() -> Path:
    from agent_core import current_adapter
    return current_adapter().knowledge_dir / "pr_library"


def _find_pr_src_dir(pr_number: int) -> tuple[str, Path] | None:
    """Locate `(tier, src_dir)` for a PR, or None if no extracted tree exists."""
    library_dir = _pr_library_dir()
    for tier in _TIERS:
        candidate = library_dir / tier / "src" / f"pr_{pr_number:04d}"
        if candidate.is_dir() and (candidate / "_manifest.json").is_file():
            return tier, candidate
    return None


def _load_manifest(src_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((src_dir / "_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"files": []}


def _safe_resolve(src_dir: Path, rel_path: str) -> Path | None:
    """Resolve `rel_path` strictly inside `src_dir` (no traversal)."""
    if not rel_path or rel_path.startswith("/"):
        return None
    candidate = (src_dir / rel_path).resolve()
    try:
        candidate.relative_to(src_dir.resolve())
    except ValueError:
        return None
    return candidate


def _read_pr_source_impl(
    pr_number: int,
    path: str,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    if offset < 0:
        return {"ok": False, "error": f"offset must be >= 0 (got {offset})"}
    effective_limit = _DEFAULT_LINE_CAP if (limit is None or limit <= 0) else min(limit, 10_000)

    hit = _find_pr_src_dir(pr_number)
    if hit is None:
        return {
            "ok": False,
            "error": (
                f"PR #{pr_number} has no extracted source tree. This means "
                f"either (a) the PR is not in the library (check INDEX.md), "
                f"or (b) its source has not been extracted yet. Call "
                f"`read_pr_library({pr_number})` first to confirm the PR "
                f"is in the library."
            ),
        }
    tier, src_dir = hit
    manifest = _load_manifest(src_dir)
    available = manifest.get("files", [])

    target = _safe_resolve(src_dir, path)
    if target is None or not target.is_file():
        return {
            "ok": False,
            "error": f"Path {path!r} not found inside PR #{pr_number}.",
            "hint": (
                f"Valid paths in this PR (from _manifest.json, {len(available)} total):"
            ),
            "available_files": available,
        }

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"ok": False, "error": f"failed reading {target}: {e}"}

    is_patch = target.name.endswith(".diff.patch")
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    # Line-level offset/limit first — this is the agent's primary pagination tool
    windowed = lines[offset : offset + effective_limit]
    content = "".join(windowed)

    # Secondary byte cap — defends against long-line pathologies
    byte_truncated = False
    if len(content.encode("utf-8")) > _DEFAULT_BYTE_CAP:
        # Shrink by re-chopping at the byte boundary (keep whole lines where possible)
        running = 0
        kept: list[str] = []
        for ln in windowed:
            enc = ln.encode("utf-8")
            if running + len(enc) > _DEFAULT_BYTE_CAP:
                break
            kept.append(ln)
            running += len(enc)
        content = "".join(kept)
        byte_truncated = True

    end_exclusive = offset + len(content.splitlines(keepends=False))
    has_more = end_exclusive < total_lines

    return {
        "ok": True,
        "pr_number": pr_number,
        "tier": tier,
        "path": path,
        "is_patch": is_patch,
        "content": content,
        "offset": offset,
        "lines_returned": len(content.splitlines(keepends=False)),
        "total_lines": total_lines,
        "has_more": has_more,
        "byte_truncated": byte_truncated,
        "next_offset": end_exclusive if has_more else None,
    }


@tool(
    "read_pr_source",
    (
        "Fetch one source file extracted from an external Parameter Golf PR. "
        "Use this when read_pr_library's summary is too abstract and you "
        "need to see the actual implementation (e.g. the exact forward pass "
        "of a GatedDeltaNet block, or the exact TTT inner loop). "
        "Each PR's available file list is included in the read_pr_library "
        "response under `available_files`. Files suffixed .diff.patch are "
        "unified-diff slices of modified files (full pre-image not "
        "reconstructable); .py/.md/etc are fully reconstructed source. "
        "Use offset+limit (line-based) for files larger than ~1000 lines."
    ),
    {
        "type": "object",
        "properties": {
            "pr_number": {
                "type": "integer",
                "description": "GitHub PR number, e.g. 1711 or 1350.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Repo-relative path as listed in the PR's available_files, "
                    "e.g. 'records/track_10min_16mb/.../train_gpt.py'."
                ),
            },
            "offset": {
                "type": "integer",
                "description": "0-indexed starting line (default 0).",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to return (default 1000, hard max 10000).",
            },
        },
        "required": ["pr_number", "path"],
    },
)
async def read_pr_source(args: dict[str, Any]) -> dict[str, Any]:
    return _mcp(_read_pr_source_impl(
        int(args["pr_number"]),
        str(args["path"]),
        int(args.get("offset", 0)),
        None if args.get("limit") is None else int(args["limit"]),
    ))
