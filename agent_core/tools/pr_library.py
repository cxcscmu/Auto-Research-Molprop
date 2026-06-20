"""Read-only access to the external PR knowledge base — task-agnostic.

The PR library is built offline by the task package's
`scripts/build_pr_library/` (scrape → LLM summarize → render) and lives
under `<task_pkg>/knowledge/pr_library/`. Each candidate PR has its own
markdown file under `L1/`, `L2/`, `L3/`, or `L_future/`.

INDEX.md, techniques.md, and gaps.md are injected into every
specialist's system prompt (see `agents.prompts`). Per-PR details are
NOT in the prompt (would balloon the static prefix) -- the agent calls
`read_pr_library(pr_number)` on demand.

Knowledge-base location is read from the active task adapter
(`current_adapter().knowledge_dir`) so the same helper works for any
task that has its own PR-style external library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import tool


# Apr-25 re-tier: single visible "L2" tier (no L1/L3) plus deferred-scope
# "L_future". L3 PRs are now under .archive/ — agent-invisible by design,
# so the glob below intentionally excludes them. See render_v2.py.
_TIERS = ("L2", "L_future")
_MAX_BYTES = 12_000  # cap payload so a verbose PR can't blow prompt budget


def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}


def _pr_library_dir() -> Path:
    """Resolve `<task_pkg>/knowledge/pr_library/` via the active adapter."""
    from agent_core import current_adapter
    return current_adapter().knowledge_dir / "pr_library"


def _find_pr_md(pr_number: int) -> tuple[str, Path] | None:
    """Locate (tier, file) for a given PR number, or None if not in library."""
    pattern = f"pr_{pr_number:04d}_*.md"
    library_dir = _pr_library_dir()
    for tier in _TIERS:
        tier_dir = library_dir / tier
        if not tier_dir.is_dir():
            continue
        for p in sorted(tier_dir.glob(pattern)):
            return tier, p
    return None


def _load_available_files(tier: str, pr_number: int) -> list[str]:
    """Return the list of source paths extracted for this PR, if any.

    Built offline by `scripts/build_pr_library/extract_sources.py` and
    stored as `knowledge/pr_library/<tier>/src/pr_XXXX/_manifest.json`.
    An empty list means: no source has been extracted for this PR
    (either the PR has no extractable text files, or extract_sources.py
    hasn't been run yet against this tier).
    """
    manifest_path = (
        _pr_library_dir() / tier / "src" / f"pr_{pr_number:04d}" / "_manifest.json"
    )
    if not manifest_path.is_file():
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    files = data.get("files") or []
    return sorted(str(f) for f in files)


def _read_pr_library_impl(pr_number: int) -> dict[str, Any]:
    hit = _find_pr_md(pr_number)
    if hit is None:
        from agent_core import current_adapter
        score_field = current_adapter().score_field
        return {
            "ok": False,
            "error": (
                f"PR #{pr_number} not in the library. Valid PR numbers are "
                f"listed in INDEX.md (already in your system prompt). The "
                f"library only contains records whose reported {score_field} beat "
                f"our current SOTA threshold."
            ),
        }
    tier, path = hit
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return {"ok": False, "error": f"failed reading {path}: {e}"}
    truncated = False
    if len(content) > _MAX_BYTES:
        content = content[:_MAX_BYTES] + (
            f"\n\n... [truncated at {_MAX_BYTES} bytes]\n"
        )
        truncated = True

    available_files = _load_available_files(tier, pr_number)
    return {
        "ok": True,
        "pr_number": pr_number,
        "tier": tier,
        "path": f"knowledge/pr_library/{tier}/{path.name}",
        "content": content,
        "truncated": truncated,
        "available_files": available_files,
        "source_hint": (
            f"Call `read_pr_source({pr_number}, '<path>')` to fetch any of "
            f"the {len(available_files)} listed file(s) for implementation "
            f"detail."
            if available_files
            else "No source files extracted for this PR (summary only)."
        ),
    }


@tool(
    "read_pr_library",
    (
        # Description text retained verbatim from the original PG-only impl
        # so MCP tool description (visible to the model) is byte-equivalent.
        # nc/cifar fork can override the description by registering their
        # own variant in their task package's tools/ — adapter.bind_tools()
        # lets each task return its own @tool-decorated callable.
        "Look up an external Parameter Golf PR from the knowledge base by "
        "integer PR number. Returns that PR's full structured summary -- "
        "core_idea, what_changed (with file:line citations), reported "
        "metrics, caveats, scope_category, tags -- as captured from the "
        "public GitHub repo. Use this when INDEX.md / techniques.md / "
        "gaps.md (already in your system prompt) reference a PR number you "
        "want to dig into. These PRs are benchmarks to BEAT, not recipes "
        "to copy verbatim. Returns {ok, pr_number, tier, path, content, "
        "truncated}; content is capped at 12 KB."
    ),
    {
        "type": "object",
        "properties": {
            "pr_number": {
                "type": "integer",
                "description": "GitHub PR number, e.g. 1711 or 1350.",
            },
        },
        "required": ["pr_number"],
    },
)
async def read_pr_library(args: dict[str, Any]) -> dict[str, Any]:
    return _mcp(_read_pr_library_impl(int(args["pr_number"])))
