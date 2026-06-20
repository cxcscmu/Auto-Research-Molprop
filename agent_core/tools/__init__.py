"""SDK @tool wrappers — shim + decorator (task-agnostic).

Every public tool is decorated with `@tool` so the SDK can register it
into an in-process MCP server (see `create_sdk_mcp_server`). The
underlying sync work lives in companion modules (`code_inspect`,
`workdir`, `submit`, `pr_library`, `pr_source`) as plain
`_<name>_impl(...)` functions so they're unit-testable without a
running SDK session.

Local-dev shim
──────────────
`claude_agent_sdk` may not be installed in lightweight dev environments.
We provide a tiny shim for both `@tool` and `create_sdk_mcp_server` so
importing this package succeeds on the laptop; the real names replace
the shim transparently at supervisor runtime (head node has the SDK).

The shim mimics just enough of the real API surface to let
`base.py.run_once` *type-check* locally — it never actually runs an SDK
session (that path raises RuntimeError on the laptop).

Generic tools live in core (workdir, pr_library, pr_source, syntax_check
+ param_count from code_inspect). Task-specific tools (size_project,
submit_trial in PG; whatever cifar / nc add) live in their task package
and are exposed via the task package's own `tools/__init__.py`
composition.
"""

from __future__ import annotations

try:
    from claude_agent_sdk import tool, create_sdk_mcp_server  # type: ignore[import-not-found]
except ImportError:
    from dataclasses import dataclass
    from typing import Any, Awaitable, Callable

    @dataclass
    class _ShimTool:
        """Stand-in for the SDK's SdkMcpTool — same attribute names."""
        name: str
        description: str
        input_schema: Any
        handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

    def tool(name: str, description: str, input_schema, annotations=None):  # type: ignore[no-redef]
        """Shim @tool: preserves name/description/schema + the async handler."""
        def decorate(fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]) -> _ShimTool:
            return _ShimTool(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=fn,
            )
        return decorate

    def create_sdk_mcp_server(name: str, version: str = "1.0.0", tools=None):  # type: ignore[no-redef]
        """Shim server: returns a plain dict describing the registration."""
        return {
            "type":    "sdk-shim",
            "name":    name,
            "version": version,
            "tools":   list(tools or []),
        }


# Re-exports of generic tools — populated as Day 5 progresses. The task
# package's own `tools/__init__.py` is responsible for composing the final
# tool list (generic from here + task-specific from its own modules).
from .code_inspect import syntax_check, param_count
from .workdir import diff_snapshots, read_snapshot, rebase_to
from .pr_library import read_pr_library
from .pr_source import read_pr_source

# ── with_description helper (Gap 7 / nc + cifar wrappers) ────────────────────
#
# Each task package builds its own bind_tools() list. nc/cifar will wrap
# core tools with task-specific descriptions (the description is
# model-visible context; round-3 audit lesson). PG keeps the originals →
# byte-equal preserved.

def with_description(tool_obj, new_description: str):
    """Return a new tool wrapper with `description` swapped — original
    untouched. Works for both _ShimTool (laptop-dev) and the real SDK's
    SdkMcpTool (head-node)."""
    return _swap(tool_obj, description=new_description)


def with_overrides(tool_obj, *, description: str | None = None,
                   input_schema=None):
    """Return a new tool wrapper with `description` and/or `input_schema`
    swapped. Original untouched. Pass only the fields you want to override.

    Tasks use this when both the description and the JSON schema's example
    text are PG-flavored (e.g. expected_delta=-0.002 doesn't fit cifar's
    higher-better accuracy direction). Example:

        cifar_submit = with_overrides(
            core_submit_trial,
            description="Submit … airbench94_muon.py …",
            input_schema={... "expected_delta": {"description": "+0.005 …"}}
        )
    """
    fields = {}
    if description is not None:
        fields["description"] = description
    if input_schema is not None:
        fields["input_schema"] = input_schema
    return _swap(tool_obj, **fields)


def _swap(tool_obj, **fields):
    """Internal: copy tool_obj with fields replaced, supporting both
    dataclass (_ShimTool) and arbitrary attribute-bearing objects."""
    if not fields:
        return tool_obj
    try:
        import dataclasses
        if dataclasses.is_dataclass(tool_obj):
            return dataclasses.replace(tool_obj, **fields)
    except Exception:
        pass
    import copy as _copy
    new = _copy.copy(tool_obj)
    for k, v in fields.items():
        setattr(new, k, v)
    return new


__all__ = [
    "tool",
    "create_sdk_mcp_server",
    "with_description",
    "with_overrides",
    "syntax_check",
    "param_count",
    "read_snapshot",
    "rebase_to",
    "diff_snapshots",
    "read_pr_library",
    "read_pr_source",
]
