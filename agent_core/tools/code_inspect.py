"""Cheap, local checks a specialist can run on its own baseline source
*before* paying for a GPU job — task-agnostic helpers.

Only the truly generic tools live here:
  * `syntax_check` — py_compile against the baseline source
  * `param_count`  — AST walk for nn.Linear / nn.Embedding param totals

`size_project` is task-specific (knows the 16 MB cap, calls the task's
`pack_code`) and lives in the task package's own tools module.

Both tools take a workdir path and read the baseline filename from the
active task adapter (`current_adapter().baseline_filename`), so the same
helper works for `train_gpt.py` (PG), `train_airbench.py` (cifar), etc.
"""

from __future__ import annotations

import ast
import json
import py_compile
from pathlib import Path
from typing import Any

from . import tool


# ── Shared MCP wrapper ───────────────────────────────────────────────────────

def _mcp(result: dict[str, Any]) -> dict[str, Any]:
    """Wrap an impl result dict into the MCP content shape.

    We JSON-encode rather than pretty-print because the model is the
    consumer: a parseable text block is more useful than a human summary.
    `default=str` handles Paths / other non-JSON values defensively.
    """
    return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}


def _baseline_path(workdir: str) -> Path:
    """Resolve `<workdir>/<baseline filename>` via the active task adapter."""
    from agent_core import current_adapter
    return Path(workdir) / current_adapter().baseline_filename


# ── param_count ──────────────────────────────────────────────────────────────

# Static-AST heuristic: sum literal sizes in nn.Parameter / nn.Linear /
# nn.Embedding constructors. Not as accurate as instantiating the model,
# but (a) costs ~5ms, (b) avoids torch, (c) catches gross 10x mistakes in
# dim/depth, which is what the agent needs pre-submit. If the agent wants
# exact counts, param_count_exact via a subprocess + torch.no_grad is a
# future addition — not needed for the first iteration.


def _ast_numeric_value(node: ast.AST) -> int | None:
    """Literal int, or negated literal. Everything else → None (skip)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _ast_numeric_value(node.operand)
        return -v if v is not None else None
    return None


def _estimate_params_from_call(call: ast.Call) -> int:
    """Best-effort param estimate from an nn.Linear / nn.Embedding call."""
    fn_name = ""
    if isinstance(call.func, ast.Attribute):
        fn_name = call.func.attr
    elif isinstance(call.func, ast.Name):
        fn_name = call.func.id

    args = [_ast_numeric_value(a) for a in call.args]
    kwargs = {kw.arg: _ast_numeric_value(kw.value) for kw in call.keywords if kw.arg}

    def _pick(name_list: tuple[str, ...], pos_idx: int) -> int | None:
        for n in name_list:
            if kwargs.get(n) is not None:
                return kwargs[n]
        if pos_idx < len(args) and args[pos_idx] is not None:
            return args[pos_idx]
        return None

    if fn_name == "Linear":
        i = _pick(("in_features",), 0)
        o = _pick(("out_features",), 1)
        if i is not None and o is not None:
            bias = kwargs.get("bias")
            bias_n = o if bias is not False else 0
            return i * o + bias_n
    elif fn_name == "Embedding":
        n = _pick(("num_embeddings",), 0)
        d = _pick(("embedding_dim",), 1)
        if n is not None and d is not None:
            return n * d
    elif fn_name == "Conv2d":
        # nn.Conv2d(in_channels, out_channels, kernel_size, ...)
        # kernel_size may be int or tuple; handle the common int case
        ic = _pick(("in_channels",),  0)
        oc = _pick(("out_channels",), 1)
        k_node = call.args[2] if len(call.args) > 2 else None
        for kw in call.keywords:
            if kw.arg == "kernel_size":
                k_node = kw.value
        if isinstance(k_node, ast.Constant) and isinstance(k_node.value, int):
            kh = kw_ = k_node.value
            kh = kw_
        elif isinstance(k_node, ast.Tuple) and len(k_node.elts) == 2:
            kh = _ast_numeric_value(k_node.elts[0])
            kw_ = _ast_numeric_value(k_node.elts[1])
        else:
            kh = kw_ = None
        if ic is not None and oc is not None and kh is not None and kw_ is not None:
            bias = kwargs.get("bias")
            bias_n = oc if bias is not False else 0
            return ic * oc * kh * kw_ + bias_n
    return 0


def _param_count_impl(workdir: str) -> dict[str, Any]:
    """Sum nn.Linear / nn.Embedding / nn.Conv2d params via AST walk.

    Single-file tasks: walks just baseline_filename. Multi-file tasks
    (NC v2-B): walks every .py under the editable tree, aggregates per-file
    + grand total. Returns {ok, total_params, by_kind, by_file?, line_hits, note}.
    """
    files = _editable_py_files(workdir)
    if not files:
        from agent_core import current_adapter
        baseline = current_adapter().baseline_filename
        return {"ok": False,
                "error": f"{baseline} not found in {workdir} (and no editable_tree)"}

    by_kind: dict[str, int] = {"Linear": 0, "Embedding": 0, "Conv2d": 0}
    by_file: dict[str, int] = {}
    line_hits: list[dict[str, Any]] = []
    multi = len(files) > 1
    wd = Path(workdir).resolve()

    for f in files:
        try:
            source = f.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError) as e:
            # Skip un-parseable file but flag it; don't fail the whole walk.
            try:
                rel = str(f.resolve().relative_to(wd))
            except ValueError:
                rel = str(f)
            by_file[rel] = -1     # sentinel: "parse failed"
            continue

        try:
            rel = str(f.resolve().relative_to(wd))
        except ValueError:
            rel = str(f)
        file_total = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = ""
            if isinstance(node.func, ast.Attribute):
                fn = node.func.attr
            elif isinstance(node.func, ast.Name):
                fn = node.func.id
            if fn not in by_kind:
                continue
            n_params = _estimate_params_from_call(node)
            if n_params > 0:
                by_kind[fn] += n_params
                file_total += n_params
                hit: dict[str, Any] = {"lineno": node.lineno,
                                       "kind": fn, "params": n_params}
                if multi:
                    hit["file"] = rel
                line_hits.append(hit)
        by_file[rel] = file_total

    total = sum(by_kind.values())
    result: dict[str, Any] = {
        "ok":           True,
        "total_params": total,
        "by_kind":      by_kind,
        "line_hits":    line_hits[:40],           # truncate for context budget
        "note": (
            "static-AST estimate only; excludes LayerNorm/bias/scalar params "
            "and dynamic shape deps. For exact count run a trial."
        ),
    }
    # Multi-file mode: include per-file breakdown.
    if len(files) > 1:
        result["by_file"] = by_file
        result["files_walked"] = len(files)
    return result


@tool(
    "param_count",
    (
        "Static AST estimate of trainable parameter count in train_gpt.py. "
        "Sums nn.Linear and nn.Embedding literal sizes. Fast (~5 ms) but "
        "only catches gross structural changes — for exact counts, run a "
        "trial and read result.json. Returns {total_params, by_kind, "
        "line_hits} where line_hits lists the lines each count came from."
    ),
    {
        "type": "object",
        "properties": {
            "workdir": {"type": "string"},
        },
        "required": ["workdir"],
    },
)
async def param_count(args: dict[str, Any]) -> dict[str, Any]:
    return _mcp(_param_count_impl(args["workdir"]))


# ── syntax_check ─────────────────────────────────────────────────────────────

def _editable_py_files(workdir: str) -> list[Path]:
    """Return the list of .py files the agent may edit, in stable order.

    Single-file tasks (PG, CIFAR): just `<workdir>/<baseline_filename>`.
    Multi-file tasks (NC v2-B): baseline + every .py under
    `<workdir>/<editable_tree>/` (recursive walk, sorted).

    Used by syntax_check + param_count to walk the full editable surface.
    """
    from agent_core import current_adapter
    a = current_adapter()
    wd = Path(workdir)
    files: list[Path] = []
    base = wd / a.baseline_filename
    if base.is_file():
        files.append(base)
    tree = a.editable_tree
    if tree:
        tree_root = wd / tree
        if tree_root.is_dir():
            files.extend(sorted(p for p in tree_root.rglob("*.py") if p.is_file()))
    return files


def _syntax_check_impl(workdir: str) -> dict[str, Any]:
    """py_compile every editable .py; report the FIRST SyntaxError if any.

    For single-file tasks (PG, CIFAR), checks only the baseline and
    returns the legacy {ok, error, line, column} shape — byte-equal to
    the pre-multi-file core. For multi-file tasks (NC v2-B), walks the
    entire editable tree and adds `file` + `files_checked` fields.
    """
    files = _editable_py_files(workdir)
    multi = len(files) > 1
    if not files:
        from agent_core import current_adapter
        baseline = current_adapter().baseline_filename
        return {"ok": False,
                "error": f"{baseline} not found in {workdir}"}
    for f in files:
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as e:
            exc = getattr(e, "exc_value", None)
            wd = Path(workdir).resolve()
            try:
                rel = f.resolve().relative_to(wd)
                rel_str = str(rel)
            except ValueError:
                rel_str = str(f)
            if isinstance(exc, SyntaxError):
                out: dict[str, Any] = {
                    "ok":     False,
                    "error":  f"{type(exc).__name__}: {exc.msg}",
                    "line":   exc.lineno,
                    "column": exc.offset,
                    "text":   (exc.text or "").rstrip(),
                }
                if multi:
                    out["file"] = rel_str
                    out["files_checked"] = files.index(f) + 1
                return out
            out = {"ok": False, "error": str(e)}
            if multi:
                out["file"] = rel_str
                out["files_checked"] = files.index(f) + 1
            return out
    out = {"ok": True, "error": "", "line": None, "column": None}
    if multi:
        out["files_checked"] = len(files)
    return out


@tool(
    "syntax_check",
    (
        "Compile train_gpt.py with py_compile and report any SyntaxError "
        "without executing the module. Much faster than a GPU trial "
        "(millisecond-scale) and catches the most common edit mistake. "
        "Returns {ok, error, line, column} — on success, error is empty."
    ),
    {
        "type": "object",
        "properties": {
            "workdir": {"type": "string"},
        },
        "required": ["workdir"],
    },
)
async def syntax_check(args: dict[str, Any]) -> dict[str, Any]:
    return _mcp(_syntax_check_impl(args["workdir"]))
