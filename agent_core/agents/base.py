"""DoerBase — the SDK-facing skeleton every specialist subclasses.

Concurrency model
─────────────────
One iter = one SDK session = one DoerBase.run_once() call. The session
carries extended-thinking state through its tool-use turns, but DOES NOT
persist across iters — fresh blackboard read on every entry.

SDK integration
───────────────
`claude_agent_sdk.ClaudeSDKClient(options=ClaudeAgentOptions(...))` is the
canonical entrypoint. If the SDK isn't importable (local dev), run_once()
raises RuntimeError rather than silently returning — that's the signal to
run via the supervisor on the head node where the SDK is installed.

Tool binding
────────────
Custom tools (syntax_check, size_project, param_count, read_snapshot,
rebase_to, submit_trial) are registered through an in-process MCP server
built with `create_sdk_mcp_server(name="apg", ...)`. The SDK surfaces
them to the model under the namespace `mcp__apg__<tool_name>` — that's
what must appear in `allowed_tools`. Each decorated tool is an
`SdkMcpTool` instance carrying its JSONSchema via `.input_schema`, so
the SDK can auto-derive what the model sees.

WebSearch / WebFetch / Read / Edit / Bash are SDK built-ins — we enable
them by name in the `allowed_tools` list; no local binding needed. Bash
runs under the SDK's OS-level sandbox (bubblewrap on Linux) — reads are
unrestricted, writes are restricted to the specialist's workdir. Write
is deliberately absent: `pack_submission.py` would silently drop any
sidecar files the agent wrote next to train_gpt.py.

Streaming response
──────────────────
`await client.query(prompt)` returns None. We iterate
`client.receive_response()` to collect AssistantMessage / UserMessage /
ResultMessage events and pull out:
  * ToolUseBlocks the model emitted (count + match submit_trial's id)
  * ToolResultBlocks the SDK sent back for our tools (parse the MCP
    text payload to recover the row dict for submit_trial)
  * usage + session_id from the terminal ResultMessage
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _no_lineage_active() -> bool:
    """Single source of truth for the MAGENT_NO_LINEAGE env-var check.

    Set by `agent_core/supervisor/__main__.py:main()` when the
    operator passes `--no-lineage`. Read by render_user_message,
    DoerBase._allowed_tools, DoerBase._preload_tools, and the
    block_bash_blackboard hook.

    Boolean string "1" enables; anything else (unset, "0", "true", ...)
    disables. We deliberately use strict "1" to avoid surprises from
    arbitrary truthy values.
    """
    return os.environ.get("MAGENT_NO_LINEAGE", "0") == "1"

from ..harness import blackboard, config, events, tracker
from ..tools import (
    create_sdk_mcp_server, diff_snapshots, param_count, read_pr_library,
    read_pr_source, read_snapshot, rebase_to, syntax_check,
)
from ..tools.submit import _stage_workdir, submit_trial


def build_system_prompt(domain: str) -> str:
    """Resolve via the active task adapter (delegates to the task's prompts)."""
    from agent_core import current_adapter
    return current_adapter().build_system_prompt(domain)


def _adapter_tools_by_name() -> dict:
    """Map of `tool.name -> tool` for the active task adapter's bound tools.

    Strict: bind_tools MUST be implemented by every task adapter (PGTaskAdapter
    fills it; nc/cifar must too). No silent fallback to a specific package.
    """
    from agent_core import current_adapter
    return {t.name: t for t in current_adapter().bind_tools()}


# ── Context budget ───────────────────────────────────────────────────────────

_LEADERBOARD_MAX_BYTES = 4_000
_KNOWLEDGE_MAX_BYTES   = 8_000
_RECENT_N              = 10

# Rows with this status carry no experimental signal — the supervisor
# synthesised a `failed` verdict after the local sched CLI name cache lost
# the job entry, not because the trial itself crashed. Hide them from
# Recent Activity so specialists don't treat those hypothesis directions
# as explored-and-broken.
_QUARANTINED_STATUSES = frozenset({"harness_abort"})

# ── MCP namespace ────────────────────────────────────────────────────────────
# The string that lands in allowed_tools is built from this server name —
# the SDK namespaces every tool as `mcp__<server_name>__<tool_name>`.
_MCP_SERVER_NAME = "apg"


# ── Iteration record ─────────────────────────────────────────────────────────

@dataclass(slots=True)
class IterRecord:
    """Outcome of a single DoerBase.run_once() call."""
    specialist: str
    iter_start: str                 # ISO-8601 UTC
    iter_end:   str
    session_id: Optional[str]       # SDK-assigned, if available
    final_row:  Optional[dict]      # the TSV row written by submit_trial
    tool_calls: int                 # how many tools the agent invoked
    tool_trace: list[str]           # ordered tool names called (short-form)
    error:      Optional[str]       # surface string on exception, else None
    usage:      Optional[dict]      # {input_tokens, output_tokens, cache_*}


# ── Context rendering ────────────────────────────────────────────────────────

def _truncate(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n… (truncated: {label} exceeded {limit} bytes)\n"


def _read_md_safely(path: Path, limit: int, label: str) -> str:
    if not path.exists():
        return f"*(no {label} yet — this is the first session.)*"
    try:
        return _truncate(path.read_text(encoding="utf-8"), limit, label)
    except OSError as e:
        return f"*(failed to read {label}: {e})*"


def _render_recent_activity(rows: list[dict], n: int = _RECENT_N) -> str:
    """Compact per-row line: `exp_NNN [spec, status, bpb=..., Δ=...] hyp`.

    Same shape the tree in KNOWLEDGE.md uses, but flat + most-recent-first
    so the agent sees what just happened without walking the whole tree.
    """
    filtered = [r for r in rows if r.get("status") not in _QUARANTINED_STATUSES]
    recent = list(reversed(filtered))[:n]
    if not recent:
        return "*(no trials yet)*"
    from agent_core import current_adapter
    adapter = current_adapter()
    score_field = adapter.score_field
    short = adapter.score_short_label
    lines = []
    for r in recent:
        bpb = r.get(score_field, "") or "—"
        delta = r.get("delta_vs_best", "") or "—"
        notes = (r.get("notes", "") or "").strip()
        if len(notes) > 160:
            notes = notes[:157] + "…"
        hyp = (r.get("hypothesis", "") or "")[:140]
        lines.append(
            f"- exp_{r.get('exp_id','?')} "
            f"[{r.get('specialist','?')}, {r.get('status','?')}, "
            f"{short}={bpb}, Δ={delta}] {hyp}"
            + (f"\n    └─ {notes}" if notes else "")
        )
    return "\n".join(lines)


# Saturation threshold: |Δ_val_bpb| considered "sub-noise-floor" for the
# purpose of flagging stuck hill-climbing. Half the conventional keep bar
# (0.001 nat) — so if every recent trial in this specialist's domain
# moved ≤ 0.0005, that's weak signal that small tweaks aren't buying
# anything and the agent should pivot to something structural.
_SATURATION_EPSILON = 0.0005
_SATURATION_WINDOW = 5


def _render_saturation_warning(rows: list[dict], specialist: str) -> str:
    """Soft nudge when the specialist's recent trials all produced ≤ epsilon
    deltas — suggests the domain has saturated in its current local minimum.

    Returns an empty string when no nudge is warranted (too few trials,
    or at least one recent trial broke threshold). Otherwise emits a
    single-paragraph soft warning. The warning is advisory — agents can
    override with a specific planned direction.
    """
    own = [
        r for r in rows
        if r.get("specialist") == specialist
        and r.get("status") not in _QUARANTINED_STATUSES
    ]
    if len(own) < _SATURATION_WINDOW:
        return ""  # not enough trials yet to judge saturation

    recent_own = list(reversed(own))[:_SATURATION_WINDOW]
    sub_threshold = 0
    for r in recent_own:
        delta_raw = (r.get("delta_vs_best") or "").strip()
        try:
            if abs(float(delta_raw)) < _SATURATION_EPSILON:
                sub_threshold += 1
        except (TypeError, ValueError):
            # Empty delta (e.g. crash, size_blocked) counts as sub-threshold
            # — no improvement signal means we're stuck either way.
            sub_threshold += 1

    if sub_threshold < _SATURATION_WINDOW:
        return ""  # at least one recent trial showed real movement

    from agent_core import current_adapter
    score_field = current_adapter().score_field
    return (
        f"## ⚠ Saturation signal\n"
        f"Your last {_SATURATION_WINDOW} trials in the **{specialist}** "
        f"domain all produced |Δ_{score_field}| < {_SATURATION_EPSILON} or "
        f"non-VALID status — sub-noise-floor at our measurement precision. "
        f"The current stack has likely absorbed the obvious tweaks on your "
        f"usual axes. Consider proposing something STRUCTURALLY different "
        f"this iteration (see Anti-anchoring and the 'Edit radius' line in "
        f"your preamble) — for example: pull an untried mechanism from "
        f"`read_pr_library` / `gaps.md`, or rebase to a different "
        f"snapshot via `rebase_to`. This is a soft nudge; skip it if you "
        f"have a specific pre-planned direction from a prior lineage "
        f"entry."
    )


def render_user_message(specialist: str) -> str:
    """Assemble the per-iteration user message from live blackboard state.

    The agent reads this on every session start. Ordering is deliberate:
    LEADERBOARD first (answers "what beats what"), KNOWLEDGE second
    (answers "what's been tried"), Recent Activity third (answers
    "what just happened").

    When `MAGENT_NO_LINEAGE=1` is set (no-lineage ablation), all
    cross-session lineage feedback is removed: LEADERBOARD, KNOWLEDGE,
    Recent Activity, and Saturation sections are blanked. Only the
    "current best" exp_id + score line is exposed (this is allowed —
    the agent needs to know which snapshot to rebase from), plus the
    workdir state and task instruction. The agent must propose only
    from static priors (system prompt: INIT/SOTA_STACK/LESSONS) and
    its own in-session memory.
    """
    from agent_core import current_adapter
    score_field = current_adapter().score_field
    best = blackboard.read_best() or {}
    best_exp = best.get("exp_id", "none")
    best_bpb = best.get(score_field, "n/a")

    workdir = config.workdir_for(specialist)
    wd_state = _render_workdir_state(workdir)

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if _no_lineage_active():
        # ABLATION MODE — no cross-session lineage. Only exposes:
        #   - current best exp_id + score (needed for rebase_to)
        #   - workdir state (own files)
        #   - task instruction + explicit "no blackboard reads via Bash" rule
        # block_bash_blackboard PreToolUse hook (agents/hooks.py) enforces
        # the Bash rule at the SDK level; the prompt rule below is the
        # parallel soft signal so the agent doesn't waste turns trying.
        return (
            f"# Session start — {now}\n"
            f"You are specialist **{specialist}**. Your workdir is `{workdir}`.\n"
            f"Current best: **exp_{best_exp}** ({score_field}={best_bpb}).\n"
            f"\n"
            f"## Your workdir\n{wd_state}\n"
            f"\n"
            f"## Your task this session\n"
            f"Propose an edit within the **{specialist}** domain, validate "
            f"it locally, and submit via submit_trial. A single submit is "
            f"a complete session — stopping there is the default. You MAY "
            f"submit again only if the returned row points to a concrete "
            f"next edit worth trying; a crash, an uninformative result, or "
            f"the absence of a clear refinement means stop. Don't force "
            f"extra submits.\n"
            f"\n"
            f"## Lineage policy (this run)\n"
            f"Prior-trial logs are unavailable for this run. The "
            f"LEADERBOARD / KNOWLEDGE / Recent Activity sections you may "
            f"have seen in other runs are intentionally absent. The "
            f"`read_snapshot` and `diff_snapshots` tools are also disabled. "
            f"**Do NOT attempt to query blackboard files via Bash** "
            f"(tree.tsv, results.tsv, lineage_snapshots/, events.jsonl, "
            f"best.json, supervisor_audit.jsonl, anything under "
            f"blackboard/) — those reads are rejected at the harness "
            f"level. Propose from your static priors (INIT.md / "
            f"SOTA_STACK.md / LESSONS.md in the system prompt), the "
            f"current-best score above, your workdir state, and your "
            f"own in-session reasoning.\n"
        )

    leaderboard = _read_md_safely(
        config.LEADERBOARD_MD, _LEADERBOARD_MAX_BYTES, "LEADERBOARD.md",
    )
    knowledge = _read_md_safely(
        config.KNOWLEDGE_MD, _KNOWLEDGE_MAX_BYTES, "KNOWLEDGE.md",
    )
    rows = tracker.read_results()
    recent = _render_recent_activity(rows)
    saturation = _render_saturation_warning(rows, specialist)

    # Saturation warning (when present) is placed directly AFTER Recent
    # Activity so the agent reads it with the evidence fresh — the
    # Recent Activity rows are the supporting data for the nudge.
    saturation_block = f"\n{saturation}\n" if saturation else ""

    return (
        f"# Session start — {now}\n"
        f"You are specialist **{specialist}**. Your workdir is `{workdir}`.\n"
        f"Current best: **exp_{best_exp}** ({score_field}={best_bpb}).\n"
        f"\n"
        f"## LEADERBOARD.md\n{leaderboard}\n"
        f"\n"
        f"## KNOWLEDGE.md\n{knowledge}\n"
        f"\n"
        f"## Recent Activity (most recent {_RECENT_N})\n{recent}\n"
        f"{saturation_block}"
        f"\n"
        f"## Your workdir\n{wd_state}\n"
        f"\n"
        f"## Your task this session\n"
        f"Propose an edit within the **{specialist}** domain, validate it "
        f"locally, and submit via submit_trial. A single submit is a "
        f"complete session — stopping there is the default. You MAY submit "
        f"again only if the returned row points to a concrete next edit "
        f"worth trying; a crash, an uninformative result, or the absence of "
        f"a clear refinement means stop. Don't force extra submits.\n"
    )


def _render_workdir_state(workdir: Path) -> str:
    """Terse snapshot of the workdir so the agent doesn't have to ls."""
    if not workdir.exists():
        return "*(workdir does not yet exist — submit_trial will create it)*"
    from agent_core import current_adapter
    baseline = current_adapter().baseline_filename
    code = workdir / baseline
    if not code.is_file():
        return (
            f"Workdir exists but has no {baseline}. "
            f"Your first step should be rebase_to(best_exp, '{workdir}')."
        )
    size = code.stat().st_size
    # One-line head so the agent can spot stale placeholders without reading.
    try:
        head = code.read_text(encoding="utf-8", errors="replace").splitlines()[:2]
    except OSError:
        head = []
    head_str = " | ".join(h.strip() for h in head)[:200]
    return (
        f"- {baseline} present, {size:,} bytes\n"
        f"  head: `{head_str}`"
    )


# ── Doer base class ──────────────────────────────────────────────────────────

@dataclass(slots=True)
class DoerConfig:
    """Per-iteration knobs the supervisor may override.

    `model` defaults to `None` and is resolved from
    `config.model_for(specialist)` in `__post_init__` — i.e. the source of
    truth is `multi_agent/swarm_config.json` (default + per-spec overrides),
    not a hardcoded model id. Pass `model="claude-..."` explicitly to
    short-circuit the lookup (used by ad-hoc dev runs / verify_candidate /
    one-off A/B tests).
    """
    specialist:            str
    thinking_budget:       int = config.DOER_THINKING_BUDGET_TOKENS
    max_turns:             int = 200        # cap on tool-use turns — sized for
                                            # multi-submit sessions with PR-library
                                            # source drill-down (prior: 100, then 50)
    model:                 Optional[str] = None
    enable_web:            bool = True

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = config.model_for(self.specialist)


# Built-in SDK tools we want PRELOADED into the agent's palette (via the
# `tools=...` ClaudeAgentOptions field → CLI `--tools <list>`). The CLI's
# `--tools <list>` REPLACES the claude_code default preset entirely, so we
# enumerate every built-in the agent legitimately uses; anything not in
# this list becomes deferred (visible only after a ToolSearch call).
#
# Why this exists: WebSearch / WebFetch are deferred in the claude_code
# default preset. Agents in our swarm never proactively ToolSearch them
# (they only ToolSearch the `mcp__apg__*` custom MCP tools), so over 89K
# iters we observed ZERO WebSearch / WebFetch calls despite enabling them
# in `allowed_tools` and explicitly encouraging in the system prompt
# (events.jsonl audit, 2026-04-25). Forcing preload makes the schema
# always available without the agent needing to know about ToolSearch.
#
# Selected from empirical agent usage on this swarm:
#   Read 4853, Bash 4708, Grep 3470, Edit 1336, ToolSearch 1147, Glob 329,
#   TodoWrite 46, Agent 3 — all kept. Web added explicitly.
_PRELOAD_BUILTIN_TOOLS = (
    "Read", "Edit", "Bash", "Grep", "Glob",
    "ToolSearch", "TodoWrite", "Agent",
    "WebSearch", "WebFetch",
)


# Every @tool's name, in the order we pass them to create_sdk_mcp_server.
# Kept here so `_allowed_tools()` stays in sync with `_bind_tools()` without
# reflecting on the @tool-decorated objects (the real SDK returns
# SdkMcpTool instances whose `.name` attribute could be accessed; the shim
# returns _ShimTool with the same attr — but making the coupling explicit
# here is clearer than attribute-poking and keeps the allowlist auditable).
_CUSTOM_TOOL_NAMES = (
    "syntax_check", "size_project", "param_count",
    "read_snapshot", "rebase_to", "diff_snapshots",
    "submit_trial", "read_pr_library", "read_pr_source",
)


class DoerBase:
    """One-shot SDK session for a single specialist.

    Subclass + override `specialist` to make a concrete doer. The default
    implementation is already usable — subclasses mostly exist so we have
    a hook for per-domain custom checks later (e.g. arch might run a
    param-budget sanity check before submit).
    """
    specialist: str = ""           # must be set by subclasses

    def __init__(self, cfg: Optional[DoerConfig] = None) -> None:
        if not self.specialist:
            raise TypeError(
                f"{type(self).__name__} must set class attr `specialist`"
            )
        from agent_core import current_adapter
        if self.specialist not in current_adapter().all_domains:
            raise ValueError(f"unknown specialist {self.specialist!r}")
        self.cfg = cfg or DoerConfig(specialist=self.specialist)
        # Make sure the workdir exists AND has a baseline train_gpt.py before
        # the SDK session starts. Without staging here, the very first iter
        # gives the agent an empty workdir: Read fails, Edit fails, the
        # agent spins until max_turns without ever reaching submit_trial.
        # _stage_workdir is idempotent — on subsequent iters it leaves the
        # agent's edited train_gpt.py alone and only refreshes helper scripts.
        self.workdir = config.workdir_for(self.specialist)
        from agent_core import current_adapter
        _stage_workdir(self.workdir, current_adapter().pkg_root)

    # ── SDK wiring ───────────────────────────────────────────────────────────

    def _bind_tools(self) -> list:
        """The @tool-decorated callables registered into the MCP server.

        Order is authoritative — comes from `current_adapter().bind_tools()`.
        Each task adapter is responsible for returning the tools in the
        order it wants registered (which determines events.jsonl ordering).
        """
        from agent_core import current_adapter
        return current_adapter().bind_tools()

    def _preload_tools(self) -> list[str]:
        """Tools to PRELOAD into the SDK CLI palette via `--tools <list>`.

        Mirror of `_allowed_tools()` but force-includes WebSearch / WebFetch
        so their schema is always loaded (vs the claude_code default preset
        which keeps them deferred). Without this, agents would need to
        proactively ToolSearch web tools; empirical 89K-iter audit showed
        they never do — observed 0 web calls Apr 25 2026 → forced preload.

        Returns the full explicit list (custom MCP + built-ins) since
        `--tools <list>` REPLACES the default preset, not extends it.

        Honors `enable_web=False`: when web is disabled at config layer,
        we exclude WebSearch / WebFetch from preload too (otherwise
        they'd appear in the palette but be blocked at allowed_tools,
        producing confusing permission-denied errors).

        Honors `MAGENT_NO_LINEAGE=1`: removes `read_snapshot` and
        `diff_snapshots` from the custom-tool list (they read prior
        trials' submitted code and are a lineage channel). `rebase_to`
        is kept since it's needed to start the workdir from the current
        best (current-best exp_id is the one piece of lineage info the
        ablation explicitly allows).
        """
        from agent_core import current_adapter
        custom_names = list(current_adapter().custom_tool_names)
        if _no_lineage_active():
            custom_names = [n for n in custom_names
                            if n not in ("read_snapshot", "diff_snapshots")]
        custom = [f"mcp__{_MCP_SERVER_NAME}__{n}" for n in custom_names]
        builtins = [t for t in _PRELOAD_BUILTIN_TOOLS
                    if (t not in ("WebSearch", "WebFetch") or self.cfg.enable_web)]
        return custom + builtins

    def _allowed_tools(self) -> list[str]:
        """Allowlist passed to ClaudeAgentOptions.

        Custom tools are namespaced `mcp__<server>__<tool>` — matches the
        SDK's MCP routing. Read + Edit are the SDK's file-editing primitives,
        scoped to `cwd` (set on ClaudeAgentOptions below) so the agent can
        only see / mutate files under its own workdir_<domain>/.

        Bash is enabled and runs under the SDK's OS-level sandbox
        (bubblewrap on Linux; see `sandbox=` in `run_once` below) — reads
        are unrestricted, writes are confined to cwd. That is how the
        agent can slice tree.tsv / results.tsv from the blackboard dir
        without being able to mutate them. Write is still absent: it
        would let the agent create sidecar files that `pack_submission.py`
        ignores, silently inflating the artifact.

        Honors `MAGENT_NO_LINEAGE=1`: see `_preload_tools` for rationale.
        Bash itself is kept (the agent needs it for own-workdir slicing
        and `wc` / `awk` on its own train_gpt.py); the
        `block_bash_blackboard` PreToolUse hook (agents/hooks.py) is what
        actually gates Bash reads of blackboard files.
        """
        from agent_core import current_adapter
        custom_names = list(current_adapter().custom_tool_names)
        if _no_lineage_active():
            custom_names = [n for n in custom_names
                            if n not in ("read_snapshot", "diff_snapshots")]
        custom = [f"mcp__{_MCP_SERVER_NAME}__{n}" for n in custom_names]
        builtins = ["Read", "Edit", "Bash"]
        if self.cfg.enable_web:
            builtins += ["WebSearch", "WebFetch"]
        return custom + builtins

    def _system_prompt(self) -> str:
        return build_system_prompt(self.specialist)

    def _user_message(self) -> str:
        return render_user_message(self.specialist)

    # ── Main entrypoint ──────────────────────────────────────────────────────

    async def run_once(self) -> IterRecord:
        """Execute one SDK session. Returns an IterRecord on completion.

        Blocks from session start to terminal ResultMessage. Exceptions from
        the SDK (rate limit, network) are caught and surfaced in
        `IterRecord.error` so the supervisor can decide to retry rather
        than crash.
        """
        try:
            from claude_agent_sdk import (  # type: ignore[import-not-found]
                AssistantMessage,
                ClaudeAgentOptions,
                ClaudeSDKClient,
                HookMatcher,
                ResultMessage,
                ToolResultBlock,
                ToolUseBlock,
                UserMessage,
            )
            from .hooks import (
                block_bash_blackboard, block_bash_writes,
                cap_builtin_tool_output,
            )
        except ImportError as e:
            raise RuntimeError(
                "claude_agent_sdk not installed — this method only runs on the "
                f"head node. Underlying error: {e}"
            ) from e

        rec = IterRecord(
            specialist=self.specialist,
            iter_start=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            iter_end="",
            session_id=None,
            final_row=None,
            tool_calls=0,
            tool_trace=[],
            error=None,
            usage=None,
        )

        # Sandbox decision via `config.should_disable_sandbox()`:
        #   - MAGENT_DISABLE_SANDBOX=1 explicit → off
        #   - MAGENT_DISABLE_SANDBOX=0 explicit → on
        #   - unset → auto-probe bwrap pivot_root+proc-mount (cached, runs
        #     once per supervisor process). LXC / nested containers with
        #     nested-userns restrictions auto-disable.
        # session_start event records the resolved value for paper trail.
        sandbox_enabled = not config.should_disable_sandbox()

        # Emit the earliest visibility marker: SDK session is about to start
        # thinking. This lands in events.jsonl BEFORE any tool is called, so
        # the dashboard stops showing "(no events yet)" for a specialist that
        # is genuinely in its read/grep/edit phase.
        events.emit(
            self.specialist, "session_start",
            model=self.cfg.model,
            max_turns=self.cfg.max_turns,
            thinking_budget=self.cfg.thinking_budget,
            sandbox=sandbox_enabled,
        )

        try:
            # Register our 6 custom tools as an in-process MCP server. Name
            # must match `_MCP_SERVER_NAME` so the allowed_tools prefixes line
            # up with the SDK's routing.
            server = create_sdk_mcp_server(
                name=_MCP_SERVER_NAME,
                version="1.0.0",
                tools=self._bind_tools(),
            )

            # cwd pins the SDK's Read/Edit primitives to the specialist's
            # workdir — the agent can inspect and mutate train_gpt.py but
            # cannot reach the package source, blackboard, or other
            # specialists' workdirs. permission_mode=bypassPermissions is
            # the correct mode for an autonomous run (no human in the loop
            # to approve each tool call); the allowlist itself remains the
            # real safety boundary.
            #
            # sandbox={"enabled": True} hands Bash to bubblewrap: reads
            # are unrestricted (so the agent can `awk '…' tree.tsv` from
            # the blackboard) but writes are confined to cwd. This
            # replaces an earlier in-process command-allowlist validator
            # — the OS-level sandbox is both stricter (no escape via
            # shell parsing edge cases) and more permissive (any
            # read-only command works, not just the whitelisted ones).
            # PostToolUse hook caps Bash/Grep output at 16 KB per call.
            # Log analysis (2026-04-25) showed cache_creation ≈ 161K tokens
            # per active iter, with Bash + Grep being the #2 / #3 most-used
            # tools — uncapped outputs were the dominant source of new
            # cache content per turn. The cap is generous (16 KB) so
            # legit slicing still works; the truncation marker tells the
            # agent how to recover. See agents/hooks.py for the rationale.
            #
            # `sandbox_enabled` was computed before the session_start emit
            # so the audit trail records which mode was live for this iter.
            # ccbox-direct deployments leave it True (default); LXC / nested-
            # container hosts set MAGENT_DISABLE_SANDBOX=1 because bwrap's
            # pivot_root + nested-userns proc-mount is blocked there.
            # When disabled, the SDK runs Bash unsandboxed and relies on the
            # OUTER container's namespace isolation.
            options = ClaudeAgentOptions(
                system_prompt=self._system_prompt(),
                model=self.cfg.model,
                cwd=str(self.workdir),
                allowed_tools=self._allowed_tools(),
                # Explicit preload list → CLI's --tools <list> REPLACES
                # the claude_code default preset and forces WebSearch /
                # WebFetch into the agent's initial palette (otherwise
                # they're deferred and never get used; see _PRELOAD_BUILTIN_TOOLS).
                tools=self._preload_tools(),
                mcp_servers={_MCP_SERVER_NAME: server},
                thinking={"type": "enabled", "budget_tokens": self.cfg.thinking_budget},
                max_turns=self.cfg.max_turns,
                permission_mode="bypassPermissions",
                sandbox={"enabled": sandbox_enabled, "autoAllowBashIfSandboxed": True},
                hooks={
                    # PreToolUse: deny destructive / write Bash commands.
                    # Hard line: Bash is read-only in this swarm. Primary
                    # cross-agent corruption defense on LXC (sandbox=False)
                    # hosts; defense-in-depth even with bwrap on.
                    #
                    # `block_bash_blackboard` is the no-lineage ablation
                    # gate (only fires when MAGENT_NO_LINEAGE=1; otherwise
                    # passes through). Listed alongside `block_bash_writes`
                    # so both run on every Bash call; the SDK invokes hooks
                    # in list order and either's `decision: block` is
                    # terminal.
                    "PreToolUse": [
                        HookMatcher(matcher="Bash", hooks=[
                            block_bash_writes,
                            block_bash_blackboard,
                        ]),
                    ],
                    # PostToolUse: cap oversized outputs to bound cache.
                    "PostToolUse": [
                        HookMatcher(matcher="Bash",      hooks=[cap_builtin_tool_output]),
                        HookMatcher(matcher="Grep",      hooks=[cap_builtin_tool_output]),
                        HookMatcher(matcher="WebFetch",  hooks=[cap_builtin_tool_output]),
                        HookMatcher(matcher="WebSearch", hooks=[cap_builtin_tool_output]),
                    ],
                },
            )

            # Expected submit_trial tool name, as the SDK surfaces it to the
            # model and stamps it onto ToolUseBlock.name / ToolResultBlock
            # (via tool_use_id matching).
            submit_tool_full_name = f"mcp__{_MCP_SERVER_NAME}__submit_trial"

            tool_calls = 0
            tool_trace: list[str] = []
            # tool_use_id → short tool name. Used to name the matching
            # tool_result event so paper analysis can correlate call → result.
            tool_id_to_name: dict[str, str] = {}

            # Capture the rendered user message before the SDK starts so we
            # can persist it post-session (paper-analysis: "what context did
            # the agent see this iter?"). Stored in a local var; the actual
            # disk write happens at iter_end below, off the SDK loop.
            user_msg = self._user_message()
            final_row: Optional[dict] = None
            session_id: Optional[str] = None
            latest_usage: Optional[dict] = None
            submit_tool_use_ids: set[str] = set()
            mcp_prefix = f"mcp__{_MCP_SERVER_NAME}__"

            async with ClaudeSDKClient(options=options) as client:
                await client.query(user_msg)

                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        if getattr(msg, "session_id", None):
                            session_id = msg.session_id
                        if getattr(msg, "usage", None):
                            latest_usage = dict(msg.usage)
                        for blk in msg.content or []:
                            if isinstance(blk, ToolUseBlock):
                                tool_calls += 1
                                short = blk.name[len(mcp_prefix):] if blk.name.startswith(mcp_prefix) else blk.name
                                tool_trace.append(short)
                                tool_id_to_name[blk.id] = short
                                # Truncated args digest (≤256 B) — enough to know
                                # which file was Read, which pattern was Grep'd,
                                # which PR was read_pr_library'd, etc., without
                                # exploding events.jsonl on big tool inputs.
                                try:
                                    args_digest = json.dumps(
                                        blk.input, ensure_ascii=False, default=str,
                                    )[:256]
                                except Exception:
                                    args_digest = ""
                                events.emit(
                                    self.specialist, "tool_called",
                                    tool=short, turn=tool_calls,
                                    tool_use_id=blk.id,
                                    args=args_digest or None,
                                )
                                if blk.name == submit_tool_full_name:
                                    submit_tool_use_ids.add(blk.id)
                    elif isinstance(msg, UserMessage):
                        # Tool results come back in the next UserMessage's
                        # content list as ToolResultBlock. We only care
                        # about the result for submit_trial — that's the
                        # TSV row we need to surface to the supervisor.
                        content = msg.content
                        if isinstance(content, list):
                            for blk in content:
                                if not isinstance(blk, ToolResultBlock):
                                    continue
                                # Per-result event for paper-analysis: outcome
                                # (ok/error) + size + first 256 B digest.
                                _emit_tool_result_event(
                                    self.specialist, blk,
                                    tool_id_to_name.get(blk.tool_use_id, "?"),
                                )
                                if blk.tool_use_id in submit_tool_use_ids:
                                    parsed = _parse_tool_result_content(blk.content)
                                    if parsed is not None:
                                        final_row = parsed
                    elif isinstance(msg, ResultMessage):
                        if getattr(msg, "session_id", None):
                            session_id = msg.session_id
                        if getattr(msg, "usage", None):
                            latest_usage = dict(msg.usage)

            rec.session_id = session_id
            rec.tool_calls = tool_calls
            rec.tool_trace = tool_trace
            rec.final_row = final_row
            rec.usage = _simplify_usage(latest_usage)
        except Exception as e:  # noqa: BLE001 — we genuinely want to catch anything
            rec.error = f"{type(e).__name__}: {e}"
            events.emit(
                self.specialist, "session_error",
                err=f"{type(e).__name__}: {str(e)[:120]}",
            )

        rec.iter_end = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Persist the rendered user-message context the agent saw this iter
        # (paper-analysis: "what lineage / leaderboard / KNOWLEDGE was visible
        # when the decision was made?"). Best-effort; never blocks iter_end.
        # Always lands on ccbox-local ext4 (BLACKBOARD_DIR), never on SharedFS.
        if session_id and user_msg:
            try:
                ldir = config.BLACKBOARD_DIR / "lineage_snapshots"
                ldir.mkdir(parents=True, exist_ok=True)
                (ldir / f"{session_id}.txt").write_text(user_msg, encoding="utf-8")
            except OSError:
                pass

        # Aggregate per-tool usage from the trace (preserves insertion order
        # for deterministic wire format). Dashboard ignores unknown fields, so
        # adding this dict is strictly additive and backward-compatible.
        tool_calls_by_name: dict[str, int] = {}
        for t in (rec.tool_trace or []):
            tool_calls_by_name[t] = tool_calls_by_name.get(t, 0) + 1
        events.emit(
            self.specialist, "session_end",
            tool_calls=rec.tool_calls,
            tool_calls_by_name=tool_calls_by_name or None,
            submit_captured=rec.final_row is not None,
            error=bool(rec.error),
        )
        return rec


# ── Session-result helpers ──────────────────────────────────────────────────

def _emit_tool_result_event(specialist: str, blk: Any, tool_name: str) -> None:
    """Best-effort: emit a per-tool-result event for paper-analysis.

    Captures size, error flag, and a 256-B content digest so trajectory
    analysis can answer "did syntax_check pass? what was the val_bpb?
    did Bash error?" without storing full payloads. Never raises —
    a malformed block just gets skipped.
    """
    try:
        if isinstance(blk.content, str):
            rsize = len(blk.content.encode("utf-8", "ignore"))
            digest = blk.content[:256]
        elif isinstance(blk.content, list):
            rsize = 0
            first_text = ""
            for c in blk.content:
                if not isinstance(c, dict):
                    continue
                t = c.get("text") or ""
                if isinstance(t, str):
                    rsize += len(t.encode("utf-8", "ignore"))
                    if not first_text:
                        first_text = t
            digest = first_text[:256]
        else:
            rsize = 0
            digest = ""
        events.emit(
            specialist, "tool_result",
            tool=tool_name,
            tool_use_id=getattr(blk, "tool_use_id", "") or None,
            result_bytes=rsize,
            is_error=bool(getattr(blk, "is_error", False)) or None,
            digest=digest or None,
        )
    except Exception:
        pass  # never let logging break the session


def _parse_tool_result_content(content: Any) -> Optional[dict]:
    """Recover the dict payload from an MCP ToolResultBlock.content.

    Our `@tool` wrappers wrap each impl's dict result as:
        {"content": [{"type": "text", "text": json.dumps(row)}]}
    The SDK surfaces the `.content` portion as either a plain string (some
    transports) or a list of content-block dicts (spec-compliant). Handle
    both shapes; if nothing parseable is found return None.
    """
    if content is None:
        return None
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(content, list):
        for c in content:
            if not isinstance(c, dict):
                continue
            text = c.get("text")
            if c.get("type") == "text" and isinstance(text, str):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    continue
    return None


def _simplify_usage(usage: Optional[dict]) -> Optional[dict]:
    """Normalise an Anthropic usage dict down to the fields we log."""
    if usage is None:
        return None
    return {
        "input_tokens":                usage.get("input_tokens"),
        "output_tokens":               usage.get("output_tokens"),
        "cache_read_input_tokens":     usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
    }
