"""SDK session hooks for doers.

Three hooks:
1. **PreToolUse** `block_bash_writes` — deny destructive / file-write Bash
   commands. The swarm shares one filesystem (blackboard, source tree,
   sibling workdirs); on LXC hosts where bwrap sandbox is disabled
   (`MAGENT_DISABLE_SANDBOX=1`), this hook is the primary defense
   against accidental cross-agent corruption. Read-only commands
   (awk / head / tail / grep / cat / find / ls / wc) pass through.
2. **PreToolUse** `block_bash_blackboard` — deny Bash reads of blackboard
   files (tree.tsv / results.tsv / lineage_snapshots/ / etc.) when the
   `MAGENT_NO_LINEAGE=1` env var is set. Used by the no-lineage ablation
   to close the Bash-on-blackboard back-channel. Otherwise pass-through.
3. **PostToolUse** `cap_builtin_tool_output` — cap oversized outputs
   (Bash / Grep / WebFetch / WebSearch) at 16 KB so cache_creation
   stays bounded.

Both hooks are defensive (`try/except → pass-through`) — never raise.
A buggy hook never kills a session; worst case it allows a single
suspicious call through that the model still has to argue for via
cwd-scoped Edit later.

Original PostToolUse motivation:
caps oversized SDK built-in tool outputs (Bash,
Grep, WebFetch, WebSearch) before they enter the model's context.
Motivation: log analysis showed cache_creation ~161K tokens / active
iter, with Bash + Grep being the #2 / #3 most-used tools — uncapped
outputs from `cat large_file`, `ls -R`, `grep -rn`, web pages etc. are
the dominant new-content source per turn. WebFetch / WebSearch are
covered too because (a) we now actively encourage web research, and
(b) web pages routinely return 30-100 KB of mostly-boilerplate text.

We deliberately keep the cap GENEROUS (16 KB) so legitimate uses
(slicing tree.tsv, reading a stack trace, scoped grep, an arxiv
abstract) keep working. The truncation marker tells the agent how
to recover (rerun with narrower scope, or switch to Read/Glob with
offset/limit) — so the loss of a tail in pathological cases costs
at most one extra turn.

Hook signature (per claude_agent_sdk.types):
    async def hook(input_data, tool_use_id, context) -> dict

Return shape: SyncHookJSONOutput with hookSpecificOutput.updatedMCPToolOutput
to overwrite the tool response, or `{}` to pass through unchanged.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

# 16 KB byte cap — generous enough that real grep / head / tail / cat
# results pass through, but anchors a hard ceiling against `ls -R`
# / `cat huge_log` / `grep -rn` storms. Anything beyond this is almost
# certainly waste from the agent's reasoning POV.
_CAP_BYTES = 16 * 1024
_TRUNC_TAIL = (
    "\n\n... [output truncated by harness at {n} bytes "
    "to protect cache budget — rerun with a narrower pattern, "
    "or switch to Read/Glob with offset/limit if you need more]"
)


def _truncate_str(s: str) -> tuple[str, bool]:
    """Return (possibly-truncated string, was_truncated). Cuts at the
    last newline before the cap when possible to keep the tail readable."""
    enc = s.encode("utf-8")
    if len(enc) <= _CAP_BYTES:
        return s, False
    cut = enc[: _CAP_BYTES - 256]  # leave room for the tail marker
    # Decode forgivingly (a multibyte char may straddle the cut)
    truncated = cut.decode("utf-8", "ignore")
    # Prefer to end on a newline boundary so the agent doesn't get a
    # half-line of mojibake right before the marker.
    nl = truncated.rfind("\n")
    if nl > 0 and len(truncated) - nl < 1024:  # nl is "near" the end
        truncated = truncated[: nl + 1]
    return truncated + _TRUNC_TAIL.format(n=_CAP_BYTES), True


def _cap_tool_response(resp: Any) -> tuple[Any, bool]:
    """Walk a tool_response value, cap any oversized string fields.

    Bash typically returns a string or {'stdout': str, ...} dict.
    Grep returns a list of string matches or a similar dict shape.
    We don't know exactly which transport variant is in play, so we
    handle: str | dict[str, str|...] | list[str] | None. Anything
    else passes through untouched."""
    if resp is None:
        return resp, False
    if isinstance(resp, str):
        new, did = _truncate_str(resp)
        return new, did
    if isinstance(resp, dict):
        any_changed = False
        new = dict(resp)
        for k, v in resp.items():
            if isinstance(v, str):
                truncated, did = _truncate_str(v)
                if did:
                    new[k] = truncated
                    any_changed = True
            elif isinstance(v, list):
                joined = "\n".join(str(x) for x in v)
                if len(joined.encode("utf-8")) > _CAP_BYTES:
                    truncated, _ = _truncate_str(joined)
                    new[k] = [truncated]
                    any_changed = True
        return new, any_changed
    if isinstance(resp, list):
        joined = "\n".join(str(x) for x in resp)
        if len(joined.encode("utf-8")) > _CAP_BYTES:
            truncated, _ = _truncate_str(joined)
            return [truncated], True
        return resp, False
    return resp, False


async def cap_builtin_tool_output(input_data, tool_use_id, context):
    """PostToolUse hook for Bash / Grep / WebFetch / WebSearch — cap
    oversized outputs.

    Conservative: only modifies tool_response when it actually exceeds
    the cap. Returns `{}` (= pass-through) for any result already
    within budget. Never raises — on unexpected shape we emit nothing
    and let the original response flow through.
    """
    try:
        resp = input_data.get("tool_response")
        new_resp, was_capped = _cap_tool_response(resp)
        if not was_capped:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "updatedMCPToolOutput": new_resp,
            }
        }
    except Exception:
        # Defensive: a buggy hook would crash the whole session. Pass
        # through on any internal error.
        return {}


# ── PreToolUse: deny Bash writes ────────────────────────────────────────────
#
# Bash is a SHARED-filesystem operation in this swarm — 10 agents run as
# the same Linux UID, all see the same blackboard / source tree / sibling
# workdirs. With sandbox=False (LXC hosts), no OS-level write isolation
# exists. This hook is the primary defense: agents may READ via Bash but
# may NOT write or destroy.
#
# Patterns are conservative — false positives cost the agent at most one
# extra turn (it gets a "rejected" reply and re-plans). False negatives
# (a destructive op slipping through) are the real cost we're optimising
# against. Real Bash usage in this swarm (per events.jsonl): awk / wc /
# head / tail / grep on tree.tsv. None of those legitimate uses match
# any deny pattern below.

_BASH_DENY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # File redirect to anywhere except /dev/null or fd-only redirects.
    # Allowed: `2>&1`, `>/dev/null`, `2>/dev/null`, `1>&2`, `&>>/dev/null`.
    # Blocked: `> file`, `>> file`, `> /path/file`.
    (
        re.compile(
            r"""(?<![0-9])      # not part of a fd number like 2>&1
            (?<!<)              # not part of <<EOF heredoc-ish
            >{1,2}\s*           # redirect operator
            (?!                 # but NOT followed by:
                /dev/null
              | /dev/stderr
              | /dev/stdout
              | &\d             # &1 / &2 fd redirect
              | \d&             # 1& / 2&
            )
            """,
            re.VERBOSE,
        ),
        "Bash file-redirect rejected — Bash is read-only in this swarm "
        "(use Edit / syntax_check / size_project for writes; agent's "
        "cwd-scoped Edit tool is the only sanctioned write path).",
    ),
    # Pipe-to-tee writes a file.
    (
        re.compile(r"\btee\b(?!\s+(?:-h|--help|--version))"),
        "tee rejected — writes a file path; use Edit instead.",
    ),
    # Destructive verbs (match as whole-word).
    (
        re.compile(
            r"\b(rm|rmdir|mv|cp|truncate|dd|mkdir|chmod|chown|chgrp|"
            r"unlink|shred|ln)\b"
        ),
        "destructive Bash command rejected (rm / mv / cp / mkdir / chmod / "
        "etc.) — agents must not modify shared filesystem state. "
        "If you need a working file, use Edit (cwd-scoped).",
    ),
    # In-place sed.
    (
        re.compile(r"\bsed\s+(?:-[A-Za-z]*i|--in-place)"),
        "in-place sed rejected — use Edit for file modification.",
    ),
    # tar in create mode (-c / --create).
    (
        re.compile(r"\btar\b[^|;&]*\s(?:-[A-Za-z]*c|--create)"),
        "tar create rejected — agents don't need to package files.",
    ),
    # Process control — must not affect supervisor or sibling specialists.
    (
        re.compile(r"\b(kill|pkill|killall|reboot|shutdown|halt|poweroff)\b"),
        "process-control command rejected — agent must not kill processes.",
    ),
    # System modification.
    (
        re.compile(
            r"\b(useradd|usermod|userdel|groupadd|groupdel|"
            r"systemctl|service|mount|umount|swapon|swapoff)\b"
        ),
        "system-modifying command rejected.",
    ),
    # Package install / network state change.
    (
        re.compile(
            r"\b(apt(?:-get)?|dpkg|yum|dnf|pacman|brew|pip|pipx|uv|npm|cargo)\s+"
            r"(install|remove|uninstall|upgrade|update|add|publish|init|sync)"
        ),
        "package-management command rejected — agent must not change the "
        "Python / system environment.",
    ),
]


_SQUOTE = re.compile(r"'[^']*'")
_DQUOTE = re.compile(r'"(?:[^"\\]|\\.)*"')

# `bash -c '...'` / `sh -c '...'` / `python -c '...'` etc. — the quoted body
# can hide arbitrary destructive ops that our pattern scanner can't safely
# inspect (we strip quotes before scanning, so a `bash -c 'rm ...'` would
# slip through). Reject the wrapper outright; if the agent genuinely needs
# a multi-step script, they should use Edit to create a file and run it.
_EXEC_C_PATTERN = re.compile(
    r"\b(?:ba|z|d)?sh\s+(?:[^|;&]*\s)?-c\b|\bpython3?\s+(?:[^|;&]*\s)?-c\b"
)


def _strip_shell_quotes(cmd: str) -> str:
    """Remove single- and double-quoted regions from a shell command.

    awk / sed / grep scripts inside `'...'` routinely contain `>`, `<`,
    `rm`, `mv` as DSL operators / string literals — never as shell
    redirects or commands. Stripping the quoted regions before
    pattern-matching prevents false positives like `awk 'NR>1'` (the
    `>` is awk's numeric comparison, not a file redirect).

    Bypass risk: `bash -c '<destructive>'` would have its body
    stripped too, so we ALSO check for exec-c patterns on the
    *original* command before this stripping.
    """
    cmd = _SQUOTE.sub("''", cmd)
    cmd = _DQUOTE.sub('""', cmd)
    return cmd


async def block_bash_writes(input_data, tool_use_id, context):
    """PreToolUse hook for Bash — reject writes / destructive ops.

    Only inspects the `Bash` tool. Other tools (Read, Edit, Grep,
    WebFetch, WebSearch, custom MCP) pass through untouched.

    Defensive: any internal error → pass-through (return `{}`). Better
    to allow one suspicious call than to crash the session and lose
    the iter.
    """
    try:
        if input_data.get("tool_name") != "Bash":
            return {}
        tool_input = input_data.get("tool_input") or {}
        cmd = tool_input.get("command", "")
        if not isinstance(cmd, str) or not cmd.strip():
            return {}

        # Pass-1: catch exec-via-`-c` BEFORE quote stripping; the body
        # is by definition opaque to our regex so we can't safely allow
        # it. (rare in practice — agents normally just run the command
        # directly, not via bash -c '...'.)
        if _EXEC_C_PATTERN.search(cmd):
            return {
                "decision": "block",
                "reason": (
                    "exec-via-`-c` rejected — bash/sh/python `-c` "
                    "wrappers hide their body from the safety scanner. "
                    "Run the command directly, or use Edit to write a "
                    "script file scoped to your workdir."
                ),
            }

        # Pass-2: strip quoted DSL bodies, then check for shell-level
        # writes / destructive verbs. This is the path that allows
        # `awk 'NR>1'`, `awk '$5=="rm"'`, `grep '>'` etc.
        scan = _strip_shell_quotes(cmd)
        for pattern, reason in _BASH_DENY_PATTERNS:
            if pattern.search(scan):
                return {
                    "decision": "block",
                    "reason": (
                        f"{reason}\n\n"
                        "If you need to write a file: use the `Edit` tool "
                        "(scoped to your workdir). The Bash tool in this "
                        "swarm is intentionally read-only because all "
                        "specialists share the same filesystem (blackboard "
                        "+ source tree + sibling workdirs). For data "
                        "extraction, pipe to head/awk/grep but DON'T "
                        "redirect to a file."
                    ),
                }
        return {}
    except Exception:
        return {}  # never block on hook bug


# ── PreToolUse: deny Bash reads of blackboard (no-lineage ablation only) ────
#
# Active only when MAGENT_NO_LINEAGE=1. The no-lineage ablation closes four
# lineage-feedback channels: prompt rendering (LEADERBOARD/KNOWLEDGE/Recent
# Activity/Saturation are blanked), tools (read_snapshot/diff_snapshots are
# removed from allowed_tools), the static system-prompt priors (kept; not
# feedback from this run), and Bash on blackboard (closed by THIS hook).
#
# Empirically (PG ckpt 2026-04-29 events.jsonl) ~58% of Bash calls with
# parseable args target blackboard paths — `awk '$5=="arch"' tree.tsv` and
# similar. Without this hook, the prompt-side ablation is undermined.
#
# Pattern matches:
#   - any path containing "blackboard" (the dir name in MAGENT_LOCAL_ROOT)
#   - blackboard-specific basenames: tree.tsv, results.tsv,
#     lineage_snapshots, events.jsonl, supervisor_audit.jsonl, best.json
#
# These names are blackboard-specific (don't appear in workdir / source
# tree / vendor), so false-positive risk is low. The agent's own workdir
# files (train_gpt.py, syntax_check output, etc.) pass through unchanged.

_BLACKBOARD_PATH_PATTERN = re.compile(
    r"(?:"
    r"\bblackboard\b"
    r"|\btree\.tsv\b"
    r"|\bresults\.tsv\b"
    r"|\blineage_snapshots\b"
    r"|\bsupervisor_audit\.jsonl\b"
    r"|\bevents\.jsonl\b"
    r"|\bbest\.json\b"
    r")"
)


async def block_bash_blackboard(input_data, tool_use_id, context):
    """PreToolUse hook for Bash — deny reads of blackboard files when
    MAGENT_NO_LINEAGE=1.

    Pass-through when env var is unset or '0'. Pass-through for non-Bash
    tools. Defensive: any internal error → pass-through (return `{}`).
    """
    try:
        if os.environ.get("MAGENT_NO_LINEAGE", "0") != "1":
            return {}
        if input_data.get("tool_name") != "Bash":
            return {}
        tool_input = input_data.get("tool_input") or {}
        cmd = tool_input.get("command", "")
        if not isinstance(cmd, str) or not cmd.strip():
            return {}
        if _BLACKBOARD_PATH_PATTERN.search(cmd):
            return {
                "decision": "block",
                "reason": (
                    "Bash access to blackboard files (tree.tsv, "
                    "results.tsv, lineage_snapshots/, events.jsonl, "
                    "supervisor_audit.jsonl, best.json, or any path "
                    "under blackboard/) is rejected for this run. "
                    "Prior-trial logs are unavailable in this ablation; "
                    "rely on your workdir state, the static knowledge "
                    "files (INIT.md / SOTA_STACK.md / LESSONS.md), "
                    "and your own current-session memory only."
                ),
            }
        return {}
    except Exception:
        return {}  # never block on hook bug

