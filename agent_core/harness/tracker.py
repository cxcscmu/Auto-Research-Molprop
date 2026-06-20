"""Experiment tracking for the multi-agent harness — task-agnostic.

Reads task-specific schema (TSV column list, primary score field name,
baseline source filename) via `current_adapter()`. PG values flow
through the adapter's PGTaskAdapter; nc / cifar forks supply their own.

The primary consumer is blackboard.py, which serialises TSV writes
behind the blackboard lock. Never write results.tsv from a specialist
directly — go through the blackboard.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional

from . import config


# ── Adapter helpers (lazy — adapter must be registered at call time) ─────────

def _adapter():
    from agent_core import current_adapter
    return current_adapter()


def _tsv_fields() -> list[str]:
    return _adapter().tsv_fields


def _score_field() -> str:
    return _adapter().score_field


def _score_lower_is_better() -> bool:
    return _adapter().score_lower_is_better


def _baseline_filename() -> str:
    return _adapter().baseline_filename


# ── Status constants (task-agnostic) ─────────────────────────────────────────

_STATUS_INFORMATIVE = frozenset({
    "keep", "discard", "crash",
    "size_blocked", "preflight_crash",
    "eval_budget_overrun", "train_budget_overrun",
})

# Maps run_classify.py's coarse status → blackboard's status enum.
# Task-agnostic by convention; nc/cifar can override by replacing this
# constant in their task package or via adapter (future).
_STATUS_MAP = {
    "VALID":      "keep",      # agent decides keep vs discard based on delta
    "INCOMPLETE": "crash",
    "CRASH":      "crash",
    "DQ_TRAIN":   "train_budget_overrun",
    "DQ_EVAL":    "eval_budget_overrun",
    "DQ_SIZE":    "size_blocked",
}


# ── Regex for log-tail extraction ────────────────────────────────────────────

_LOG_TAIL_BYTES = 50_000

_EXC_RE = re.compile(
    r"^((?:[A-Za-z_][\w]*\.)*[A-Za-z_]\w*"
    r"(?:Error|Exception|Failed|Unsupported)"
    r"|Fatal Python error"
    r"|CUDA error)(?::\s*(.*))?$",
    re.MULTILINE,
)
_ANY_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')

_PHASE_SUMMARY_RE = re.compile(
    r"---\s*PHASE_SUMMARY\s*---\s*\n(.+?)(?:\n---|\Z)", re.DOTALL,
)
_PHASE_KEYS = (
    "train_s", "eval_s", "bpb_eval_s",
    "compress_s", "gptq_s", "wrap_up_s", "total_wall_s",
)

_POST_PACK_CODE_RE  = re.compile(r"^Packed code:\s*(\d+)\s*bytes", re.MULTILINE)
_POST_PACK_MODEL_RE = re.compile(r"^Model blob\s*:\s*(\d+)\s*bytes", re.MULTILINE)


# ── TSV I/O ──────────────────────────────────────────────────────────────────

def read_results() -> list[dict]:
    """Return every row of results.tsv, in insertion order. Empty list on ENOENT."""
    path = config.RESULTS_TSV
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def append_result(row: dict) -> None:
    """Append a single row to results.tsv. Caller must hold the blackboard lock.

    Unknown keys are silently dropped; missing keys become empty strings.
    """
    fields = _tsv_fields()
    path = config.RESULTS_TSV
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, delimiter="\t",
            extrasaction="ignore", lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def next_exp_id() -> str:
    """Return the next `NNN` id, lexically sortable up to 999. Zero-padded."""
    rows = read_results()
    ids: list[int] = []
    for r in rows:
        try:
            ids.append(int(r.get("exp_id", "")))
        except (TypeError, ValueError):
            pass
    return f"{max(ids, default=-1) + 1:03d}"


def find_best() -> Optional[dict]:
    """Scan results.tsv for the kept row with the best score (per adapter).

    Does not read best.json directly — the TSV is the source of truth;
    best.json is a cached view for the LEADERBOARD renderer.
    """
    score_field = _score_field()
    lower_is_better = _score_lower_is_better()
    best: Optional[dict] = None
    best_score = float("inf") if lower_is_better else float("-inf")
    for r in read_results():
        if r.get("status") not in ("keep", "baseline"):
            continue
        try:
            score = float(r.get(score_field, ""))
        except (TypeError, ValueError):
            continue
        if (lower_is_better and score < best_score) or (not lower_is_better and score > best_score):
            best_score, best = score, r
    return best


# ── Empty validate_row helper ────────────────────────────────────────────────

def empty_validate_row(status: str) -> dict:
    """Return an empty validate_row carrying just the status.

    Delegates to `current_adapter().empty_validate_row(status)` — each
    task defines its own empty-row shape (PG: 5 measurement fields;
    nc/cifar may differ).
    """
    from agent_core import current_adapter
    return current_adapter().empty_validate_row(status)


# ── Log extraction ───────────────────────────────────────────────────────────

def _read_log_tail(log_path: Path) -> Optional[str]:
    """Return the last _LOG_TAIL_BYTES bytes of log_path, or None on ENOENT."""
    if not log_path.exists():
        return None
    try:
        data = log_path.read_bytes()
    except OSError:
        return None
    return data[-_LOG_TAIL_BYTES:].decode("utf-8", errors="replace")


def _baseline_frame_re() -> re.Pattern:
    """Lazy regex: matches `File "...<baseline_filename>", line N, in fn`."""
    baseline = _baseline_filename()
    return re.compile(rf'File "[^"]*{re.escape(baseline)}", line (\d+), in (\S+)')


def extract_crash_excerpt(log_path: Path) -> Optional[str]:
    """Return `<ExceptionType>: <msg> (<baseline>:N in fn)` or None.

    Last-match semantics: the outermost exception and the deepest
    baseline-source frame tend to be closest to the real bug. Falls back
    to any Python frame when torch.compile/Dynamo swallows the user frame.
    """
    tail = _read_log_tail(log_path)
    if tail is None:
        return None

    baseline = _baseline_filename()
    frame_re = _baseline_frame_re()

    exc_matches = list(_EXC_RE.finditer(tail))
    frame_matches = list(frame_re.finditer(tail))
    if not exc_matches and not frame_matches:
        return None

    parts: list[str] = []
    if exc_matches:
        m = exc_matches[-1]
        exc_type = m.group(1)
        exc_msg = (m.group(2) or "").strip()
        if len(exc_msg) > 200:
            exc_msg = exc_msg[:200] + "…"
        parts.append(f"{exc_type}: {exc_msg}" if exc_msg else exc_type)
    if frame_matches:
        m = frame_matches[-1]
        parts.append(f"({baseline}:{m.group(1)} in {m.group(2)})")
    else:
        any_frames = list(_ANY_FRAME_RE.finditer(tail))
        if any_frames:
            m = any_frames[-1]
            fname = Path(m.group(1)).name
            parts.append(f"({fname}:{m.group(2)} in {m.group(3)})")
    return " ".join(parts) if parts else None


def extract_phase_summary(log_path: Path) -> Optional[dict]:
    """Return the PHASE_SUMMARY block as {key: float}. None if absent."""
    tail = _read_log_tail(log_path)
    if tail is None:
        return None
    m = _PHASE_SUMMARY_RE.search(tail)
    if not m:
        return None
    block = m.group(1)
    out: dict[str, float] = {}
    for key in _PHASE_KEYS:
        m2 = re.search(rf"\b{key}=([0-9.]+)", block)
        if m2:
            try:
                out[key] = float(m2.group(1))
            except ValueError:
                pass
    return out or None


def extract_pack_breakdown(log_path: Path) -> Optional[dict]:
    """Return {code_bytes, model_bytes, source} from pack_submission output."""
    tail = _read_log_tail(log_path)
    if tail is None:
        return None
    code_m = _POST_PACK_CODE_RE.search(tail)
    model_m = _POST_PACK_MODEL_RE.search(tail)
    if not (code_m and model_m):
        return None
    try:
        return {
            "code_bytes":  int(code_m.group(1)),
            "model_bytes": int(model_m.group(1)),
            "source":      "post-run",
        }
    except ValueError:
        return None


# ── run_classify.py JSONL → TSV row ──────────────────────────────────────────

def parse_validate_result(jsonl_path: Path) -> dict:
    """Convert run_classify.py's run_seed<N>.jsonl into a partial TSV row.

    Reads the jsonl, takes the last record (multi-seed future-proofing),
    then delegates the dict→row mapping to
    `current_adapter().parse_validate_record(record)` so each task can
    define its own field translation. PG's adapter implementation lives
    in `multi_agent_pg/task_config.py:PGTaskAdapter.parse_validate_record`.
    """
    raw = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    if not raw:
        raise ValueError(f"empty jsonl: {jsonl_path}")
    record = json.loads(raw[-1])
    from agent_core import current_adapter
    return current_adapter().parse_validate_record(record)


def _parse_validate_record_default(record: dict) -> dict:
    """Default dict→TSV-row translation (PG-shape).

    Tasks can override `TaskAdapter.parse_validate_record` to customise;
    PGTaskAdapter delegates back to this function unchanged.
    """
    classify_status = record.get("status", "CRASH")
    mapped = _STATUS_MAP.get(classify_status, "crash")
    score_field = _score_field()

    return {
        "status":         mapped,
        score_field:      _fmt_float(record.get(score_field)),
        "artifact_bytes": record.get("artifact_bytes") or "",
        "train_s":        _fmt_float(record.get("train_s")),
        "eval_s":         _fmt_float(record.get("eval_s")),
        "total_s":        _fmt_float(record.get("total_wall_s")),
        "raw_status":     classify_status,          # preserved for debugging
        "kill_reason":    record.get("kill_reason") or "",
    }


def _fmt_float(value) -> str:
    """None/non-numeric → ""; float → 6-dec str; int → int str."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""          # guard: True/False shouldn't leak into numeric cols
    if isinstance(value, (int,)):
        return str(value)
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


# ── Back-compat: TSV_FIELDS module-level access ──────────────────────────────

def __getattr__(name: str):
    """Lazy module-level attribute access for TSV_FIELDS.

    Some PG code (and test snippets) does `tracker.TSV_FIELDS` directly.
    Resolve via adapter at access time. Same as a module property.
    """
    if name == "TSV_FIELDS":
        return _tsv_fields()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
