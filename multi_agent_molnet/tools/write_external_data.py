"""write_external_data — controlled MCP tool for data_only ablation.

Agents call this instead of directly writing files, so the harness can
enforce path restrictions and size limits before the CSV lands on disk.

The harness (run_trial_drug.py) will deduplicate the CSV against:
  1. MoleculeNet final test molecules
  2. Internal val molecules (carved from train_val)
  3. Existing train molecules (dedup)
before merging into the training set.  The agent never sees this dedup.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from agent_core.tools import tool

# Limits (must match run_trial_drug.py constants)
_MAX_ROWS  = 5_000
_MAX_BYTES = 2 * 1024 * 1024   # 2 MB

# Valid task names come from the active benchmark's data provider — single source of
# truth (the task universe lives in benchmark_data.py, not duplicated here).
_VALID_TASKS_CACHE: "set | None" = None


def _valid_tasks() -> set:
    """Valid task names for the active benchmark (cached after first lookup)."""
    global _VALID_TASKS_CACHE
    if _VALID_TASKS_CACHE is None:
        from agent_core import current_adapter
        _VALID_TASKS_CACHE = set(current_adapter().data_provider().task_names())
    return _VALID_TASKS_CACHE


@tool(
    name="write_external_data",
    description=(
        "Write external training data for a specific MoleculeNet endpoint (data_only mode only).\n\n"
        "LEAKAGE-SAFE FILTER (harness-side, you cannot bypass it). After you submit a "
        "trial the harness applies three layers and reports the outcome in "
        "per_task[task]['data_aug'] ('verdict' + 'agent_note'):\n"
        "  1. Identity dedup: removes molecules that match MoleculeNet test / val / train "
        "by standardized InChIKey (salt & charge variants included).\n"
        "  2. SAME-SOURCE REJECTION: if >5% of the task's MoleculeNet test molecules appear "
        "in your dataset, the WHOLE file is rejected (0 rows used). This means a dataset "
        "that is the benchmark's own/original source — e.g. Delaney/AqSolDB (esol), "
        "SAMPL/FreeSolv (freesolv), the BACE-1 inhibitor literature set (bace), the NCI "
        "DTP AIDS Antiviral screen (hiv), the Tox21 Data Challenge (tox21_*), the SIDER "
        "database (sider_*), FDA/ClinicalTrials.gov (clintox_ct_tox) — will be REJECTED. "
        "Do not submit those, and do not resubmit a rejected source.\n"
        "  3. Analog filter: removes molecules that are near-duplicates (ECFP4 "
        "Tanimoto >= 0.9) of any test molecule.\n"
        "AIM FOR GENUINELY INDEPENDENT assays (different lab/database than the task's "
        "original source).\n\n"
        "CSV format: must have 'Drug' (SMILES) and 'Y' (label) columns.\n"
        "First line SHOULD be: #source: <dataset name + where it came from> — recorded "
        "in the audit trail (and helps you avoid resubmitting rejected sources).\n\n"
        "Limits: max 5000 rows, max 2 MB. Classification Y must be 0 or 1. "
        "Regression Y should be in task-relevant range (extreme outliers are winsorized).\n\n"
        "Call once per task. Overwrite by calling again with the same task_name.\n"
        "Returns {ok, task_name, rows_written, note}."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_name": {
                "type": "string",
                "description": "Exact MoleculeNet endpoint name, e.g. 'esol', 'bace', 'tox21_nr_ar'.",
            },
            "workdir": {
                "type": "string",
                "description": "Your specialist workdir path (e.g. /home/.../workdirs/workdir_data).",
            },
            "csv_text": {
                "type": "string",
                "description": (
                    "CSV content with Drug and Y columns. "
                    "First line may be: #source: ChEMBL assay CHEMBL123456"
                ),
            },
        },
        "required": ["task_name", "workdir", "csv_text"],
    },
)
async def write_external_data(args: dict) -> dict:
    task_name = str(args.get("task_name", "")).strip()
    workdir   = str(args.get("workdir", "")).strip()
    csv_text  = str(args.get("csv_text", ""))

    # Validate task name
    if task_name not in _valid_tasks():
        return {"ok": False, "error": f"unknown task_name {task_name!r}. "
                f"Must be one of the 10 MoleculeNet endpoints."}

    # Validate workdir (must be under LOCAL_ROOT/workdirs/ — no path traversal)
    from agent_core.harness import config
    wd = Path(workdir).resolve()
    allowed_root = (config.WORKDIRS_ROOT).resolve()
    if not str(wd).startswith(str(allowed_root)):
        return {"ok": False,
                "error": f"workdir must be under {allowed_root}. Got: {wd}"}

    # Size limit
    if len(csv_text.encode("utf-8", errors="replace")) > _MAX_BYTES:
        return {"ok": False,
                "error": f"csv_text exceeds {_MAX_BYTES // 1024} KB limit."}

    # Parse CSV to count rows and validate columns
    try:
        import pandas as pd
        parse_text = csv_text
        if csv_text.lstrip().startswith("#"):
            parse_text = "\n".join(csv_text.splitlines()[1:])
        df = pd.read_csv(io.StringIO(parse_text))
    except Exception as exc:
        return {"ok": False, "error": f"CSV parse error: {exc}"}

    if "Drug" not in df.columns or "Y" not in df.columns:
        return {"ok": False,
                "error": "CSV must have 'Drug' and 'Y' columns."}

    n_rows = len(df)
    note_parts = []
    if n_rows > _MAX_ROWS:
        note_parts.append(f"harness will sample to {_MAX_ROWS} rows")

    # Write to workdir/external_data/{task_name}.csv
    ext_dir = wd / "external_data"
    ext_dir.mkdir(parents=True, exist_ok=True)
    out_path = ext_dir / f"{task_name}.csv"
    out_path.write_text(csv_text, encoding="utf-8")

    note_parts.append(
        "harness will apply the leakage-safe filter (identity dedup + same-source "
        "rejection if >5% test overlap + analog filter) before merging into fit. "
        "After the trial, read per_task['" + task_name + "']['data_aug'] "
        "('verdict','agent_note') — if 'rejected_same_source', do NOT resubmit this "
        "source; find an independent dataset."
    )
    return {
        "ok":          True,
        "task_name":   task_name,
        "rows_written": n_rows,
        "path":        str(out_path),
        "note":        " ".join(note_parts),
    }
