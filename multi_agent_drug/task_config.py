"""DrugTaskAdapter — TDC ADMET molecular property prediction task.

Score: aggregate_score (higher is better) — normalized improvement over
baseline, averaged across 22 TDC ADMET tasks.

Submission mode: local subprocess (no SCHED, no GPU pod). Trials run with
.venv_drug Python on the local host.

Editable surface: experiment.py (root seed) + pipeline/ tree (features,
models, calibration). Agents edit pipeline/*.py; run_trial_drug.py is
staged but not editable.

Leakage guard: run_trial_drug.py controls all TDC data access. The agent's
pipeline code receives only train/val DataFrames and test SMILES (no Y).
TDC final test evaluation is disabled during the agent loop; it only runs
in calibrate_baseline.py and at final paper reporting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent_core.task_adapter import TaskAdapter

_PKG_ROOT = Path(__file__).resolve().parent

# Per-task baseline scores (higher = better for AUROC/Spearman;
# lower = better for MAE). Used by aggregate_score normalisation in
# run_trial_drug.py. Populated by calibrate_baseline.py at first run;
# the dict here is a placeholder that run_trial_drug.py will override
# from the baseline_scores.json file written by calibrate_baseline.
BASELINE_SCORES_FILE = _PKG_ROOT / "knowledge" / "baseline_scores.json"

# Lazy singleton — the benchmark-data provider (group loader / metric / task type).
_DATA_PROVIDER = None


class DrugTaskAdapter(TaskAdapter):
    """TDC ADMET drug property prediction adapter — local execution."""

    # ── Paths ────────────────────────────────────────────────────────────────

    @property
    def pkg_root(self) -> Path:
        return _PKG_ROOT

    @property
    def knowledge_dir(self) -> Path:
        return _PKG_ROOT / "knowledge"

    @property
    def baseline_filename(self) -> str:
        return "experiment.py"

    @property
    def seed_file(self) -> str:
        return "experiment.py"

    @property
    def editable_tree(self) -> str:
        # Agent edits files inside pipeline/: features.py, models.py,
        # calibration.py, pipeline.py. run_trial_drug.py is staged
        # separately and is NOT inside editable_tree (agent cannot touch it).
        return "pipeline"

    @property
    def run_script(self) -> str:
        return "run_trial.sh"

    # ── TSV schema ───────────────────────────────────────────────────────────

    @property
    def tsv_fields(self) -> list[str]:
        return [
            "exp_id", "timestamp", "specialist", "parent_exp", "baseline_exp",
            "domain", "hypothesis", "expected_delta", "status",
            "aggregate_score",   # primary (higher better): normalised improvement
            "n_tasks_ok",        # how many of 22 tasks produced a valid score
            "delta_vs_best",
            "elapsed_s",         # wall time for full trial
            "snapshot_path", "notes",
        ]

    @property
    def score_field(self) -> str:
        return "aggregate_score"

    @property
    def score_short_label(self) -> str:
        return "agg"

    @property
    def score_lower_is_better(self) -> bool:
        return False

    def parse_validate_record(self, record: dict) -> dict:
        """Map run_trial_drug JSONL → TSV row fields.

        Status taxonomy:
          OK              → keep
          CRASH           → crash
          TIMEOUT         → timeout
          PREFLIGHT_CRASH → preflight_crash
        """
        status_raw = record.get("status", "CRASH")
        status_map = {
            "OK":              "keep",
            "CRASH":           "crash",
            "TIMEOUT":         "timeout",
            "PREFLIGHT_CRASH": "preflight_crash",
        }
        return {
            "status":          status_map.get(status_raw, "crash"),
            "aggregate_score": _fmt_float(record.get("aggregate_score")),
            "n_tasks_ok":      str(int(record.get("n_tasks_ok") or 0)),
            "elapsed_s":       _fmt_float(record.get("elapsed_s")),
            "raw_status":      status_raw,
            "kill_reason":     record.get("kill_reason") or "",
        }

    def empty_validate_row(self, status: str) -> dict:
        return {
            "status":          status,
            "aggregate_score": "",
            "n_tasks_ok":      "0",
            "elapsed_s":       "",
        }

    # ── Specialists ──────────────────────────────────────────────────────────

    @property
    def doer_domains(self) -> tuple[str, ...]:
        # fphs:  physchem feature discovery (RDKit 2D descriptors, physchem rules)
        # fsub:  substructure feature discovery (ECFP, MACCS, alerts, scaffolds)
        # lit:   literature agent (endpoint mechanism → feature/data hypothesis)
        # data:  data quality, SMILES cleaning, class imbalance (pipeline-internal only)
        # daugm: data augmentation via external datasets (write_external_data tool)
        # modl:  model backbone search (GBDT, RF, MLP, multitask, hyperparams)
        # calib: probability calibration, threshold tuning, uncertainty
        # meta:  meta analyst (lineage analysis, next-round direction)
        return ("fphs", "fsub", "lit", "data", "daugm", "modl", "calib", "meta")

    @property
    def analyst_domains(self) -> tuple[str, ...]:
        return ()

    def specialist_classes(self) -> dict[str, type]:
        from multi_agent_drug.agents import fphs, fsub, lit, data, daugm, modl, calib, meta
        return {
            "fphs":  fphs.PhyschemFeatureDoer,
            "fsub":  fsub.SubstructFeatureDoer,
            "lit":   lit.LiteratureDoer,
            "data":  data.DataDoer,
            "daugm": daugm.DataAugDoer,
            "modl":  modl.ModelDoer,
            "calib": calib.CalibrationDoer,
            "meta":  meta.MetaDoer,
        }

    # ── Pipeline / stage / size ──────────────────────────────────────────────

    @property
    def stage_files(self) -> tuple[tuple[str, str], ...]:
        # run_trial_drug.py is staged to workdir but NOT inside editable_tree
        # so agent code in pipeline/ cannot import or modify it.
        return (
            ("run_trial.sh",          "run_trial.sh"),
            ("run_trial_drug.py",     "run_trial_drug.py"),
            ("tools/run_classify.py", "run_classify.py"),
        )

    @property
    def trial_output_dirs(self) -> tuple[str, ...]:
        return ("full_eval_results",)

    def size_check(self, workdir: str) -> dict:
        """No artifact size constraint for Drug Discovery."""
        from pathlib import Path
        total = sum(
            f.stat().st_size
            for f in Path(workdir).rglob("*.py")
            if f.is_file()
        )
        return {
            "ok":          True,
            "verdict":     "ok",
            "code_bytes":  total,
            "model_bytes": None,
            "total_bytes": total,
            "limit_bytes": None,
        }

    # ── Tools ────────────────────────────────────────────────────────────────

    @property
    def custom_tool_names(self) -> tuple[str, ...]:
        names = [
            "syntax_check", "read_snapshot", "diff_snapshots",
            "rebase_to", "submit_trial",
        ]
        import os
        mode = os.environ.get("HARNESS_ABLATION_MODE", "feature_only")
        if mode in ("data_only", "joint"):
            names.append("write_external_data")
        return tuple(names)

    def bind_tools(self) -> list[Any]:
        from agent_core.tools.submit import submit_trial
        from agent_core.tools.code_inspect import syntax_check
        from agent_core.tools.workdir import read_snapshot, diff_snapshots, rebase_to
        tools = [syntax_check, read_snapshot, diff_snapshots, rebase_to, submit_trial]
        import os
        mode = os.environ.get("HARNESS_ABLATION_MODE", "feature_only")
        if mode in ("data_only", "joint"):
            from multi_agent_drug.tools.write_external_data import write_external_data
            tools.append(write_external_data)
        return tools

    @property
    def snapshot_extra_dirs(self) -> list[str]:
        """Extra workdir directories to include in keep snapshots.

        In data_only mode, external_data/ holds the agent's contributed CSV
        files — without snapshotting them, lineage and rebase_to would be broken.
        """
        import os
        if os.environ.get("HARNESS_ABLATION_MODE", "feature_only") == "data_only":
            return ["external_data"]
        return []

    # ── Prompts ──────────────────────────────────────────────────────────────

    def build_system_prompt(self, domain: str) -> str:
        from multi_agent_drug.agents.prompts import build_system_prompt
        return build_system_prompt(domain)

    def specialist_preamble(self, domain: str) -> str:
        from multi_agent_drug.agents.prompts import DOMAIN_PREAMBLES
        try:
            return DOMAIN_PREAMBLES[domain]
        except KeyError as e:
            raise ValueError(f"unknown domain {domain!r}") from e

    def hard_limits_section(self) -> str:
        return (
            "## Hard limits (enforced by the harness)\n"
            "\n"
            "- **OBJECTIVE**: maximise `aggregate_score` — normalised improvement "
            "over baseline, averaged across 22 TDC ADMET tasks. Higher is better.\n"
            "- **Internal validation only**: the agent loop reward uses an internal "
            "scaffold-split validation set carved from train_val. The TDC test set "
            "is NOT used during search — it is frozen for final paper reporting.\n"
            "- **No test label access**: `pipeline.py` receives only train/val "
            "DataFrames and test SMILES (no Y column). Any attempt to access test "
            "labels or call TDC evaluate() directly in pipeline code will crash "
            "the trial.\n"
            "- **No artifact size cap**: code can be any size.\n"
            "- **Local execution**: trials run on CPU / single GPU via local "
            "subprocess. No SCHED, no DDP.\n"
        )

    def keep_discard_semantics(self) -> str:
        return (
            "## Keep / discard semantics\n"
            "\n"
            "After `submit_trial` returns, `status` is one of:\n"
            "\n"
            "- **`keep`** — aggregate_score improved over best. Snapshot saved; "
            "descendants can rebase to it.\n"
            "- **`discard`** — valid score but did not beat current best.\n"
            "- **`crash`** — pipeline raised an exception or produced invalid scores.\n"
            "- **`timeout`** — trial exceeded wall time limit.\n"
            "- **`preflight_crash`** — syntax check failed before trial ran.\n"
            "\n"
            "Only `keep` rows update best.json and are rebase-able.\n"
        )

    # ── Bootstrap ────────────────────────────────────────────────────────────

    @property
    def baseline_score_default(self) -> float:
        # Placeholder — operator must run calibrate_baseline first.
        return 0.0

    @property
    def baseline_score_flag(self) -> str:
        return "--baseline-score"

    @property
    def requires_calibrated_baseline(self) -> bool:
        return True

    @property
    def bootstrap_hypothesis(self) -> str:
        return ("RDKit physchem + ECFP4 (1024 bit) + XGBoost baseline, "
                "scaffold split, no calibration")

    @property
    def baseline_note(self) -> str:
        return "RDKit+ECFP4+XGBoost baseline reference (22 TDC ADMET tasks)"

    # ── SCHED / submission ──────────────────────────────────────────────────────

    @property
    def sched_name_prefix(self) -> str:
        return "drug"

    @property
    def submission_mode(self) -> str:
        env = os.environ.get("MAGENT_SUBMISSION_MODE", "").strip().lower()
        if env in ("job", "notebook", "local"):
            return env
        return "local"

    # ── Benchmark data provider ────────────────────────────────────────────────

    def data_provider(self):
        """TDC ADMET data contract (group loader / metric / task type / isolation).

        Lazy module-level singleton; consumed by calibrate_baseline and
        write_external_data on the orchestration side. (The staged runner builds
        its own provider by MAGENT_TASK — see run_trial_drug.py.)
        """
        global _DATA_PROVIDER
        if _DATA_PROVIDER is None:
            from multi_agent_drug.benchmark_data import TdcAdmetDataProvider
            _DATA_PROVIDER = TdcAdmetDataProvider()
        return _DATA_PROVIDER

    @property
    def pod_env_for_trial(self) -> dict[str, str]:
        """Inject both venv paths and TDC data dir into run_trial.sh env.

        HARNESS_PYTHON  → full venv with PyTDC (used by run_trial_drug.py).
        AGENT_PYTHON    → stripped venv without PyTDC (used for experiment.py
                          subprocesses — hard TDC isolation at venv level).
        """
        from multi_agent_drug.harness.config import VENV_PYTHON, AGENT_PYTHON, TDC_DATA_DIR
        env: dict[str, str] = {}
        if VENV_PYTHON and VENV_PYTHON != "skip":
            env["HARNESS_PYTHON"] = VENV_PYTHON
        if AGENT_PYTHON and AGENT_PYTHON != "skip":
            env["AGENT_PYTHON"] = AGENT_PYTHON
        env["HARNESS_TDC_DATA_DIR"] = TDC_DATA_DIR
        # HARNESS_PKG_ROOT: absolute path to multi_agent_drug/ source directory.
        # run_trial_drug.py is staged into WORKDIR (cd "$WORKDIR" in run_trial.sh),
        # so Path(__file__) there points to WORKDIR, not the source tree.
        # _ablation_freeze() needs the REAL source to copy frozen baseline files.
        env["HARNESS_PKG_ROOT"] = str(_PKG_ROOT)
        # MAGENT_TASK selects the benchmark-data provider in the staged runner
        # (run_trial_drug.py:_load_data_provider). "drug" → TdcAdmetDataProvider.
        env["MAGENT_TASK"] = "drug"
        # Inner soft limit: run_trial_drug.py stops the task loop here and
        # writes the JSONL result.  Must be < local_config.timeout_s to ensure
        # json.dump() completes before the outer SIGTERM fires.
        # 3600s inner + 4200s outer = 600s buffer for writing.
        env["HARNESS_WALL_LIMIT_S"] = "3600"
        baseline_file = _PKG_ROOT / "knowledge" / "baseline_scores.json"
        if baseline_file.is_file():
            env["HARNESS_BASELINE_SCORES"] = str(baseline_file)
        return env

    @property
    def local_config(self) -> dict:
        return {
            "cuda_visible_devices": "",   # CPU-first; single GPU optional
            "timeout_s": 4200,            # 70 min outer hard kill (600s buffer over 3600s inner)
        }


def _fmt_float(v) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.6f}"
    except (TypeError, ValueError):
        return ""


__all__ = ["DrugTaskAdapter"]
