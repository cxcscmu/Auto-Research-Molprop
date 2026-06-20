"""MolNetTaskAdapter — MoleculeNet task adapter.

Inherits DrugTaskAdapter — shared specialists, pipeline staging, TSV schema, scoring,
size check, keep/discard semantics are all benchmark-agnostic — and overrides only what
must point at MoleculeNet: package root, knowledge dir, the data provider, and the
per-trial env (MAGENT_TASK=molnet + MoleculeNet data dir).

TODO(task#6 prompts): build_system_prompt / specialist_preamble / hard_limits_section /
bind_tools are still inherited from the drug (ADMET) prompts+tool. They are rewritten
for MoleculeNet in multi_agent_molnet/agents/prompts.py + tools/write_external_data.py
and overridden here. The data layer (loader/provider) and calibrate_baseline do NOT use
agent prompts, so this placeholder is safe for the data-layer + calibration steps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from multi_agent_drug.task_config import DrugTaskAdapter

_PKG_ROOT = Path(__file__).resolve().parent
_MOLNET_DATA_DIR = _PKG_ROOT.parent / "molnet_data"
_DATA_PROVIDER = None


class MolNetTaskAdapter(DrugTaskAdapter):
    """MoleculeNet adapter — reuses the drug harness, retargeted at molnet data."""

    # ── Paths (must override — drug versions use multi_agent_drug's _PKG_ROOT) ──

    @property
    def pkg_root(self) -> Path:
        return _PKG_ROOT

    @property
    def knowledge_dir(self) -> Path:
        return _PKG_ROOT / "knowledge"

    # ── Benchmark data provider ────────────────────────────────────────────────

    def data_provider(self):
        global _DATA_PROVIDER
        if _DATA_PROVIDER is None:
            from multi_agent_molnet.benchmark_data import MolNetDataProvider
            _DATA_PROVIDER = MolNetDataProvider()
        return _DATA_PROVIDER

    # ── Per-trial env (select molnet provider + molnet data dir) ────────────────

    @property
    def pod_env_for_trial(self) -> dict[str, str]:
        from multi_agent_drug.harness.config import VENV_PYTHON, AGENT_PYTHON
        env: dict[str, str] = {}
        if VENV_PYTHON and VENV_PYTHON != "skip":
            env["HARNESS_PYTHON"] = VENV_PYTHON
        if AGENT_PYTHON and AGENT_PYTHON != "skip":
            env["AGENT_PYTHON"] = AGENT_PYTHON
        # MolNet data is plain CSV — the bind-mount over this dir is the primary
        # isolation (venv layer gives no protection; there is no molnet package to omit).
        env["HARNESS_MOLNET_DATA_DIR"] = str(_MOLNET_DATA_DIR)
        env["HARNESS_PKG_ROOT"] = str(_PKG_ROOT)
        env["MAGENT_TASK"] = "molnet"
        env["HARNESS_WALL_LIMIT_S"] = "3600"
        baseline_file = _PKG_ROOT / "knowledge" / "baseline_scores.json"
        if baseline_file.is_file():
            env["HARNESS_BASELINE_SCORES"] = str(baseline_file)
        return env

    # ── Prompts / tools (MoleculeNet — override drug's ADMET versions) ──────────

    def build_system_prompt(self, domain: str) -> str:
        from multi_agent_molnet.agents.prompts import build_system_prompt
        return build_system_prompt(domain)

    def specialist_preamble(self, domain: str) -> str:
        from multi_agent_molnet.agents.prompts import DOMAIN_PREAMBLES
        try:
            return DOMAIN_PREAMBLES[domain]
        except KeyError as e:
            raise ValueError(f"unknown domain {domain!r}") from e

    def hard_limits_section(self) -> str:
        return (
            "## Hard limits (enforced by the harness)\n"
            "\n"
            "- **OBJECTIVE**: maximise `aggregate_score` — normalised improvement "
            "over baseline, averaged across 10 MoleculeNet endpoints. Higher is better.\n"
            "- **Internal validation only**: the agent loop reward uses an internal "
            "validation set carved from train_val. The MoleculeNet test set "
            "is NOT used during search — it is frozen for final paper reporting.\n"
            "- **No test label access**: `pipeline.py` receives only train/val "
            "DataFrames and test SMILES (no Y column). Any attempt to access test "
            "labels or call group.evaluate() directly in pipeline code will crash "
            "the trial.\n"
            "- **No artifact size cap**: code can be any size.\n"
            "- **Local execution**: trials run on CPU / single GPU via local "
            "subprocess. No SCHED, no DDP.\n"
        )

    def bind_tools(self) -> list[Any]:
        from agent_core.tools.submit import submit_trial
        from agent_core.tools.code_inspect import syntax_check
        from agent_core.tools.workdir import read_snapshot, diff_snapshots, rebase_to
        tools = [syntax_check, read_snapshot, diff_snapshots, rebase_to, submit_trial]
        import os
        mode = os.environ.get("HARNESS_ABLATION_MODE", "feature_only")
        if mode in ("data_only", "joint"):
            from multi_agent_molnet.tools.write_external_data import write_external_data
            tools.append(write_external_data)
        return tools

    # ── Bootstrap text (MoleculeNet; data layer/calibrate don't read these) ─────

    @property
    def bootstrap_hypothesis(self) -> str:
        return ("MapLight features (Morgan+Avalon counts + ErG + 200 RDKit desc, 2563d) "
                "+ default CatBoost baseline; MoleculeNet recommended splits")

    @property
    def baseline_note(self) -> str:
        return "MapLight CatBoost baseline reference (10 MoleculeNet endpoints)"


__all__ = ["MolNetTaskAdapter"]
