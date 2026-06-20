"""Generate results/tables/automl_comparison.md — the non-agent AutoML (FLAML)
control on the model axis, for the benchmark where the agent's model-axis gain is
largest (Polaris).

Reads results/heldout/polaris_test_{baseline,model}.json + polaris_flaml_modelaxis{,_scaled}.json
(+ polaris_unimol_official_test.json for context). Numbers are held-out-test Pearson,
normalised vs the baseline on the same split. Re-runnable. Run with .venv_drug."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import test_eval as te  # noqa: E402

A = ROOT / "results" / "heldout"
OUT = ROOT / "results" / "tables" / "automl_comparison.md"


def load(name):
    p = A / name
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    prov = te.get_provider("polaris")
    B = load("polaris_test_baseline.json")
    M = load("polaris_test_model.json")
    Fr = load("polaris_flaml_modelaxis.json")
    Fs = load("polaris_flaml_modelaxis_scaled.json")
    U = load("polaris_unimol_official_test.json")

    L = []
    w = L.append
    w("# Non-agent AutoML control on the model axis\n")
    w("Is the agent's **model-axis** gain something a standard, off-the-shelf AutoML "
      "would also find — i.e. is it \"just hyperparameter search\"? We test this on "
      "**Polaris adme-fang**, the benchmark where the model axis is strongest "
      "(its largest held-out-test gain of any axis on any suite).\n")
    w("## Setup — matched to the agent's model axis\n")
    w("- **AutoML**: FLAML 2.6.0, `estimator_list = {lgbm, xgboost, catboost}` (the "
      "MapLight model family), `max_iter = 30` (matched to the agent's ~25 model-axis "
      "trials on Polaris), model selected on the **same** internal-validation split "
      "(holdout), optimising the **same** metric (Pearson).")
    w("- **Same everything else**: identical frozen MapLight features (2563-dim), "
      "identical scaffold split (frac 0.20 / seed 42), identical official test.")
    w("- **agent model axis**: the LLM edits `models.py`; here it wrote a multi-seed "
      "CatBoost ensemble (per-seed prediction averaging, variance reduction) on top of "
      "the baseline's scaled-target CatBoost, kept over ~25 trials by validation.")
    w("- **FLAML-raw vs FLAML-fair**: `raw` trains on the raw target; `fair` applies the "
      "**same target transform** the baseline and agent use (log1p for positive skewed "
      "targets, then z-score). The fair column removes any handicap from preprocessing, "
      "isolating the model+HP **search** itself.\n")

    w("## Held-out test (Pearson) — Polaris adme-fang\n")
    w("| Endpoint | Baseline | Agent (model) | FLAML-raw | FLAML-fair | Uni-Mol |")
    w("|---|---|---|---|---|---|")
    agg = {"agent": 0.0, "raw": 0.0, "fair": 0.0}
    n = 0
    for t in B:
        m = B[t]["metric"]; bt = B[t]["test"]
        row = [t, f"{bt:.3f}", f"{M[t]['test']:.3f}", f"{Fr[t]['test']:.3f}",
               f"{Fs[t]['test']:.3f}", f"{U[t]['unimol_test']:.3f}" if t in U else "--"]
        w("| " + " | ".join(row) + " |")
        agg["agent"] += prov.normalise(M[t]["test"], bt, m)
        agg["raw"] += prov.normalise(Fr[t]["test"], bt, m)
        agg["fair"] += prov.normalise(Fs[t]["test"], bt, m)
        n += 1

    w("\n### Normalised improvement vs baseline-test (mean over endpoints)\n")
    w("| Method | TEST-norm |")
    w("|---|---|")
    w(f"| Agent (model axis) | **{agg['agent']/n:+.4f}** |")
    w(f"| FLAML (raw target) | {agg['raw']/n:+.4f} |")
    w(f"| FLAML (fair, same target transform) | {agg['fair']/n:+.4f} |")

    w("\n## Finding\n")
    w(f"A standard, matched-trial AutoML **does not reproduce the agent's model-axis "
      f"gain**: the agent reaches **{agg['agent']/n:+.4f}** normalised improvement on the "
      f"held-out test, FLAML only **{agg['fair']/n:+.4f}** — about a {agg['agent']/max(agg['fair'],1e-9):.0f}× "
      "gap. Adding the baseline's target transform (fair column) does **not** close it "
      "(it slightly lowers FLAML, via `adme_rlm`), so the difference is not a preprocessing "
      "artefact. The agent's edge is the **multi-seed ensemble** it wrote: averaging "
      "predictions across seeds reduces variance, whereas FLAML's single-configuration "
      "holdout selection on a small, noisy validation set overfits that split and "
      "generalises worse — the **same winner's-curse mechanism** this paper characterises. "
      "The agent's broader (code-level) search space lets it express this hedge; the AutoML's "
      "fixed numeric search does not.\n")
    w("## Scope and caveats\n")
    w("- This is **off-the-shelf** FLAML at a **matched trial count** (its internal "
      "ensembling option is not enabled and the trial count is not enlarged). The claim is "
      "therefore that *standard, matched-trial AutoML* does not reproduce the gain — **not** "
      "that AutoML could not in principle with heavier configuration.")
    w("- Evaluated on Polaris (4 endpoints, the model-axis stronghold), single seed. The "
      "model axis does not generalise on TDC at all (variance failure), so the comparison "
      "is only meaningful where the model axis itself transfers (Polaris, MoleculeNet).\n")
    w("Generated by `analysis/gen_automl_md.py` (re-runnable).\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines); agent={agg['agent']/n:+.4f} "
          f"flaml_raw={agg['raw']/n:+.4f} flaml_fair={agg['fair']/n:+.4f}")


if __name__ == "__main__":
    main()
