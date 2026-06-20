# Polaris adme-fang Auto Research — Task Overview

## Task definition

Maximise `aggregate_score`: normalised improvement over baseline, averaged across
**4 Polaris adme-fang endpoints** (ALL regression). Higher is better. Zero = same as
baseline.

**Reward signal**: internal validation split (carved from `train_val`). The held-out
test labels are NEVER accessible in the agent loop.

## Editable files

```
experiment.py           root seed (orchestrates pipeline/)
pipeline/features.py    MapLight features (Morgan+Avalon counts, ErG, RDKit desc)
pipeline/models.py      CatBoost / backbone definitions (regressors here)
pipeline/calibration.py post-hoc post-processing / ensembling (no prob. calibration — all regression)
pipeline/pipeline.py    DrugPipeline.fit() / predict() interface
```

**DO NOT edit**: `run_trial_drug.py`, `run_classify.py`.

## Polaris adme-fang — 4 endpoints (all REGRESSION, metric = pearson)

Source: **biogen/adme-fang-v1** (Fang et al. 2023, *J. Chem. Inf. Model.*) — an
industrial DMPK dataset. Exported from Polaris Hub to a static CSV. Targets are
log10-transformed (`LOG_*`).

| # | Task name | Type | Metric | Endpoint | Key mechanism / notes |
|---|---|---|---|---|---|
| 1 | adme_hlm  | regression | pearson | Human liver microsomal intrinsic clearance (log CLint) | Metabolic stability; lipophilicity (LogP/LogD), CYP soft spots, MW. ~3087 mols. |
| 2 | adme_rlm  | regression | pearson | Rat liver microsomal intrinsic clearance (log CLint)   | As HLM, rat species; HLM/RLM share metabolic SAR (multitask candidates). ~3054. |
| 3 | adme_mdr1 | regression | pearson | MDR1-MDCK efflux ratio (log ER)                        | P-gp efflux / passive permeability; TPSA, HBD, MW, aromatic rings, charge. ~2642. |
| 4 | adme_solu | regression | pearson | Aqueous solubility (log)                               | LogP, crystal-packing (aromatic stacking/symmetry), TPSA, ionizable groups. ~2173. |

*Confirm at runtime via `group.dataset_names`. All 4 endpoints share ONE scaffold
train/test partition of the 3521-molecule file; each endpoint's label column is then
NaN-dropped on whichever side it fell — so there is no cross-endpoint leakage.*

## Baseline stack (MapLight — strong CPU baseline)

- **Features** (2563-dim, universal): Morgan **count** fingerprint 1024 + Avalon
  **count** fingerprint 1024 + ErG 315 + 200 RDKit 2D descriptors.
- **Model**: default CatBoost **regressor** (MAE loss on a scaled target). No per-task
  tuning in the baseline.
- **Split**: scaffold 80:20 (test = 20%) on the shared molecule set. Internal train/val
  is a further scaffold split carved from train_val. (Fang 2023 used random 80:20; we use
  scaffold for stricter generalization and consistency with our other benchmarks.)

Per-endpoint baseline metrics live in `knowledge/baseline_scores.json` (the
normalisation reference). Sanity targets: Fang 2023 reports a Random-Forest baseline
around R²≈0.38; the Polaris adme-fang leaderboard reports pearson per endpoint — the
MapLight CatBoost baseline should land in a comparable range. Beating it requires
genuine ADME signal.

## Aggregate score formula

```
Regression (pearson, higher better):  norm_i = (pearson_i - base_pearson_i) / |base_pearson_i|
aggregate_score = mean(norm_i)  over the 4 endpoints with a valid baseline
```

Positive = better than baseline. Zero = same. Negative = worse.

## Task-specific feature optimization (anti-seesaw design)

The harness evaluates all 4 endpoints per trial; aggregate_score is their mean. To
avoid the seesaw problem (improving endpoint A while hurting B), use the
**task-conditional feature pattern** in `get_task_features()`:

```python
# pipeline/features.py — get_task_features()

# FEATURE: clearance_lipophilicity | endpoint: adme_hlm | source: CLint-LogP QSAR
if task_name in ('adme_hlm', 'adme_rlm'):
    feats = compute_lipophilicity_features(df)   # shape (n, k)
    parts.append(feats)

# FEATURE: solubility_packing | endpoint: adme_solu | source: Yalkowsky GSE
if task_name in ('adme_solu',):
    feats = compute_packing_features(df)
    parts.append(feats)
```

**Why this is safe**: each `if task_name in (...)` block is physically absent from
every other endpoint's feature matrix. Improving adme_solu has ZERO effect on adme_hlm.

**Universal features** (`get_rdkit_descriptors()` / `get_morgan_fingerprint()`): change
only when you believe a feature helps the MAJORITY of endpoints — universal changes
carry seesaw risk. (LogP/TPSA/MW are plausibly broadly useful across all four.)

**How to find weak endpoints**: after each trial, `per_task` shows each endpoint's
`val_metric` (pearson) and `norm_improvement`; `meta` reads these and writes directives.

## data_only mode: external training data

When `HARNESS_ABLATION_MODE=data_only`, all pipeline code is frozen. The only editable
surface is `external_data/{task_name}.csv` written via the `write_external_data` tool.

**🚫 NO TEST LEAKAGE (most important rule).** The adme-fang endpoints are held-out
splits of the **Biogen / Fang 2023 DMPK release**. Re-adding that release (or a
re-aggregation/copy) leaks the test set. The harness enforces a **leakage-safe filter
you cannot bypass**, reporting the outcome in `per_task[task]["data_aug"]["verdict"]`:
1. **Identity dedup** — removes external rows matching test/val/train by *standardized*
   InChIKey (desalt + neutralize).
2. **Same-source rejection** — if **>5%** of an endpoint's test molecules appear in your
   file, the **WHOLE file is rejected** (`verdict='rejected_same_source'`).
3. **Analog filter** — removes near-duplicates (ECFP4 Tanimoto ≥ 0.9 to any test molecule).
Also winsorizes regression Y (1st–99th pct), max 5000 rows/endpoint.

**AVOID the Biogen/Fang source; seek an INDEPENDENT assay**:
| Endpoint | Original source — DO NOT re-add | Independent alternatives to seek |
|---|---|---|
| adme_hlm / adme_rlm | Biogen/Fang 2023 microsomal CLint | non-Biogen ChEMBL human/rat liver-microsome CLint or t½ assays |
| adme_mdr1 | Biogen/Fang MDR1-MDCK | independent MDR1-MDCK / Caco-2 efflux-permeability assays |
| adme_solu | Biogen/Fang solubility | independent kinetic/thermodynamic aqueous-solubility sets |
General hunting grounds: ChEMBL, PubChem BioAssay — confirm the assay is **not** the
Biogen/Fang release. If `verdict='rejected_same_source'`, do NOT resubmit that source.

## Hard limits

- Internal validation only during the agent loop. No test access.
- Never import `tdc`, `deepchem`, or `polaris` in pipeline code (all blocked at runtime).
- Never modify `run_trial_drug.py` or `run_classify.py`.
- Wall time per trial: **3600s for all 4 endpoints combined**. Each endpoint's fit
  subprocess is capped at **600s**. The adme-fang endpoints are small (~2–3k molecules
  each), so fits are fast — but keep all feature computation **O(n)**: one pass over
  molecules, no nested loops, no repeated SMILES re-parsing.
