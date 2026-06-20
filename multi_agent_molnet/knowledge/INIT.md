# MoleculeNet Auto Research — Task Overview

## Task definition

Maximise `aggregate_score`: normalised improvement over baseline, averaged across
**10 MoleculeNet endpoints**. Higher is better. Zero = same as baseline.

**Reward signal**: internal validation split (carved from `train_val`). MoleculeNet
test labels are NEVER accessible in the agent loop.

## Editable files

```
experiment.py           root seed (orchestrates pipeline/)
pipeline/features.py    MapLight features (Morgan+Avalon counts, ErG, RDKit desc)
pipeline/models.py      CatBoost / backbone definitions
pipeline/calibration.py post-hoc calibration / thresholding
pipeline/pipeline.py    DrugPipeline.fit() / predict() interface
```

**DO NOT edit**: `run_trial_drug.py`, `run_classify.py`.

## MoleculeNet — 10 endpoints

| # | Task name | Type | Metric | Endpoint | Key mechanism / notes |
|---|---|---|---|---|---|
| 1 | freesolv | regression | RMSE | Hydration free energy (ΔG) | Solvation thermodynamics; H-bonding, polarity. Smallest set (~640). |
| 2 | esol | regression | RMSE | Aqueous solubility (logS) | Lipophilicity, MW, aromatic proportion, TPSA. |
| 3 | bace | classification | ROC-AUC | BACE-1 inhibition | Aspartyl-protease pharmacophore; ~balanced. |
| 4 | hiv | classification | ROC-AUC | HIV replication inhibition | Largest (~41k); highly imbalanced (~3.5% positive). |
| 5 | tox21_nr_ar | classification | ROC-AUC | Androgen receptor (Tox21 NR) | Nuclear-receptor binding toxicophore; imbalanced (~4%). |
| 6 | tox21_sr_mmp | classification | ROC-AUC | Mitochondrial membrane-potential stress | Electrophiles / redox-active groups; ~16% positive. |
| 7 | tox21_sr_p53 | classification | ROC-AUC | p53 DNA-damage stress response | Genotoxic / electrophilic alerts; ~6%. |
| 8 | sider_hepatobiliary | classification | ROC-AUC | Hepatobiliary-disorder side effect | Marketed-drug ADR; ~balanced. |
| 9 | sider_reproductive | classification | ROC-AUC | Reproductive-system side effect | Marketed-drug ADR; ~balanced. |
| 10 | clintox_ct_tox | classification | ROC-AUC | Clinical-trial toxicity | Failed-for-tox vs approved; ~7.5%. |

*Confirm at runtime via `group.dataset_names`. Multi-label sources (Tox21/SIDER/ClinTox)
are split into single-label sub-tasks; molecules share ONE train/test partition per
source file, so there is no cross-assay leakage.*

## Baseline stack (MapLight — strong CPU SOTA)

- **Features** (2563-dim, universal): Morgan **count** fingerprint 1024 + Avalon
  **count** fingerprint 1024 + ErG 315 + 200 RDKit 2D descriptors.
- **Model**: default CatBoost (classification: Logloss; regression: MAE loss on a
  scaled target). No per-task tuning in the baseline.
- **Split**: MoleculeNet-recommended per endpoint — ESOL/BACE/HIV scaffold;
  FreeSolv/Tox21/SIDER/ClinTox random — 80/10/10 (test = 10%). Internal train/val is a
  scaffold split carved from train_val.

Per-endpoint baseline metrics live in `knowledge/baseline_scores.json` (the
normalisation reference). This MapLight baseline already matches or beats the published
MoleculeNet CPU results on 9/10 endpoints — beating it requires genuine signal.

## Aggregate score formula

```
Classification (ROC-AUC, higher better):  norm_i = (auroc_i - base_auroc_i) / |base_auroc_i|
Regression     (RMSE, lower better):       norm_i = (base_rmse_i - rmse_i) / |base_rmse_i|
aggregate_score = mean(norm_i)  over endpoints with a valid baseline
```

Positive = better than baseline. Zero = same. Negative = worse.

## Task-specific feature optimization (anti-seesaw design)

The harness evaluates all 10 endpoints per trial; aggregate_score is their mean. To
avoid the seesaw problem (improving endpoint A while hurting B), use the
**task-conditional feature pattern** in `get_task_features()`:

```python
# pipeline/features.py — get_task_features()

# FEATURE: tox21_electrophile | endpoint: tox21_sr_mmp | source: structural alerts
if task_name in ('tox21_sr_mmp', 'tox21_sr_p53'):
    feats = compute_electrophilicity_features(df)   # shape (n, k)
    parts.append(feats)

# FEATURE: solubility_polarity | endpoint: esol | source: Yalkowsky GSE
if task_name in ('esol',):
    feats = compute_polarity_features(df)
    parts.append(feats)
```

**Why this is safe**: each `if task_name in (...)` block is physically absent from
every other endpoint's feature matrix. Improving esol has ZERO effect on hiv or tox21.

**Universal features** (`get_rdkit_descriptors()` / `get_morgan_fingerprint()`): change
only when you believe a feature helps the MAJORITY of endpoints — universal changes
carry seesaw risk.

**How to find weak endpoints**: after each trial, `per_task` shows each endpoint's
`val_metric` and `norm_improvement`; `meta` reads these and writes target directives.

## data_only mode: external training data

When `HARNESS_ABLATION_MODE=data_only`, all pipeline code is frozen. The only editable
surface is `external_data/{task_name}.csv` written via the `write_external_data` tool.

**🚫 NO TEST LEAKAGE (most important rule).** Each endpoint is a held-out split of a
specific *source* dataset. Re-adding that source (or a re-aggregation/copy) leaks the
test set. The harness enforces a **leakage-safe filter you cannot bypass**, reporting
the outcome in `per_task[task]["data_aug"]["verdict"]`:
1. **Identity dedup** — removes external rows matching test/val/train by *standardized*
   InChIKey (desalt + neutralize).
2. **Same-source rejection** — if **>5%** of an endpoint's test molecules appear in your
   file (InChIKey skeleton), the **WHOLE file is rejected** (`verdict='rejected_same_source'`).
3. **Analog filter** — removes near-duplicates (ECFP4 Tanimoto ≥ 0.9 to any test molecule).
Also winsorizes regression Y (1st–99th pct), max 5000 rows/endpoint.

**AVOID each endpoint's original source; seek an INDEPENDENT assay**:
| Endpoint | Original source — DO NOT re-add | Independent alternatives to seek |
|---|---|---|
| esol | Delaney / AqSolDB | a distinct aqueous-solubility assay |
| freesolv | SAMPL / FreeSolv | independent hydration-energy calc/exp |
| bace | the BACE-1 benchmark set | a separate BACE-1 inhibition assay |
| hiv | NCI DTP AIDS Antiviral screen | independent antiviral activity data |
| tox21_* | Tox21 Data Challenge | a non-Tox21 assay for that target |
| sider_* | SIDER database | independent pharmacovigilance / FAERS-derived |
| clintox_ct_tox | FDA / ClinicalTrials.gov failures | independent clinical-tox outcomes |
General hunting grounds: ChEMBL, PubChem BioAssay, ExCAPE-DB — confirm the assay is
**not** the endpoint's MoleculeNet source. If `verdict='rejected_same_source'`, do NOT
resubmit that source or any copy.

## Hard limits

- Internal validation only during the agent loop. No MoleculeNet test access.
- Never import `tdc` or `deepchem` in pipeline code (both are blocked at runtime).
- Never modify `run_trial_drug.py` or `run_classify.py`.
- Wall time per trial: **3600s for all 10 endpoints combined**. Each endpoint's fit
  subprocess is capped at **600s** (HIV ~41k mols takes ~305s). Features that are O(n²)
  in molecule count, or re-parse SMILES repeatedly, will make large endpoints (hiv ~41k,
  tox21 ~6k) timeout and score 0.0. **Keep all feature computation O(n): one pass over
  molecules, no nested loops.**
