# Drug Discovery Auto Research — Task Overview

## Task definition

Maximise `aggregate_score`: normalised improvement over baseline, averaged
across 22 TDC ADMET tasks. Higher is better. Zero = same as baseline.

**Reward signal**: internal scaffold-split validation (from TDC `train_val`).
TDC test labels are NEVER accessible in the agent loop.

## Editable files

```
experiment.py          root seed (orchestrates pipeline/)
pipeline/features.py   RDKit descriptors + ECFP fingerprints
pipeline/models.py     XGBoost / backbone definitions
pipeline/calibration.py post-hoc calibration
pipeline/pipeline.py   DrugPipeline.fit() / predict() interface
```

**DO NOT edit**: `run_trial_drug.py`, `run_classify.py`.

## TDC ADMET — 22 tasks

| # | Task name | Type | Endpoint | Key mechanism |
|---|-----------|------|----------|---------------|
| 1 | caco2_wang | regression | Caco-2 permeability | Membrane permeability; lipophilicity, low MW |
| 2 | hia_hou | classification | Human Intestinal Absorption | Passive diffusion; lipophilicity, MW, H-bond donors |
| 3 | pgp_broccatelli | classification | P-glycoprotein inhibition | Efflux transporter; lipophilicity, MW, aromatic rings |
| 4 | bioavailability_ma | classification | Oral bioavailability | Composite (absorption + first-pass); Lipinski rules |
| 5 | lipophilicity_astrazeneca | regression | LogD (lipophilicity) | Direct logP/logD measurement |
| 6 | solubility_aqsoldb | regression | Aqueous solubility | Lipophilicity, crystal packing; polar surface area |
| 7 | bbb_martins | classification | Blood–Brain Barrier | MW<450, PSA<90, lipophilicity, H-bond donors<3 |
| 8 | ppbr_az | regression | Plasma protein binding | Lipophilicity, charge, acidic groups |
| 9 | vdss_lombardo | regression | Volume of distribution | Lipophilicity, basic nitrogen, plasma binding |
| 10 | cyp2d6_veith | classification | CYP2D6 inhibition | Basic amine, aromatic ring at ~5Å from N |
| 11 | cyp3a4_veith | classification | CYP3A4 inhibition | Large MW, lipophilic, basic nitrogen |
| 12 | cyp2c9_veith | classification | CYP2C9 inhibition | Acidic group, lipophilic, specific aromatic pattern |
| 13 | cyp2d6_substrate_carbonmangels | classification | CYP2D6 substrate | Basic N ~5Å from site of metabolism |
| 14 | cyp3a4_substrate_carbonmangels | classification | CYP3A4 substrate | Large flexible, multiple H-acceptors |
| 15 | cyp2c9_substrate_carbonmangels | classification | CYP2C9 substrate | Acidic functional group |
| 16 | half_life_obach | regression | Half-life | Metabolic stability, clearance, volume |
| 17 | clearance_microsome_az | regression | Microsomal clearance | CYP metabolism; lipophilicity, MW |
| 18 | clearance_hepatocyte_az | regression | Hepatocyte clearance | Phase I + II metabolism |
| 19 | herg | classification | hERG cardiotoxicity | Lipophilicity, basic N, aromatic rings, cation-π |
| 20 | ames | classification | Ames mutagenicity | Structural alerts: nitro, epoxide, aziridine, aromatic amine |
| 21 | dili | classification | Drug-induced liver injury | Reactive metabolites, mitochondrial liability |
| 22 | ld50_zhu | regression | Acute toxicity (LD50) | Reactive groups, electrophilicity, lipophilicity |

*Verified against TDC ADMET group runtime 2026-05-30. Use `group.dataset_names` to confirm at runtime.*

## Baseline stack

- **Features**: RDKit 2D physchem (14 descriptors) + ECFP4 Morgan (1024 bit)
- **Model**: XGBoost (300 trees, lr=0.05, max_depth=6)
- **Calibration**: none
- **Split**: scaffold split (80/20 from train_val)

## Aggregate score formula

```
For classification tasks (AUROC, higher better):
  norm_i = (auroc_i - base_auroc_i) / |base_auroc_i|

For regression tasks (MAE, lower better):
  norm_i = (base_mae_i - mae_i) / |base_mae_i|

aggregate_score = mean(norm_i)  over tasks with valid baseline
```

Positive = better than baseline. Zero = same. Negative = worse.

## Task-specific feature optimization (anti-seesaw design)

The harness evaluates all 22 tasks per trial. The aggregate_score is their
mean. To avoid the seesaw problem (improving task A while hurting task B),
use the **task-conditional feature pattern** in `get_task_features()`:

```python
# pipeline/features.py — get_task_features()

# FEATURE: herg_charge | endpoint: herg | source: Cavalli 2002
if task_name in ('herg',):
    feats = compute_charge_features(df)   # shape (n, k)
    parts.append(feats)

# FEATURE: solubility_polarity | endpoint: solubility_aqsoldb | source: Yalkowsky
if task_name in ('solubility_aqsoldb',):
    feats = compute_polarity_features(df)
    parts.append(feats)
```

**Why this is safe**: each `if task_name in (...)` block is physically absent
from every other task's feature matrix. Improving hERG has ZERO effect on
solubility or CYP.

**When to use universal features** (`get_rdkit_descriptors()` / `get_morgan_fingerprint()`):
only when you believe the feature helps the MAJORITY of tasks. Universal
changes carry seesaw risk.

**Workflow for targeted improvement**:
1. Read lineage / LESSONS.md to identify which tasks are underperforming.
2. Add a task-conditional block in `get_task_features()` for that task.
3. Submit. Only that task's score changes.

**How to identify weak tasks**: after each trial, `per_task` in the lineage
shows individual `val_metric` and `norm_improvement` for all 22 tasks.
`meta` reads these and writes target directives to LESSONS.md.

## data_only mode: external training data

When `HARNESS_ABLATION_MODE=data_only`, all pipeline code is frozen.
The only editable surface is `external_data/{task_name}.csv` written via
the `write_external_data` tool.

**🚫 NO TEST LEAKAGE (most important rule).** Every TDC task is a held-out split
of a specific *source* dataset. Re-adding that source (or a re-aggregation/copy of
it) leaks the test set. The harness enforces a **leakage-safe filter you cannot
bypass**, and reports the outcome in `per_task[task]["data_aug"]["verdict"]`:
1. **Identity dedup** — removes external rows matching TDC test / val / train by
   *standardized* InChIKey (desalt + neutralize, so salt/charge variants match).
2. **Same-source rejection** — if **>5%** of a task's TDC *test* molecules appear
   in your file (InChIKey skeleton), the **WHOLE file is rejected** (0 rows used,
   `verdict='rejected_same_source'`). This blocks the benchmark's own source.
3. **Analog filter** — removes near-duplicates (ECFP4 Tanimoto ≥ 0.9 to any test
   molecule).
Also: winsorizes regression Y (1st–99th pct), max 5000 rows/task.

**AVOID each task's original source; seek an INDEPENDENT assay**:
| Task | Original source — DO NOT re-add | Independent alternatives to seek |
|---|---|---|
| half_life_obach | Obach 2008 (also inside PKSmart) | a different clinical-PK cohort |
| vdss_lombardo | Lombardo 2018 (also inside PKSmart) | independent Vdss measurements |
| ames | Hansen mutagenicity benchmark | a separate Ames screen |
| solubility_aqsoldb | AqSolDB | a distinct solubility assay |
| caco2_wang | Wang 2016 | another Caco-2 permeability set |
| hERG/DILI/CYP/… | (check the TDC source) | ChEMBL/PubChem assays not derived from it |
General hunting grounds: ChEMBL assays, PubChem BioAssay, ExCAPE-DB — but confirm
the assay is **not** the task's TDC source.

**After every trial, read `per_task[task]["data_aug"]`**:
```json
{"source": "...", "verdict": "accepted | rejected_same_source | accepted_empty | error",
 "agent_note": "<plain-English what happened / what to do next>",
 "external_rows_raw": N, "test_overlap_rate": 0.xx,
 "dropped_test_exact": N, "dropped_val_exact": N, "dropped_train_dup": N,
 "dropped_analog": N, "merged_rows": N}
```
If `verdict='rejected_same_source'`, that source is leakage — **do NOT resubmit it
or any copy**; switch to a genuinely independent dataset for that task.

## Hard limits

- Internal validation only during agent loop. No TDC test access.
- Never import `tdc` in pipeline code.
- Never modify `run_trial_drug.py` or `run_classify.py`.
- Wall time per trial: **3600s total for all 22 tasks combined** (~163s average per task).
  Each task's fit subprocess is capped at 300s. Features that are O(n²) in molecule
  count, or that re-parse SMILES repeatedly, will cause large-dataset tasks
  (cyp2d6_veith ~12k mols, solubility ~8k mols) to timeout and score 0.0.
  **Keep all feature computation O(n): one pass over molecules, no nested loops.**
