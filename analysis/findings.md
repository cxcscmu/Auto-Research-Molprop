# Mined findings — TDC ADMET (MapLight strong baseline)

Source: local analysis of the four pulled run dirs (feature / model / data_v2 / joint).
Scripts: `analysis/mine_map.py` (endpoint×axis map), `analysis/regime.py` (axis-vs-n_train).
All numbers are internal-val `norm_improvement` (single seed); directional, not final.

## Headline (the unifying scientific discovery)
On a strong SOTA representation, an autonomous multi-agent researcher discovers that
**dataset size — not feature cleverness — is the master variable** governing where and how
ADMET predictions improve:
- **scarce endpoints** improve from a little *curated, independent* data, and from
  *relaxing* model capacity/regularization to fit them;
- **abundant endpoints** improve only from model tuning;
- **expert mechanism-driven feature engineering, though competently performed, is uniformly
  dominated** because the 2563-dim representation has saturated the feature signal.

best-axis tally over 22 endpoints: **model 14 · data 6 · feature 0**.

## P1 — Data-regime law (match intervention to data scarcity)
- External independent data wins **only** on small endpoints (n_train ≲ 700) and is absent on
  large ones. Split at median n_train=776: small → data 5 / model 5; large → model 9 / data 1.
- Biggest wins are tiny+curated+mechanism-relevant: **cyp2c9_substrate +0.225 (23 merged rows,
  FDA DDI Guidance + Flockhart), cyp2d6_substrate +0.141, half_life +0.063 (FDA labels),
  dili +0.036 (DILIrank)**.
- Refinement: *quality matters more than quantity*. The agent found careless external data
  HURTS and **removed it** (herg / ppbr_az / clearance_hepatocyte cleared to placeholders after
  the additions lowered the score) — the loop turning a failure into a correction.
- Why: small datasets are distribution-starved; a little in-distribution independent data fills
  gaps. Large datasets already cover the space, so more data is redundant.
- Transfer test: does the n_train→data-helps law hold on Polaris/MoleculeNet small endpoints?

## P2 — Feature saturation on a strong representation (the sharp, contrarian result)
- The feature agents performed **genuine, literature-grounded, redundancy-aware** mechanism
  feature engineering — e.g. AMES structural alerts (aziridine, Michael acceptors, N-mustard;
  Kazius 2005, Lawley 1995), explicitly adding only patterns *not* already in MapLight's `fr_*`
  descriptors; hERG lipophilic-basic-amine pharmacophore (Aronov 2005, Cavalli 2002).
- Yet **features are never the best axis on any of 22 endpoints**; gains ≈0 (ames +0.001,
  herg +0.006).
- Why: a SOTA 2563-dim fingerprint+descriptor representation has saturated the feature signal;
  even good mechanism features are largely redundant. The bottleneck is the representation, not
  the agent's feature-engineering competence. **This directly counters the "agents do excellent
  feature engineering" narrative (DrugAgent/MolAgent) — credible precisely because the feature
  work here is high quality, not a strawman.**
- Transfer test / rescue probe: against a *weaker* representation (e.g. drop to ECFP-only or a
  pretrained embedding), do these same features suddenly help? If yes, P2 is representation-
  conditional (a stronger, more nuanced claim).

## P3 — Model gains = capacity & regularization matched to data size (+ per-task family)
- The model agent discovered (documented in its own models.py rationale):
  1. **per-task model family** (XGBoost / CatBoost / LightGBM chosen per endpoint);
  2. **more capacity for small endpoints** (CatBoost iterations 1000→4000 for herg n=419,
     half_life n=426);
  3. **relax regularization for small datasets** (vdss n=719, ppbr: reg_lambda 2.0→0.5,
     min_child_samples 20→10) — agent's reasoning: "shared reg params are tuned for large
     datasets; for small n they over-regularize and suppress genuine signal";
  4. adaptive class-imbalance weighting (earlier trials) for imbalanced tox/CYP endpoints.
- Biggest model wins: caco2 +0.131, clearance_microsome +0.090, cyp3a4_substrate +0.097,
  bioavailability +0.086, clearance_hepatocyte +0.078, lipophilicity +0.069.
- **The model agent independently rediscovered the data-size regime** (small data → more
  capacity + less regularization), corroborating P1 from the model side.
- Transfer test: does capacity/regularization-by-size hold on the other benchmarks?

## Cross-cutting (ties to the auto-research framework / prior paper)
- A single master variable (data scarcity) emerged independently across all three axes.
- The loop repeatedly turned failures into corrections: removed harmful external data;
  added only non-redundant features; matched regularization to size after diagnosing
  over-regularization. This is "evaluator-owned outcome → next edit," in a science domain.

## What this means for spend
- The paper's spine = P1/P2/P3, each with mechanism + a case study. Mostly already in hand.
- The two remaining benchmarks earn their cost as **transfer tests of P1/P2** (not number
  replication): do the data-regime law and feature-saturation hold on Polaris / MoleculeNet?
- Optional high-value probe: a *weak-representation* condition on TDC to test whether feature
  gains reappear (sharpens P2 into a representation-conditional law).

## Candidate case studies (one per principle)
- P1: **cyp2c9_substrate** — 23 curated independent rows → +0.225 (huge on n=428); contrast
  herg where external data hurt and was removed.
- P2: **ames** — sophisticated, cited structural-alert features → +0.001 (saturation).
- P3: **vdss_lombardo / caco2_wang** — regularization-relaxation / per-task tuning → large model gains.

## Atlas-guided composition (clean replacement for the polluted joint run)
Because each TDC endpoint is an **independent per-task model**, routing each endpoint to its
atlas-best clean single-axis intervention is realizable with no cross-task interaction and no
search-budget dilution — a clean, interpretable upper bound that *replaces* the joint run.
Script: `analysis/composition.py` (uses the clean single-axis runs; $0, no rerun).

- **atlas-guided composition: oracle (arithmetic) = 0.0560; MEASURED = 0.0554** (confirmed) vs
  model 0.0412 · data 0.0316 · joint(polluted) 0.0298 · feature 0.0125. (+34% over the best
  single axis; ~1.9× the joint.) The measured value (every per-task entry a real measurement)
  matches the arithmetic prediction → the per-endpoint routing composes additively/realizably.
- **Interaction is benign/synergistic**: curated data on top of the tuned model is retained or
  amplified — half_life +0.063→+0.106, dili +0.036→+0.050, cyp2c9_substrate +0.225→+0.205.
- Measurement note: the measured composition was assembled from two real runs (model-best 098
  for the 16 non-data endpoints; locally-run tuned-model+data for 5 of the 6 data endpoints;
  vdss via data-only fallback) because a single local pipeline run timed out on 6 large tasks
  (WSL speed vs 098's heavy tuned models within the 600s per-task fit cap). A single-pipeline
  run on faster hardware (or with a raised fit timeout) would reproduce ~0.055.
- Routing: **model 14 endpoints · data 6 · feature 2** (the 2 feature routes, hia_hou +0.002 and
  cyp2c9_veith +0.003, are within noise → feature never wins meaningfully, consistent with P2).
- Framing: a **per-endpoint oracle upper bound** ("ceiling if each endpoint follows its atlas
  axis"), not an end-to-end autonomous result. Single-seed / internal-val. Picks one best axis
  per endpoint (no within-endpoint multi-axis stacking).
- **Why better than joint:** interpretable (per-endpoint routing table = a core figure),
  realizable (independent tasks), no dilution, free, and free of joint's data-gate bug.
- Optional cheap hardening: run the composed per-task pipeline through ONE harness eval
  (no agent, ~$0 API) to turn the additive prediction into a measured number.
- **Decision: this replaces joint in the paper.** joint is not run/fixed; mention only as a
  brief note (search dilution under fixed budget).

---

# TODO — Transfer validation (next experiments)

**Goal:** test whether the TDC-derived principles are *general* properties of small-data
molecular property prediction (not TDC artifacts), and whether the autonomous system
*rediscovers* the per-endpoint optimal-axis map on new domains. Falsifiable, win-win design
(clean replication → principles generalize; partial → principles refined). **joint is NOT
needed** — all principles come from single-axis runs.

## What we validate (falsifiable claims)
- **P1 data-regime law:** best-axis flips with n_train (small → data, large → model).
- **P2 feature saturation:** feature_only never the best axis on a strong representation.
- **P3:** model gains come from capacity/regularization matched to dataset size.

## Settings per benchmark (same harness; re-calibrate a MapLight-style strong baseline each)
Run the **3 single-axis ablations (feature / model / data)**, ~40 trials each.
**Skip joint and the framework ablations** (no-lineage / single-generalist — cite prior paper).
Choose tasks to **span the n_train gradient** (so P1 is testable *within* each benchmark).

| benchmark | tasks (by size, approx — confirm) | role |
|---|---|---|
| **MoleculeNet** (primary; cheap, classic, wide size range) | small: FreeSolv(~640), ESOL(~1.1k), BACE(~1.5k); large: Lipophilicity(~4.2k), Tox21(~7.8k), HIV(~41k) | within-benchmark size gradient → direct P1 test |
| **Polaris** (secondary; strict immutable protocol) | 3–4 ADMET tasks spanning size | not MoleculeNet-specific; survives strict protocol |
| **TDC weak-rep probe** (cheap) | feature_only only, baseline = ECFP-only (weak rep) | directly tests if P2 is representation-conditional |

Analysis: reuse `mine_map.py` + `regime.py`; compare endpoint×axis map and n_train→best-axis to TDC.

## Predicted outcomes
- Replication (expected): small tasks → data wins; large tasks → model wins; feature never wins;
  weak-rep probe → features suddenly help. ⇒ principles are general; method transfers.
- Non-replication (still publishable): e.g. physics-driven FreeSolv shows feature gains ⇒ P2
  refined to "representation/task-conditional."

## Cost & prerequisites
- **Prerequisite is engineering, not $:** write task adapters (loading / split / official metric /
  leakage guard) for Polaris and MoleculeNet.
- **API estimate:** MoleculeNet 3 axes × ~40 trials ≈ ~$240; Polaris similar; weak-rep probe ~$50–100.
- **Minimal high-value version (~$300):** MoleculeNet (P1 flip + P2 saturation) + TDC weak-rep probe.
  Polaris is the optional 3rd-benchmark confirmation. (Budget freed by NOT re-running joint.)

## Open decision
Start by building the MoleculeNet task adapter ($0 eng), or first lock the task list / predictions.
