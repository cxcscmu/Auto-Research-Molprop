# Current SOTA Stack

## Baseline (exp_000)

| Component | Description |
|-----------|-------------|
| Features  | RDKit 14 physchem descriptors + ECFP4 Morgan fingerprint (1024 bit) |
| Imputation | Column median for NaN |
| Model | XGBoost: n_estimators=300, lr=0.05, max_depth=6, subsample=0.8 |
| Calibration | None (identity) |
| Split | Scaffold split 80/20 from TDC train_val, seed=42 |
| aggregate_score | Calibrated by calibrate_baseline.py on first run |

## Known limitations of baseline

- Generic features: same 14 descriptors for all 22 endpoints regardless of mechanism
- No endpoint-specific structural alerts (e.g. no AMES-specific nitro/epoxide features)
- No task-specific calibration (class imbalance uncorrected for DILI/AMES)
- XGBoost default hyperparams not tuned per task

## Directions with highest expected gain

1. **Endpoint-specific physchem features** (fphs): add mechanism-informed
   descriptors (e.g. charge-related for hERG, PSA for BBB, planar area for CYPs)
2. **Structural alerts for AMES/hERG** (fsub): SMARTS-based alert features
3. **Class imbalance handling** (data/modl): DILI and AMES have <10% positive rate
4. **LightGBM / RF comparison** (modl): often faster + better on small tabular data
