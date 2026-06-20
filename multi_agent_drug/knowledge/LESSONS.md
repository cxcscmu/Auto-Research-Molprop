# Lessons learned

---

## 2026-06-02 — meta analyst: timing fixes + task unlocking (exp_047, exp_052 crash logs)

### Current best: exp_038 (agg=0.012443, n_tasks_ok=6/22)

Only 6 of 22 tasks complete. Root cause: lipo/sol/ppbr timeout, CYP/hERG/AMES/etc skip due to wall budget.

### Confirmed timing facts (from local_run.log of crashed trials)

| Task | n_train | FP bits tried | Timing | Status |
|------|---------|---------------|--------|--------|
| caco2_wang | 583 | 1024 | 49s (normal) / 170s (loaded) | OK but variable |
| hia_hou | 369 | 1024 | 14s / 28s | OK |
| pgp_broccatelli | 778 | 1024 | 19s / 27s | OK |
| bioavailability_ma | 410 | 1024 | 17s / 37s | OK |
| lipophilicity_astrazeneca | 3360 | **512** | 86s / 107s | **OK with 512-bit** |
| solubility_aqsoldb | 7986 | **256** | 277s@512/82s@256 | **OK with 256-bit** |
| bbb_martins | 1250 | 1024 | 50s | OK (5-feature) |
| ppbr_az | 2229 | **256** | 302s@512/228s@256 | **OK with 256-bit** |
| vdss_lombardo | 719 | 1024 | 83-121s | OK + logP²+basic_n block works |
| cyp2d6_veith | ~12k | 512 | 305s CRASH | **Needs 256-bit** |
| cyp3a4_veith | ~12k | 512 | 310s CRASH | **Needs 256-bit** |
| cyp2c9_veith | ~12k | 512 | 307s CRASH | **Needs 256-bit** |

### DIRECTIVE G (HIGH PRIORITY): move CYP inhibitors to 256-bit in featurize()

In `pipeline/features.py`, `featurize()` function, change `_XL_TASKS` to include CYPs:
```python
_XL_TASKS = {  # 256-bit fingerprints
    'solubility_aqsoldb', 'ppbr_az',
    'cyp2d6_veith', 'cyp3a4_veith', 'cyp2c9_veith',  # ← ADD THESE
}
_LARGE_TASKS = {  # 512-bit fingerprints
    'lipophilicity_astrazeneca', 'herg', 'ames', 'ld50_zhu',
}
# everything else: 1024-bit
```
This should bring CYP inhibitor tasks from 305s CRASH → ~158s OK.
**Parent**: exp_038 (or any recent kept trial). The workdir_meta already has this tiering structure.

### DIRECTIVE H (if G alone insufficient): remove MaxPartialCharge universally

`get_rdkit_descriptors()` computes Gasteiger charges (MaxPartialCharge, MinPartialCharge).
For complex molecules, this takes ~75-100ms/mol. For ppbr (n=2229 complex drugs): 223s just from Gasteiger!
Removing these two descriptors would save 150-400s of wall budget across all tasks.
**Accuracy risk**: Low. hERG/DILI already have task-specific features (formal_chg, basic_n, nitro alerts).

### CONFIRMED: bbb_martins 7-feature upgrade hurts accuracy

In exp_047, adding pass_hba + pass_nar to bbb_martins block: AUROC 0.8845 → 0.8401 (BAD!).
**NEVER add more features to bbb_martins.** Keep it at 5 features: pass_mw, pass_psa, pass_hbd, pass_logp, cns_score.

### CONFIRMED: vdss_lombardo logP²+basic_n block improves accuracy

In both exp_047 and exp_052: vdss val_metric=4.9553 vs exp_038's 4.9654 (improvement).
This block is correct and should be kept.

### Task-specific features ready and waiting (will activate once those tasks unlock)

- hERG: logp + n_ar + basic_n + formal_chg + logp_x_ar
- AMES: 7 SMARTS alerts (nitro, aryl amine, epoxide, nitroso, hydrazine, Michael, aryl halide)
- DILI: logp + mw + basic_n + nitro + aryl amine + epoxide
- CYP inhibitors (×3): basic_n + n_ar + logp + mw
- CYP2C9 substrates: COOH + sulfonamide SMARTS
- clearance (×2): logp + mw + basic_n + fsp3
- half_life: logp + mw + basic_n + fsp3 + n_ar
- LD50: logp + mw + basic_n + Michael + epoxide + nitro
