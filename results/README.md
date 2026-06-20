# Results

Frozen results behind every number, table, and figure in the paper. Nothing here
requires an API key, model training, or a GPU.

## Contents

### `heldout/` - frozen per-endpoint scores
One JSON per (suite, axis). Each holds the per-endpoint internal-validation and
held-out-test scores for the best-aggregate configuration of that axis, scored
against the strong baseline.

- `{tdc,molnet,polaris}_test_baseline.json` - the strong MapLight-style baseline.
- `{tdc,molnet,polaris}_test_{feature,model,data}.json` - the three intervention axes.
- `{tdc,molnet,polaris}_test_combined.json` - the spliced multi-axis configuration.
- `tdc_test_le30_*.json` - the matched 30-trial budget variants used in the transfer study.
- `tdc_official_test_results.json` - TDC official-split re-evaluation.
- `{tdc,molnet,polaris}_unimol_official_test.json` - the Uni-Mol pretrained-3D held-out scores.
- `{tdc,molnet,polaris}_unimol_val.json` - the Uni-Mol internal-validation scores.
- `polaris_flaml_modelaxis*.json` - the matched-trial FLAML AutoML control.
- `tdc_traj_*.json` - best-so-far search trajectories for the trajectory figure.

This folder is the single source of truth for every frozen score; the generators
in `../analysis/` read from here.

The routed held-out gains (+0.013 TDC, +0.011 MoleculeNet, +0.042 Polaris), the two
non-transfer signatures (TDC model 0.041 -> 0.003; Polaris data 0.022 -> -0.019),
and all per-endpoint numbers are computed from these files.

### `tables/` - reproducible result documents
- `test_val_results.md` - validation vs held-out-test, per axis per suite, plus routed.
- `case_study.md` - the data-axis leakage audit (accepted and rejected external
  sources, same-source overlap rates, the necessary-not-sufficient finding).
- `automl_comparison.md` - the matched-trial AutoML control (FLAML 0.006 vs the
  agent's 0.042 on the Polaris model axis).

### `figures/`
`fig_main_teaser` (overview), `fig_transfer_map` (validation-to-test transfer per
axis), `fig_endpoint_transfer` (all 36 endpoints), `fig_data_audit` (external-data
audit), `fig_signatures` (the two non-transfer signatures), `fig_trajectory`
(best-so-far search trajectories). Each ships as PDF (vector) and PNG (for inline
viewing), except `fig_main_teaser`, which is PNG only.

## Regenerating

The documented generators live in `../analysis/` and read `heldout/` and write
`tables/`/`figures/` directly — run each from the repository root with `.venv_drug`:

```bash
MPLCONFIGDIR=/tmp/mpl .venv_drug/bin/python analysis/gen_results_md.py        # tables/test_val_results.md
MPLCONFIGDIR=/tmp/mpl .venv_drug/bin/python analysis/gen_automl_md.py         # tables/automl_comparison.md
MPLCONFIGDIR=/tmp/mpl .venv_drug/bin/python analysis/make_transfer_figures.py # figures/fig_transfer_map, fig_endpoint_transfer
MPLCONFIGDIR=/tmp/mpl .venv_drug/bin/python analysis/make_signature_figure.py # figures/fig_signatures
```

These rebuild `test_val_results.md`, `automl_comparison.md`, `fig_transfer_map`,
`fig_endpoint_transfer`, and `fig_signatures` from the frozen scores in `heldout/`
alone (no API key, training, or GPU). They only need `numpy`, `pandas`,
`matplotlib`, and the per-suite providers (for metric direction); the providers are
imported but no benchmark data is downloaded.

Some artifacts are **not** rebuilt from the frozen scores and are shipped as static
outputs because they need the data-axis agent run logs (the `*_dev_*` lineage
snapshot trees, large and not redistributed): `case_study.md` (generator
`data_audit_table.py`) and the left panel of `fig_data_audit` (in
`make_transfer_figures.py`). Both generators detect the absent logs and skip without
overwriting the shipped file. `fig_trajectory` and `fig_main_teaser` are produced by
separate workspace scripts. The remaining files in `analysis/` are the broader
analysis workspace (sweeps, consolidation, exploratory scripts) and are not required
to rebuild the paper's figures and tables.
