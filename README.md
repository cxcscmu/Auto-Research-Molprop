# Auto-Research-Molprop

Code and frozen results for the paper **"Closed-loop Auto Research for Molecular
Property Prediction: Discovering and Certifying Generalizable Improvements."**

Closed-loop Auto Research is an automation loop in which language-model agents go
beyond tuning a fixed model: they change the molecular representation, edit the
predictive model code, and acquire and vet external evidence, then submit each
change to an evaluator they do not control. This repository contains the harness
that runs that loop with **axis isolation** (a file-level ablation lock that
attributes every gain to one of three action classes: features, models, or data),
the **held-out certification** protocol that re-scores each validation-selected
configuration once on a test partition whose labels the search never reads, and
the **leakage-safe external-evidence filter** (standardised-identity de-duplication,
same-source rejection, and near-analogue removal). Evaluated across 36 endpoints in
three benchmark suites (TDC ADMET, MoleculeNet, and the Biogen adme-fang dataset on
Polaris).

## Repository layout

```
.
├── agent_core/            # adapter framework, harness, prompts, agent tooling
├── multi_agent_drug/      # core runner: evaluation, leakage-safe filter, ablation lock (TDC)
├── multi_agent_molnet/    # thin MoleculeNet adapter
├── multi_agent_polaris/   # thin Polaris (adme-fang) adapter
├── baselines/             # pretrained-representation baseline (ChemBERTa)
├── analysis/              # figure/table generators, held-out evaluation, consolidation
├── results/               # frozen results: held-out scores, result tables, figures
├── data/                  # how to obtain benchmark data and external evidence (no raw redistribution)
├── requirements.txt       # direct dependencies (loose bounds)
├── requirements-lock.txt  # full pip freeze of the tested Python 3.12 environment
├── LICENSE                # Apache-2.0 (code)
└── README.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate    # Python 3.12+
pip install -r requirements.txt                       # direct dependencies (loose bounds)
# to reproduce the exact tested environment instead:
# pip install -r requirements-lock.txt
```

RDKit, CatBoost, XGBoost, LightGBM, scikit-learn and PyTDC are the main scientific
dependencies and are all that the results verification needs. Three further
dependencies are optional and commented out in `requirements.txt`: the
`claude-agent-sdk` (only to run the agent loop), `torch` (only for the ChemBERTa
pretrained-baseline comparison), and `polaris-lib` (only to re-fetch the Polaris
data). `requirements-lock.txt` is the full `pip freeze` of the tested Python 3.12
environment.

## Minimal verification (no API key, no training, no GPU)

Everything needed to check the paper's numbers is in [`results/`](results/):

- `results/heldout/*.json` are the frozen per-endpoint validation and held-out-test
  scores that every table and figure is built from.
- `results/tables/*.md` reproduce the numbers behind every claim (held-out results,
  the leakage-controlled data case study, and the matched-trial AutoML comparison).
- `results/figures/` holds the figures used in the paper.

The result tables and the transfer-map, endpoint-transfer, and signature figures
each rebuild from the frozen scores with a single command (no API key, training, or
GPU); see [`results/README.md`](results/README.md) for the commands and for how each
artifact maps to the manuscript.

## Running the full agent loop (optional, requires an API key and significant compute)

The closed loop calls a language model for hypothesis generation and runs many
trials, each training the full model suite across all endpoints. It is not needed
to verify the results above.

```bash
cp .env.example .env                # then fill in ANTHROPIC_API_KEY (or export it directly)
python -m multi_agent_drug.supervisor \
  --baseline-score 0.0 \
  --specialists fphs,fsub,lit,meta \
  --max-trials 100
```

The active intervention axis is set with `HARNESS_ABLATION_MODE`
(`feature_only` / `model_only` / `data_only` / `joint`); the file-level ablation
lock restores all non-target files before each trial.

## Data

Benchmark datasets are public and are **not redistributed here**. The external
evidence used by the data axis is admitted only through the leakage-safe filter and
is documented by an auditable accept/reject record rather than by shipping raw
third-party files. See [`data/README.md`](data/README.md).

## License

- Code: Apache License 2.0 (see `LICENSE`).
- Derived analysis tables and results: Creative Commons Attribution 4.0 (CC BY 4.0).
- Third-party benchmark and source datasets retain their own licenses.

## Citation

```bibtex
@article{ning2026autoresearchmolprop,
  title   = {Closed-loop Auto Research for Molecular Property Prediction:
             Discovering and Certifying Generalizable Improvements},
  author  = {Ning, Jingjie and Li, Xiaochuan and Zeng, Ji and Xiong, Chenyan and Ke, Guolin},
  journal = {Journal of Cheminformatics},
  year    = {2026},
  note    = {Under review}
}
```
