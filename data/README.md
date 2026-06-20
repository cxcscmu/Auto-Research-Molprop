# Data

This repository does **not** redistribute raw datasets. Benchmark data is public
and downloadable from its original providers, and external evidence is handled
through an auditable, leakage-safe protocol rather than by shipping files.

## Benchmark suites

| Suite | Endpoints | Source |
|-------|-----------|--------|
| TDC ADMET Benchmark Group | 22 | `PyTDC` (`pip install PyTDC`); `from tdc.benchmark_group import admet_group` |
| MoleculeNet | 10 (FreeSolv, ESOL, BACE, HIV, Tox21, SIDER, ClinTox) | MoleculeNet / DeepChem |
| Polaris adme-fang (Biogen) | 4 (HLM, RLM, MDR1-MDCK, solubility) | `polaris` hub, `biogen/adme-fang-*-reg-v1` |

The harness loads these through the per-suite adapters in
`multi_agent_{drug,molnet,polaris}/` and `agent_core/task_adapter.py`. The internal
train/validation split (scaffold-based 80:20, seed 42) and the held-out test split
are described in the paper.

## External evidence (the data axis)

The data axis acquires external labelled molecules through a single controlled tool.
**Every external file passes a three-layer leakage-safe filter before any row is
merged**, enforced by the harness and not bypassable by the agent:

1. Standardised-identity de-duplication (desalt, neutralise, InChIKey) against test,
   validation, and training molecules.
2. Whole-file same-source rejection when more than 5% of an endpoint's test
   structures appear in the file (by InChIKey skeleton).
3. Near-analogue removal at ECFP4 Tanimoto >= 0.9 to any test molecule.

We do not redistribute the raw external CSVs for two reasons: some are derived from
sources with their own (including share-alike) licenses, and the files rejected by
the same-source layer by definition overlap benchmark test molecules. Instead, the
complete accept/reject audit (source, overlap rate, per-layer counts) is in
[`../results/tables/case_study.md`](../results/tables/case_study.md), and the filter
itself is in `multi_agent_drug/` (the external-data merge in the trial runner).

The independent sources that were accepted are public and can be obtained from their
original providers, for example FDA drug-interaction and clinical-pharmacology
records, the Obach 1999 clearance set, and DILIrank.

## Licenses

Benchmark and third-party source datasets retain their own licenses. Any derived
analysis tables released in this repository are under Creative Commons Attribution
4.0 (CC BY 4.0).
