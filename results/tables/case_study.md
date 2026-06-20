# Case study: auditable external-evidence actions

The external-evidence action is the one that requires the agent to bring in **new labelled molecular evidence**, so the provenance of that evidence — and the risk that an “independent” source is secretly the benchmark's own assay — is a first-order correctness concern. The harness logs a structured audit record for every proposed source and runs each through a three-stage leakage-safe filter before any row reaches training:

- **L1 identity dedup** — drop external rows whose InChIKey matches a test, validation, or train molecule.
- **L2 same-source rejection** — if more than 5% of the held-out **test** molecules (InChIKey-skeleton match) appear in a source, reject the whole source as the benchmark's own / a sibling assay.
- **L3 analog filter** — drop external rows within ECFP4 Tanimoto ≥ 0.90 of any test molecule.

All fields below are verbatim from the run log; nothing is hand-edited.


## TDC ADMET

**Independent sources accepted into the deployed model** (after row-level scrub):

| Endpoint | External source (agent's description) | L1/L3 rows dropped | Added |
|---|---|---|---|
| bbb_martins | Post-2012 FDA-approved drugs with well-established BBB status from FDA pr… | test 0/val 0/train 0 | **+12** |
| bioavailability_ma | Post-2009 FDA-approved oral kinase inhibitors/oncology drugs. Absolute or… | test 0/val 0/train 0 | **+13** |
| caco2_wang | Caco-2 A-to-B permeability reference values from Hubatsch et al 2007 Natu… | test 0/val 1/train 3 | **+4** |
| clearance_microsome_az | Human liver microsomal intrinsic clearance from Obach 1999 JPET 253:103-1… | test 1/val 0/train 1 | **+4** |
| cyp2c9_substrate_carbonmangels | Curated CYP2C9 substrate/non-substrate pairs from FDA Drug Interaction Gu… | test 3/val 3/train 5, analog 1 | **+23** |
| cyp2d6_substrate_carbonmangels | Curated CYP2D6 substrate/non-substrate pairs from FDA Drug Interaction Gu… | test 3/val 2/train 6, analog 3 | **+14** |
| cyp3a4_substrate_carbonmangels | CYP3A4 substrate/non-substrate pairs from FDA DDI Guidance (2020), Rendic… | test 3/val 2/train 6 | **+26** |
| dili | Curated DILI labels from DILIrank (Chen 2016), Xu 2015 benchmark, FDA hep… | test 2/val 2/train 8 | **+20** |
| half_life_obach | FDA clinical pharmacology prescribing information for post-2012 oncology … | test 0/val 0/train 0 | **+14** |
| pgp_broccatelli | P-glycoprotein substrate/non-substrate labels for post-2011 FDA-approved … | test 0/val 0/train 0 | **+10** |
| vdss_lombardo | Post-2019 FDA-approved small-molecule drugs. Apparent volume of distribut… | test 0/val 0/train 0 | **+8** |

## MoleculeNet

**Independent sources accepted into the deployed model** (after row-level scrub):

| Endpoint | External source (agent's description) | L1/L3 rows dropped | Added |
|---|---|---|---|
| bace | ChEMBL BACE1 (CHEMBL4822) IC50 activity data, document_year >= 2017, inde… | test 1/val 0/train 1 | **+59** |
| sider_hepatobiliary | DILIrank (Chen 2016 Drug Discov Today), Xu 2015 benchmark, FDA hepatotoxi… | test 2/val 7/train 8 | **+15** |
| sider_reproductive | Post-2015 FDA-approved drugs with documented reproductive system adverse … | test 0/val 0/train 0 | **+19** |

## Polaris adme-fang

**Independent sources accepted into the deployed model** (after row-level scrub):

| Endpoint | External source (agent's description) | L1/L3 rows dropped | Added |
|---|---|---|---|
| adme_hlm | ChEMBL HLM liver microsomal CLint (Homo sapiens) from multiple labs, IVIV… | test 0/val 0/train 0 | **+441** |
| adme_mdr1 | ChEMBL MDR1 P-gp efflux ratio (CHEMBL5034188: 12 compounds, CHEMBL5104792… | test 0/val 0/train 0 | **+12** |
| adme_rlm | ChEMBL RLM liver microsomal CLint (Rattus norvegicus) from multiple labs,… | test 0/val 0/train 0 | **+418** |
| adme_solu | ESOL/Delaney 2004 aqueous solubility dataset (doi:10.1021/ci034243x). Y c… | test 0/val 0/train 1 | **+554** |

## Sources REJECTED as same-source (L2) — across all benchmarks

| Benchmark | Endpoint | Rejected source | Test overlap |
|---|---|---|---|
| TDC ADMET | ames | mathworks/Chemistry-Deep-Learning-GCN-Mutagenicity-Classification AMES_Al… | **64%** |
| TDC ADMET | half_life_obach | PKSmart Seal et al. 2025 J.Cheminformatics human pharmacokinetics databas… | **88%** |
| TDC ADMET | vdss_lombardo | PKSmart Seal et al. 2025 J.Cheminformatics human pharmacokinetics databas… | **89%** |

## Summary

- **18** independent, literature-cited sources were accepted into the deployed models, adding **+1666** training rows — each still scrubbed row-by-row (L1/L3 removed 32 exact test/val matches even from accepted sources).
- **3** sources were rejected outright by L2 as same-source, including benchmark-origin or sibling sources: **PKSmart** for `half_life`/`vdss` (88–89% test overlap) and a public mutagenicity set for `ames` (64%). Admitting any of these would have produced a large but **illusory** validation gain.
- The agent also **self-corrected** by clearing proposed external sources after the evaluator scored them as harmful. Do not quote this as a detailed case unless the exact trial IDs and source names are pulled from the lineage log.

## Why this matters

The rejections are the point. An unconstrained agent could merge benchmark sibling assays back into training and report spurious gains; L2 blocks exactly this. The accepted sources are independent assays (FDA interaction guidance, Obach 1999, DILIrank, ChEMBL) curated with citations and still scrubbed row-by-row. This reduces leakage risk and makes the evidence action auditable. It does not make the validation signal automatically unbiased or guarantee distributional match, which is why the Polaris data action can pass the same filter and still fail on held-out evaluation. Every decision is reconstructable from the run log, the auditability property a drug-discovery setting requires.
