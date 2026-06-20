"""Molecular feature engineering — agent's primary edit surface.

Baseline: MapLight featurization — the TDC ADMET *strong tabular baseline*
(github.com/maplightrx/MapLight-TDC, arXiv:2310.00174; top-1 on 6/22 and
top-3 on 16/22 leaderboards). UNIVERSAL representation = 2563 dims:
  - Morgan/ECFP COUNT fingerprint (radius 2, 1024)  GetHashedMorganFingerprint
  - Avalon COUNT fingerprint (1024)                 GetAvalonCountFP
  - ErG reduced-graph fingerprint (315)             rdReducedGraphs.GetErGFingerprint
  - 200 RDKit 2D physicochemical descriptors        MolecularDescriptorCalculator
Paired with a default CatBoost (see models.py), no per-task tuning.
This is the STRONG baseline the agents' task-specific features must beat.

Agents (fphs / fsub / lit) should extend or replace these functions to
add task-specific descriptors motivated by endpoint mechanism:
  - hERG toxicity  → charge, lipophilicity, aromatic ring count
  - BBB penetration → MW, PSA, H-bond donors/acceptors, lipophilicity
  - CYP inhibition  → planar aromatic systems, basic nitrogen
  - AMES mutagenicity → structural alerts (nitro, epoxide, etc.)
  - Solubility      → LogP, polar surface area, rotatable bonds

Feature provenance: add a comment above each new feature block with
format: # FEATURE: <name> | endpoint: <list> | source: <literature/rule>
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# MapLight's 200 RDKit 2D descriptors (github.com/maplightrx/MapLight-TDC,
# list via blopig.com 2022). Fixed → reproducible 200-dim physchem block.
_MAPLIGHT_DESCRIPTORS = [
    "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1",
    "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",
    "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3",
    "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8",
    "EState_VSA9", "ExactMolWt", "FpDensityMorgan1", "FpDensityMorgan2",
    "FpDensityMorgan3", "FractionCSP3", "HallKierAlpha", "HeavyAtomCount",
    "HeavyAtomMolWt", "Ipc", "Kappa1", "Kappa2", "Kappa3", "LabuteASA",
    "MaxAbsEStateIndex", "MaxAbsPartialCharge", "MaxEStateIndex", "MaxPartialCharge",
    "MinAbsEStateIndex", "MinAbsPartialCharge", "MinEStateIndex", "MinPartialCharge",
    "MolLogP", "MolMR", "MolWt", "NHOHCount", "NOCount", "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles",
    "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors",
    "NumHeteroatoms", "NumRadicalElectrons", "NumRotatableBonds",
    "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings",
    "NumValenceElectrons", "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12",
    "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5",
    "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "RingCount", "SMR_VSA1",
    "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7",
    "SMR_VSA8", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12",
    "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7",
    "SlogP_VSA8", "SlogP_VSA9", "TPSA", "VSA_EState1", "VSA_EState10", "VSA_EState2",
    "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7",
    "VSA_EState8", "VSA_EState9", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN",
    "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O",
    "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2",
    "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH",
    "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid",
    "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo",
    "fr_barbitur", "fr_benzene", "fr_benzodiazepine", "fr_bicyclic", "fr_diazo",
    "fr_dihydropyridine", "fr_epoxide", "fr_ester", "fr_ether", "fr_furan", "fr_guanido",
    "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imidazole", "fr_imide", "fr_isocyan",
    "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam", "fr_lactone",
    "fr_methoxy", "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom",
    "fr_nitro_arom_nonortho", "fr_nitroso", "fr_oxazole", "fr_oxime",
    "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond", "fr_phos_acid",
    "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide", "fr_prisulfonamd",
    "fr_pyridine", "fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone",
    "fr_term_acetylene", "fr_tetrazole", "fr_thiazole", "fr_thiocyan", "fr_thiophene",
    "fr_unbrch_alkane", "fr_urea", "qed",
]
_ERG_DIM = 315  # rdReducedGraphs.GetErGFingerprint fixed length


def _mols_from_smiles(smiles_series: pd.Series) -> list:
    """Parse SMILES → RDKit Mol (None for invalid). Silences RDKit log spam."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")  # mute parse errors + deprecation warnings
    return [Chem.MolFromSmiles(str(s)) for s in smiles_series]


def _count_fp_to_array(fp, n_bits: int) -> np.ndarray:
    """Sparse count fingerprint → dense float32 vector (overflow-safe).

    MapLight uses DataStructs.ConvertToNumpyArray into an int8 buffer; we read
    GetNonzeroElements() into float32 instead to avoid int8 overflow on large
    molecules and to skip an extra upcast. Values are identical otherwise.
    """
    arr = np.zeros(n_bits, dtype=np.float32)
    for idx, cnt in fp.GetNonzeroElements().items():
        if 0 <= idx < n_bits:
            arr[idx] = cnt
    return arr


def get_rdkit_descriptors(smiles_series: pd.Series) -> pd.DataFrame:
    """MapLight's 200 RDKit 2D physicochemical descriptors.

    One row per molecule. Invalid SMILES → NaN row (median-imputed in featurize).
    """
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

    calc = MolecularDescriptorCalculator(_MAPLIGHT_DESCRIPTORS)
    nan_row = [np.nan] * len(_MAPLIGHT_DESCRIPTORS)
    rows = []
    for mol in _mols_from_smiles(smiles_series):
        if mol is None:
            rows.append(nan_row)
        else:
            try:
                rows.append(list(calc.CalcDescriptors(mol)))
            except Exception:
                rows.append(nan_row)
    return pd.DataFrame(rows, columns=_MAPLIGHT_DESCRIPTORS, index=smiles_series.index)


def get_morgan_fingerprint(smiles_series: pd.Series,
                           radius: int = 2,
                           n_bits: int = 1024) -> pd.DataFrame:
    """MapLight ECFP: count-based Morgan fingerprint (GetHashedMorganFingerprint).

    radius=2 → ECFP4. Returns DataFrame fp_0 .. fp_{n_bits-1} of COUNTS (not bits).
    """
    from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint

    cols = [f"fp_{i}" for i in range(n_bits)]
    rows = []
    for mol in _mols_from_smiles(smiles_series):
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.float32))
        else:
            try:
                fp = GetHashedMorganFingerprint(mol, nBits=n_bits, radius=radius)
                rows.append(_count_fp_to_array(fp, n_bits))
            except Exception:
                rows.append(np.zeros(n_bits, dtype=np.float32))
    return pd.DataFrame(rows, columns=cols, index=smiles_series.index)


def get_avalon_fingerprint(smiles_series: pd.Series,
                           n_bits: int = 1024) -> pd.DataFrame:
    """MapLight Avalon count fingerprint (GetAvalonCountFP). Columns av_0 .. av_{n-1}."""
    from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP

    cols = [f"av_{i}" for i in range(n_bits)]
    rows = []
    for mol in _mols_from_smiles(smiles_series):
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.float32))
        else:
            try:
                fp = GetAvalonCountFP(mol, nBits=n_bits)
                rows.append(_count_fp_to_array(fp, n_bits))
            except Exception:
                rows.append(np.zeros(n_bits, dtype=np.float32))
    return pd.DataFrame(rows, columns=cols, index=smiles_series.index)


def get_erg_fingerprint(smiles_series: pd.Series) -> pd.DataFrame:
    """MapLight ErG reduced-graph fingerprint (rdReducedGraphs, 315-dim float)."""
    from rdkit.Chem import rdReducedGraphs

    cols = [f"erg_{i}" for i in range(_ERG_DIM)]
    rows = []
    for mol in _mols_from_smiles(smiles_series):
        if mol is None:
            rows.append(np.zeros(_ERG_DIM, dtype=np.float32))
        else:
            try:
                rows.append(np.asarray(rdReducedGraphs.GetErGFingerprint(mol),
                                       dtype=np.float32))
            except Exception:
                rows.append(np.zeros(_ERG_DIM, dtype=np.float32))
    return pd.DataFrame(rows, columns=cols, index=smiles_series.index)


def get_task_features(df: pd.DataFrame, task_name: str | None) -> np.ndarray:
    """Task-specific features — each block ONLY activates for its target task(s).

    THIS IS THE ANTI-SEESAW MECHANISM.
    Features added here affect ONLY the tasks listed in the `if task_name in (...)`
    guard. A feature block for 'herg' has zero effect on solubility, CYP, BBB,
    or any other task. You can safely improve one task without risking others.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PATTERN FOR ADDING TASK-SPECIFIC FEATURES                              │
    │                                                                         │
    │  # FEATURE: <name> | endpoint: <task(s)> | source: <citation/rule>     │
    │  if task_name in ('<exact_tdc_task_name>',):                            │
    │      feats = your_function(df)   # must return np.ndarray (n, k)       │
    │      parts.append(feats)                                                │
    │                                                                         │
    │  Rules:                                                                 │
    │  - Each append must be shape (n_molecules, k), k >= 1.                 │
    │  - Use EXACT TDC task names from INIT.md.                               │
    │  - Related tasks can share a block:                                     │
    │      ('cyp2d6_veith', 'cyp2d6_substrate_carbonmangels')                 │
    │  - Add FEATURE provenance comment above every block.                    │
    └─────────────────────────────────────────────────────────────────────────┘

    Contrast with the UNIVERSAL functions (get_rdkit_descriptors /
    get_morgan_fingerprint / get_avalon_fingerprint / get_erg_fingerprint):
    those affect all 22 tasks simultaneously and carry seesaw risk.
    Use THIS function for targeted, task-safe work.

    Returns np.zeros((n, 0)) when no blocks are active, so featurize()
    handles the empty case correctly.
    """
    parts: list[np.ndarray] = []
    n = len(df)

    # ── TASK-SPECIFIC FEATURE BLOCKS ──────────────────────────────────────────
    # Add new blocks here following the pattern above.
    # Example (uncomment and adapt):
    #
    # # FEATURE: herg_charge | endpoint: herg | source: Cavalli 2002 QSAR
    # if task_name in ('herg',):
    #     from rdkit import Chem
    #     from rdkit.Chem import Descriptors
    #     rows = []
    #     for smi in df["Drug"]:
    #         mol = Chem.MolFromSmiles(str(smi))
    #         rows.append([
    #             Descriptors.MaxPartialCharge(mol) if mol else 0.0,
    #             Descriptors.NumAromaticRings(mol) if mol else 0.0,
    #         ])
    #     parts.append(np.array(rows, dtype=np.float32))
    #
    # ── END TASK-SPECIFIC FEATURE BLOCKS ──────────────────────────────────────

    if not parts:
        return np.zeros((n, 0), dtype=np.float32)
    return np.hstack(parts).astype(np.float32)


def featurize(df: pd.DataFrame, smiles_col: str = "Drug",
              task_name: str | None = None) -> np.ndarray:
    """Combine universal MapLight features + task-specific features into one matrix.

    Universal block (all 22 tasks, 2563 dims): RDKit 200 descriptors +
    Morgan counts 1024 + Avalon counts 1024 + ErG 315.
    task_name routes to the correct task-specific block in get_task_features();
    when None, only universal features are used (safe fallback).

    Two editing surfaces:
      UNIVERSAL (all 22 tasks): get_rdkit_descriptors / get_morgan_fingerprint /
        get_avalon_fingerprint / get_erg_fingerprint
        → changes here carry seesaw risk; use only for broadly beneficial features.
      TASK-SPECIFIC (one task only): get_task_features()
        → safe to change without affecting other tasks; preferred for targeted work.
    """
    smiles = df[smiles_col]
    physchem = get_rdkit_descriptors(smiles)   # 200
    morgan   = get_morgan_fingerprint(smiles)  # 1024 counts
    avalon   = get_avalon_fingerprint(smiles)  # 1024 counts
    erg      = get_erg_fingerprint(smiles)     # 315

    combined = pd.concat([physchem, morgan, avalon, erg], axis=1)

    # inf/-inf → NaN, then median imputation (Ipc/partial-charge can blow up).
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in combined.columns:
        if combined[col].isna().any():
            med = combined[col].median()
            combined[col] = combined[col].fillna(med if not np.isnan(med) else 0.0)

    base = combined.values.astype(np.float32)

    # Task-specific features: additive, isolated to their target task(s).
    task_feats = get_task_features(df, task_name)
    if task_feats.shape[1] > 0:
        task_feats = np.nan_to_num(task_feats, nan=0.0, posinf=0.0, neginf=0.0)
        return np.hstack([base, task_feats])
    return base
