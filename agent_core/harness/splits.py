"""Scaffold-split helpers shared by the trial runner and benchmark loaders.

`scaffold_partition_indices` groups molecules by Bemis–Murcko scaffold, shuffles the
groups deterministically, and greedily fills the 'small' partition up to `frac`,
keeping whole scaffold groups on one side (no scaffold straddles the boundary).
Falls back to a random index split if RDKit scaffold extraction fails.

Two callers, one implementation:
  - run_trial_drug.py `_scaffold_split` — inner train/val split (small = val)
  - MolNet loader — outer full→train_val/test split (small = test)

The grouping/shuffle/greedy-fill logic is a verbatim extraction of the former
`_scaffold_split` body, so TDC's inner split is byte-identical given the same
(smiles, frac, seed).
"""

from __future__ import annotations

from typing import Sequence


def scaffold_partition_indices(smiles: Sequence, frac: float, seed: int):
    """Partition row positions [0..len(smiles)) into (large_idx, small_idx).

    `small_idx` receives ~`frac` of the rows as whole scaffold groups; `large_idx`
    gets the rest. Deterministic given `seed`.
    """
    import numpy as np
    n = len(smiles)
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        rng = np.random.default_rng(seed)

        scaffolds: dict[str, list[int]] = {}
        for i, smi in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(str(smi))
                sca = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
            except Exception:
                sca = ""
            scaffolds.setdefault(sca, []).append(i)

        scaffold_groups = list(scaffolds.values())
        rng.shuffle(scaffold_groups)

        n_small = int(n * frac)
        small_idx: list[int] = []
        large_idx: list[int] = []
        for grp in scaffold_groups:
            if len(small_idx) < n_small:
                small_idx.extend(grp)
            else:
                large_idx.extend(grp)
        return large_idx, small_idx
    except Exception:
        # Fallback to random split if RDKit scaffold fails.
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_small = int(n * frac)
        return list(idx[n_small:]), list(idx[:n_small])


__all__ = ["scaffold_partition_indices"]
