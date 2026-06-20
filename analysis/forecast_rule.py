"""Foresight rule: predict each endpoint's best axis from PRIOR attributes only
(known before running any search), then measure how often the prior-only rule
recovers the observed best axis across the 36 endpoints.

Prior attributes (all knowable a priori, no search needed):
  - n_train               : unaugmented internal training size (data scarcity)
  - physchem_regression   : endpoint is a molecular physicochemical continuous
                            property (solubility / hydration / lipophilicity)
  - mature_source         : a compatible independent public assay source plausibly
                            exists (mature ADME/PK/tox literature: ChEMBL/FDA/etc.)

Rule (domain prior from the TDC P1/P2/P3 principles, NOT fit to the labels):
    if physchem_regression and n_train < SMALL : feature   # physics features unsaturated on small physchem
    elif mature_source     and n_train < SMALL : data      # small + independent source may fill gaps
    else                                       : model     # large or no special condition -> model default

We report overall accuracy, per-axis precision, and the critical
"data-candidate zone" (small, non-physchem, mature source) where model and data
compete -- the zone a prior rule cannot resolve.
"""
from collections import defaultdict
from foresight import build_rows

SMALL = 800     # data window upper bound (TDC P1: data wins only at n_train <~ 780)
LARGE = 1500    # model-saturation onset

# --- prior attribute annotations (a priori, auditable) ---
PHYSCHEM = {
    "freesolv", "esol", "solubility_aqsoldb", "lipophilicity_astrazeneca", "adme_solu",
}
# Endpoints with a plausible compatible independent public assay source.
# Mature ADME/PK/tox endpoints (ChEMBL/FDA/literature) -> True.
# Physicochemical experimental values and bespoke screening panels
# (FreeSolv/ESOL hydration-solubility, Tox21/SIDER/ClinTox panels, HIV screen) -> False.
NO_SOURCE = {
    "freesolv", "esol",                                  # bespoke physchem experiments
    "tox21_sr_mmp", "tox21_sr_p53", "tox21_nr_ar",       # Tox21 panel
    "sider_hepatobiliary", "sider_reproductive",         # SIDER side-effect panel
    "clintox_ct_tox", "hiv",                             # bespoke screens
}


def has_source(ep):
    return ep not in NO_SOURCE


def forecast(n_train, physchem, source):
    if physchem and n_train < SMALL:
        return "feature"
    if source and n_train < SMALL:
        return "data"
    return "model"


def main():
    rows = build_rows()
    hits = 0
    n_nonflat = 0
    conf = defaultdict(int)            # (predicted, actual) -> count
    zone_dc = defaultdict(int)         # data-candidate zone actual-axis distribution
    print(f"{'bench':8s}{'endpoint':30s}{'n_tr':>6} {'pc':>3}{'src':>4}  {'pred':>7} {'actual':>7}  hit")
    print("-" * 78)
    for r in rows:
        ep, n = r["endpoint"], r["n_train"]
        pc = ep in PHYSCHEM
        src = has_source(ep)
        pred = forecast(n, pc, src)
        actual = r["best_axis"]
        hit = (pred == actual)
        if actual != "flat":
            n_nonflat += 1
            if hit:
                hits += 1
        conf[(pred, actual)] += 1
        if n < SMALL and not pc and src:
            zone_dc[actual] += 1
        print(f"{r['benchmark']:8s}{ep:30s}{n:>6} {('Y' if pc else '.'):>3}{('Y' if src else '.'):>4}"
              f"  {pred:>7} {actual:>7}  {'OK' if hit else 'x'}")
    print("-" * 78)
    total = len(rows)
    overall_hit = sum(v for (p, a), v in conf.items() if p == a)
    print(f"overall: {overall_hit}/{total} = {overall_hit/total:.1%}   "
          f"(non-flat only: {hits}/{n_nonflat} = {hits/n_nonflat:.1%})")

    # per-axis precision
    print("\nper-predicted-axis precision (how often a prediction is correct):")
    for axis in ("feature", "model", "data"):
        pred_n = sum(v for (p, a), v in conf.items() if p == axis)
        pred_ok = conf[(axis, axis)]
        if pred_n:
            print(f"  predict {axis:7s}: {pred_ok}/{pred_n} = {pred_ok/pred_n:.0%}")

    # the unforecastable data-candidate zone
    print(f"\ndata-candidate zone (n_train<{SMALL}, non-physchem, mature source):")
    print(f"  actual best-axis distribution = {dict(zone_dc)}")
    print("  -> model vs data is ~even here; no prior attribute separates them,")
    print("     so the data window can only be settled by running the search.")

    # strong-prior zones
    big = [r for r in rows if r["n_train"] >= LARGE]
    big_model = sum(1 for r in big if r["best_axis"] == "model")
    print(f"\nlarge zone (n_train>={LARGE}): model wins {big_model}/{len(big)} "
          f"= {big_model/len(big):.0%}")
    small_pc = [r for r in rows if r["n_train"] < SMALL and r["endpoint"] in PHYSCHEM]
    sp_feat = sum(1 for r in small_pc if r["best_axis"] == "feature")
    print(f"small physchem-regression zone (n_train<{SMALL}): feature wins "
          f"{sp_feat}/{len(small_pc)}")

    # held-out external validation: the rule is fixed on TDC and applied, without
    # modification, to the 14 MoleculeNet + Polaris endpoints that did not inform it
    tr = [r for r in rows if r["benchmark"] in ("MolNet", "Polaris")]
    tr_hit = sum(1 for r in tr
                 if forecast(r["n_train"], r["endpoint"] in PHYSCHEM,
                             has_source(r["endpoint"])) == r["best_axis"])
    print(f"held-out transfer (rule fixed on TDC -> MolNet+Polaris): {tr_hit}/{len(tr)}")


if __name__ == "__main__":
    main()
