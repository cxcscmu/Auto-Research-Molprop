"""Per-endpoint x axis 'promising directions' map.

For each axis group (feature / model / data) and each endpoint, take the MAX
norm_improvement achieved across that group's kept snapshots — i.e. the best
that axis was able to do on that endpoint. Local-only (reads pulled dirs).
"""
import json, glob
from collections import defaultdict

GROUPS = {
    "feature": "drug_dev_maplight_feature",
    "model":   "drug_dev_maplight_model_only",
    "data":    "drug_dev_maplight_data_v2",
}
METRIC = {  # for context (direction already baked into norm_improvement)
 "caco2_wang":"MAE","lipophilicity_astrazeneca":"MAE","solubility_aqsoldb":"MAE","ppbr_az":"MAE","ld50_zhu":"MAE",
 "vdss_lombardo":"Spm","half_life_obach":"Spm","clearance_hepatocyte_az":"Spm","clearance_microsome_az":"Spm",
 "hia_hou":"AUROC","pgp_broccatelli":"AUROC","bioavailability_ma":"AUROC","bbb_martins":"AUROC",
 "cyp3a4_substrate_carbonmangels":"AUROC","herg":"AUROC","ames":"AUROC","dili":"AUROC",
 "cyp2d6_veith":"PRAUC","cyp3a4_veith":"PRAUC","cyp2c9_veith":"PRAUC",
 "cyp2d6_substrate_carbonmangels":"PRAUC","cyp2c9_substrate_carbonmangels":"PRAUC",
}

def per_endpoint_max(d):
    best = defaultdict(lambda: -9.9)
    n = 0
    for f in glob.glob(f"{d}/blackboard/snapshots/*/eval/run_seed0.jsonl"):
        try:
            rec = json.loads(open(f).readline())
        except Exception:
            continue
        n += 1
        for t, info in (rec.get("per_task", {}) or {}).items():
            ni = info.get("norm_improvement")
            if ni is not None and ni > best[t]:
                best[t] = ni
    return n, best

maxes, ns = {}, {}
for g, d in GROUPS.items():
    ns[g], maxes[g] = per_endpoint_max(d)

tasks = sorted(METRIC, key=lambda t: METRIC[t])
print(f"snapshots scanned: " + "  ".join(f"{g}={ns[g]}" for g in GROUPS))
print(f"\n{'endpoint':32s}{'metric':>7}{'feat':>9}{'model':>9}{'data':>9}   best-axis")
print("-"*78)
axis_win = defaultdict(int)
for t in tasks:
    f = maxes["feature"].get(t, 0.0); m = maxes["model"].get(t, 0.0); d = maxes["data"].get(t, 0.0)
    vals = {"feat": f, "model": m, "data": d}
    win = max(vals, key=vals.get)
    tag = win if vals[win] > 0.005 else "~flat"
    if vals[win] > 0.005:
        axis_win[win] += 1
    print(f"{t:32s}{METRIC[t]:>7}{f:+9.3f}{m:+9.3f}{d:+9.3f}   {tag}")
print("-"*78)
print("best-axis tally (endpoints where that axis gave the largest >0.005 gain):")
for a, c in sorted(axis_win.items(), key=lambda x:-x[1]):
    print(f"  {a}: {c} endpoints")
