"""Test the transferable principle: which axis wins vs dataset size (n_train).
Hypothesis: small endpoints -> data augmentation; larger -> model tuning."""
import json, glob
from collections import defaultdict

GROUPS = {"feature":"drug_dev_maplight_feature","model":"drug_dev_maplight_model_only","data":"drug_dev_maplight_data_v2"}

def per_endpoint_max(d):
    best = defaultdict(lambda: -9.9)
    for f in glob.glob(f"{d}/blackboard/snapshots/*/eval/run_seed0.jsonl"):
        try: rec=json.loads(open(f).readline())
        except: continue
        for t,info in (rec.get("per_task",{}) or {}).items():
            ni=info.get("norm_improvement")
            if ni is not None and ni>best[t]: best[t]=ni
    return best

def base_ntrain(d):
    # n_train from a non-augmented group (feature) = base internal train size
    n={}
    for f in glob.glob(f"{d}/blackboard/snapshots/*/eval/run_seed0.jsonl"):
        try: rec=json.loads(open(f).readline())
        except: continue
        for t,info in (rec.get("per_task",{}) or {}).items():
            if t not in n and info.get("n_train"): n[t]=info["n_train"]
    return n

mx={g:per_endpoint_max(d) for g,d in GROUPS.items()}
nt=base_ntrain(GROUPS["feature"])
tasks=sorted(nt, key=lambda t: nt[t])  # sort by training size

print(f"{'endpoint':32s}{'n_train':>8}{'feat':>8}{'model':>8}{'data':>8}  winner")
print("-"*76)
for t in tasks:
    f=mx['feature'].get(t,0); m=mx['model'].get(t,0); d=mx['data'].get(t,0)
    vals={'feat':f,'model':m,'data':d}; w=max(vals,key=vals.get)
    w = w if vals[w]>0.005 else '~flat'
    print(f"{t:32s}{nt[t]:>8}{f:+8.3f}{m:+8.3f}{d:+8.3f}  {w}")
print("-"*76)
# split at median n_train
med=sorted(nt.values())[len(nt)//2]
small=[t for t in tasks if nt[t]<med]; large=[t for t in tasks if nt[t]>=med]
def tally(ts):
    c=defaultdict(int)
    for t in ts:
        f=mx['feature'].get(t,0); m=mx['model'].get(t,0); d=mx['data'].get(t,0)
        vals={'feat':f,'model':m,'data':d}; w=max(vals,key=vals.get)
        if vals[w]>0.005: c[w]+=1
    return dict(c)
print(f"median n_train = {med}")
print(f"SMALL endpoints (n_train<{med}): winner tally = {tally(small)}")
print(f"LARGE endpoints (n_train>={med}): winner tally = {tally(large)}")
