"""Atlas-guided composition (clean replacement for the polluted joint run).

Because each TDC ADMET endpoint is an INDEPENDENT per-task model, 'route each
endpoint to its atlas-best clean intervention' is realizable with no cross-task
interaction. The composition aggregate = mean over endpoints of the best clean
single-axis norm_improvement achieved on that endpoint (or 0 = keep baseline if
no axis helps). This is an interpretable per-endpoint-oracle upper bound, free
of joint's search-budget dilution.
"""
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

mx={g:per_endpoint_max(d) for g,d in GROUPS.items()}
tasks=sorted(set().union(*[set(mx[g]) for g in mx]))

# single-axis aggregates (mean over endpoints of that axis's per-endpoint best, clipped at 0)
def axis_agg(g):
    return sum(max(0.0, mx[g].get(t,0.0)) for t in tasks)/len(tasks)

# atlas-guided composition: per endpoint pick best axis (or baseline=0)
comp_total=0.0; routing=defaultdict(int); rows=[]
for t in tasks:
    vals={g:mx[g].get(t,0.0) for g in GROUPS}
    bestg=max(vals,key=vals.get); bestv=vals[bestg]
    if bestv<=0.0: bestg="baseline"; bestv=0.0
    routing[bestg]+=1; comp_total+=bestv
    rows.append((t,bestg,bestv,vals))
comp=comp_total/len(tasks)

print(f"endpoints: {len(tasks)}")
print(f"\nsingle-axis 'oracle-per-endpoint' aggregates (mean of per-endpoint best, clipped>=0):")
for g in GROUPS: print(f"  {g:8s} {axis_agg(g):.4f}")
print(f"\nreference best-trial aggregates (from runs): feature 0.0125  model 0.0412  data 0.0316  joint 0.0298")
print(f"\n>>> ATLAS-GUIDED COMPOSITION aggregate = {comp:.4f}")
print(f"    routing: " + ", ".join(f"{g}:{c}" for g,c in sorted(routing.items(),key=lambda x:-x[1])))
print(f"\nper-endpoint routing (axis chosen, gain):")
for t,bg,bv,vals in sorted(rows,key=lambda r:-r[2]):
    print(f"  {t:32s} -> {bg:9s} {bv:+.3f}   (feat{vals['feature']:+.3f} model{vals['model']:+.3f} data{vals['data']:+.3f})")
