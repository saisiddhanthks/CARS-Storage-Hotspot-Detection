"""Aggressor attribution and victim identification.

Compares CARS against (a) the repo's NoisyNeighborDetector logic (aggressor candidate =
ensemble score >= 60; victim = co-located volume with global latency z > 2 and IOPS z < 1)
and (b) a naive 'largest IOPS z-score on the node' rule."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import load, events, OUT

name = sys.argv[1]
df, tr = load(name); te = ~tr
test = df[te].reset_index(drop=True)
evs = [e for e in events(name) if e["start"] >= test.tidx.min() and e["start"] + e["dur"] - 1 <= test.tidx.max()]
parts = np.load(f"{OUT}/{name}/parts_cars.npz")
cars = np.load(f"{OUT}/{name}/cars.npz")
test["A"] = parts["A"]; test["cars_flag"] = cars["flag"]
ens = np.load(f"{OUT}/{name}/ensemble_score_test.npy"); test["ens"] = ens

# global per-volume baselines from training period (repo NoisyNeighborDetector.fit_baselines)
b = df[tr].groupby("volume_id").agg(lm=("avg_latency_us", "mean"), ls=("avg_latency_us", "std"),
                                     im=("total_iops", "mean"), isd=("total_iops", "std"))
test = test.join(b, on="volume_id")
test["lat_z"] = (test.avg_latency_us - test.lm) / test.ls.clip(lower=1)
test["iops_z"] = (test.total_iops - test.im) / test.isd.clip(lower=1)
key = test.set_index(["tidx", "volume_id"])

res = dict(cars_correct=0, cars_detected=0, naive_correct=0, repo_correct=0, repo_detected=0, n=len(evs))
vic_tp = dict(cars=0, repo=0); vic_fp = dict(cars=0, repo=0); vic_total = 0
for e in evs:
    vols = [e["aggressor"]] + list(e["victims"])
    node_vols = test.loc[(test.tidx == e["start"]) & (test.node_id == e["node"]), "volume_id"].tolist()
    # ---- CARS: first interval with any flagged event row; aggressor = argmax A on node
    for k in range(e["dur"]):
        t = e["start"] + k
        sub = key.loc[t].loc[node_vols]
        if sub.loc[[v for v in vols if v in sub.index], "cars_flag"].any():
            res["cars_detected"] += 1
            res["cars_correct"] += int(sub["A"].idxmax() == e["aggressor"])
            break
    # ---- naive: argmax IOPS z at event start
    sub0 = key.loc[e["start"]].loc[node_vols]
    res["naive_correct"] += int(sub0["iops_z"].idxmax() == e["aggressor"])
    # ---- repo NN detector: candidates with ensemble >= 60 during the event on that node
    found = False
    for k in range(e["dur"]):
        t = e["start"] + k
        sub = key.loc[t].loc[node_vols]
        cands = sub[sub.ens >= 60].index.tolist()
        best, best_imp = None, 0
        for c in cands:
            nb = sub.drop(index=c)
            vic = nb[(nb.lat_z > 2.0) & (nb.iops_z < 1.0)]
            imp = (vic.lat_z / vic.iops_z.clip(lower=0.1)).sum()
            if len(vic) and imp > best_imp:
                best, best_imp = c, imp
        if best is not None:
            res["repo_detected"] += 1; res["repo_correct"] += int(best == e["aggressor"]); found = True
            break
    # ---- victim identification at every event interval
    for k in range(e["dur"]):
        sub = key.loc[e["start"] + k].loc[node_vols]
        true_v = set(e["victims"])
        vic_total += len(true_v)
        cars_v = set(sub.index[(sub.cars_flag) & (sub.index != e["aggressor"])])
        repo_v = set(sub.index[(sub.lat_z > 2.0) & (sub.iops_z < 1.0)])
        for nm, s in [("cars", cars_v), ("repo", repo_v)]:
            vic_tp[nm] += len(s & true_v); vic_fp[nm] += len(s - true_v)
out = dict(n_events=res["n"],
           cars_detect_rate=res["cars_detected"] / res["n"],
           cars_attr_acc_given_detected=res["cars_correct"] / max(res["cars_detected"], 1),
           cars_attr_acc_all=res["cars_correct"] / res["n"],
           naive_attr_acc=res["naive_correct"] / res["n"],
           repo_detect_rate=res["repo_detected"] / res["n"],
           repo_attr_acc_given_detected=res["repo_correct"] / max(res["repo_detected"], 1),
           repo_attr_acc_all=res["repo_correct"] / res["n"],
           victim_recall_cars=vic_tp["cars"] / vic_total, victim_precision_cars=vic_tp["cars"] / max(vic_tp["cars"] + vic_fp["cars"], 1),
           victim_recall_repo=vic_tp["repo"] / vic_total, victim_precision_repo=vic_tp["repo"] / max(vic_tp["repo"] + vic_fp["repo"], 1))
print(json.dumps(out, indent=1))
json.dump(out, open(f"{OUT}/{name}/attribution.json", "w"), indent=1)
