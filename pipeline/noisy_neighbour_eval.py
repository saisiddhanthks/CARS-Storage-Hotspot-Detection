import warnings; warnings.filterwarnings("ignore")
import os
import pandas as pd, numpy as np, json
import hotspot_pipeline as hp
df = hp.simulate_noisy_neighbour()
kept, dropped = hp.drop_correlated(df, hp.FEATURES)
df, ae, thr = hp.run_detectors(df, kept)
train, _ = hp.time_split(df)
base_plain = hp.workload_baseline(train[train["consensus_anomaly"] == 0], robust=False)
att_plain = hp.attribute_noisy_neighbours(df, base_plain)
base = hp.workload_baseline(train[train["consensus_anomaly"] == 0])
att = hp.attribute_noisy_neighbours(df, base)
ev = hp.summarise_events(att)
truth = att["nn_role"]; pred = att["nn_pred"]
def pr(role):
    tp = ((truth == role) & (pred == role)).sum(); fp = ((truth != role) & (pred == role)).sum(); fn = ((truth == role) & (pred != role)).sum()
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), precision=round(p, 3), recall=round(r, 3), f1=round(2*p*r/(p+r), 3) if p+r else 0)
def pr_on(a, role):
    t = a["nn_role"]; p = a["nn_pred"]
    tp = ((t == role) & (p == role)).sum(); fp = ((t != role) & (p == role)).sum(); fn = ((t == role) & (p != role)).sum()
    pp = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    return dict(precision=round(pp, 3), recall=round(r, 3), f1=round(2*pp*r/(pp+r), 3) if pp+r else 0)
def ev_hits(a):
    h = 0
    for e, g in a[a["nn_event"] >= 0].groupby("nn_event"):
        ta = g.loc[g["nn_role"] == "aggressor", "node_id"].iloc[0]
        h += ta in g.loc[g["nn_pred"] == "aggressor", "node_id"].unique()
    return int(h)
# naive rule: at each consensus hotspot, blame the busiest other node in the cluster
naive = df.copy(); naive["nn_pred"] = "none"
naive["demand"] = naive["read_iops"] + naive["write_iops"]
for (ts, cl), g in naive.groupby(["timestamp", "cluster_id"]):
    hot = g[g["consensus_anomaly"] == 1]
    if len(hot):
        others = g.drop(hot.index)
        if len(others):
            naive.loc[others["demand"].idxmax(), "nn_pred"] = "aggressor"
            naive.loc[hot.index, "nn_pred"] = "victim"
ablation = {
  "naive": {"aggressor": pr_on(naive, "aggressor"), "victim": pr_on(naive, "victim"), "events": ev_hits(naive)},
  "plain": {"aggressor": pr_on(att_plain, "aggressor"), "victim": pr_on(att_plain, "victim"), "events": ev_hits(att_plain)},
  "robust": {"aggressor": pr_on(att, "aggressor"), "victim": pr_on(att, "victim"), "events": ev_hits(att)},
}
res = {"ablation": ablation, "rows": len(att), "events": int(att["nn_event"].max() + 1),
       "agg_rows": int((truth == "aggressor").sum()), "vic_rows": int((truth == "victim").sum()),
       "aggressor": pr("aggressor"), "victim": pr("victim")}
# event level: event detected if any timestamp in window has correct aggressor predicted
hit = 0; wrong = 0
for e, g in att[att["nn_event"] >= 0].groupby("nn_event"):
    true_agg = g.loc[g["nn_role"] == "aggressor", "node_id"].iloc[0]
    preds = g.loc[g["nn_pred"] == "aggressor", "node_id"].unique()
    if true_agg in preds: hit += 1
    if any(p != true_agg for p in preds): wrong += 1
res["events_correct"] = hit; res["events_wrong_culprit"] = wrong
# self-inflicted (true_hotspot outside nn) classification
loc = att[(att["true_hotspot"] == 1) & (att["nn_role"] == "none")]
res["local_true"] = len(loc)
res["local_pred_counts"] = loc["nn_pred"].value_counts().to_dict()
res["false_victim_on_local"] = int((loc["nn_pred"] == "victim").sum())
res["pred_counts"] = pred.value_counts().to_dict()
res["consensus_flagged"] = int(att["consensus_anomaly"].sum())
res["victim_consensus_recall"] = round(float(att.loc[truth == "victim", "consensus_anomaly"].mean()), 3)
res["summary_events"] = len(ev)
print(json.dumps(res, indent=1, default=str))
print(ev.head(12).to_string())
att.to_csv("nn_output.csv", index=False)
ev.to_csv("nn_events.csv", index=False)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "nn_results.json"), "w"), indent=1, default=str)
