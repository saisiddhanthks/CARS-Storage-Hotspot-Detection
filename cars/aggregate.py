"""Aggregate all results into tables (JSON for the manuscript) and a CSV master file."""
import sys, os, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import load, events, evaluate, block_bootstrap_ci, OUT

NAMES = {"static_threshold": "Static p99 threshold", "rolling_3sigma": "Rolling 3σ (project)",
         "isolation_forest": "Isolation Forest (project)", "lof": "LOF (Study 1)", "dbscan": "DBSCAN (Study 1)",
         "dense_autoencoder": "Dense autoencoder (Study 1)", "lstm_autoencoder": "LSTM autoencoder (project)",
         "repo_ensemble": "Weighted ensemble (project)", "consensus_2of4": "2-of-4 consensus (Study 1)", "cars": "CARS (this work)"}
ABL = {"cars": "CARS (full)", "abl_no_persistence": "− persistence (α = 1)", "abl_no_coherence": "− peer coherence (β = 0)",
       "abl_no_queue_depth": "− queue depth in context", "abl_no_residual_model": "− expected-behaviour model (raw robust z)",
       "abl_latency_only": "− aggressor score (latency only)", "ctx_minimal": "Context without request shape",
       "ctx_pred_shape": "+ workload label (shape-only classifier)", "ctx_pred_full": "+ workload label (full classifier)",
       "ctx_oracle": "+ workload label (ground truth)"}
ORDER = list(NAMES)


def run(name, methods, ci=False):
    df, tr = load(name); test = df[~tr].reset_index(drop=True); evs = events(name)
    rows = []
    for m in methods:
        f = f"{OUT}/{name}/{m}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f); r = evaluate(test, z["score"], z["flag"], evs, m)
        if ci:
            (a0, a1), (f0, f1) = block_bootstrap_ci(test, z["score"], z["flag"])
            r.update(AP_lo=a0, AP_hi=a1, F1_lo=f0, F1_hi=f1)
        rows.append(r)
    return pd.DataFrame(rows).set_index("method")


if __name__ == "__main__":
    main = run("base", ORDER, ci=True)
    abl = run("base", list(ABL))
    main.to_csv(f"{OUT}/main_base.csv"); abl.to_csv(f"{OUT}/ablation_base.csv")
    rob = {}
    for v in ["base", "subtle", "partial", "subtle_partial"]:
        rob[v] = run(v, ORDER + ["abl_no_coherence"])
        rob[v].to_csv(f"{OUT}/rob_{v}.csv")
    print(main.round(3).to_string()); print(abl.round(3).to_string())
    for v in rob: print(v); print(rob[v][["AP", "F1", "surge_FPR", "event_recall_strict"]].round(3).to_string())
