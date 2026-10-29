"""CARS: Contention-Aware Residual Scoring for storage hotspots.

1. Workload context: a LightGBM workload classifier (trained on the training period).
2. Expected behaviour: robust (L1) LightGBM regressors predict log latency and log IOPS
   from workload, volume identity, tier, request shape (size, sequentiality, read ratio),
   queue depth and time-of-day.  IOPS is conditioned on queue depth (Little's law, L = lambda*W).
3. Residuals are standardised per volume with median/MAD from the training period.
4. Causal EWMA persistence per volume.
5. Topology coherence: mean of the top-k latency residuals among co-located peers.
6. Victim score V = e_lat + beta*min(C, e_lat+);  aggressor score A = e_iops + beta*min(C, e_iops+);
   hotspot score S = max(V, A); alarm threshold = 97th percentile of S on the training period.
"""
import sys, os, json, time, itertools
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
from common import load, OUT, WORKLOADS, SPLIT
from sklearn.metrics import average_precision_score

LGB = dict(n_estimators=300, learning_rate=0.08, num_leaves=63, min_child_samples=50,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=2, random_state=42)
DROP = {"role", "event_id", "surge", "y", "tidx", "day", "timestamp", "workload_type", "label",
        "volume_id", "node_id", "pool_id", "tier"}


SHAPE = ["io_size_avg_kb", "io_size_entropy", "sequential_ratio", "read_write_ratio"]


def workload_probs(df, fit_mask, shape_only=False):
    feats = SHAPE if shape_only else [c for c in df.columns if c not in DROP and df[c].dtype.kind in "fi"]
    clf = lgb.LGBMClassifier(**LGB).fit(df.loc[fit_mask, feats], df.loc[fit_mask, "label"])
    return clf.predict_proba(df[feats])


def context_matrix(df, fit_mask, mode, P=None, drop_qd=False):
    """mode: shape | pred_full | pred_shape | oracle | minimal"""
    X = pd.DataFrame(index=df.index)
    if mode in ("pred_full", "pred_shape", "oracle"):
        W = np.eye(5)[df["label"].to_numpy()] if mode == "oracle" else np.eye(5)[P.argmax(1)]  # hard labels: trees split binary inputs near 0
        W[fit_mask] = np.eye(5)[df.loc[fit_mask, "label"].to_numpy()]   # true labels while fitting
        for i, w in enumerate(WORKLOADS):
            X[f"wl_{w}"] = W[:, i]
    X["vol"] = df["volume_id"].astype("category").cat.codes
    X["tier"] = df["tier"].astype("category").cat.codes
    cols = ["hour_sin", "hour_cos", "day_sin", "day_cos"] + ([] if drop_qd else ["queue_depth"])
    if mode != "minimal":
        cols += ["io_size_avg_kb", "sequential_ratio", "read_write_ratio"]
    for c in cols:
        X[c] = df[c].to_numpy()
    return X


def residual_z(df, X, fit_mask, target, drop=()):
    Xc = X.drop(columns=list(drop))
    y = np.log(df[target].to_numpy(float) + 1.0)
    reg = lgb.LGBMRegressor(objective="l1", **LGB).fit(Xc[fit_mask], y[fit_mask], categorical_feature=["vol", "tier"])
    r = y - reg.predict(Xc)
    rr = pd.Series(r[fit_mask]).groupby(df.loc[fit_mask, "volume_id"].to_numpy())
    med = rr.median(); mad = rr.apply(lambda s: np.median(np.abs(s - np.median(s)))) * 1.4826
    v = df["volume_id"].to_numpy()
    return (r - pd.Series(v).map(med).to_numpy()) / np.maximum(pd.Series(v).map(mad).to_numpy(), 1e-3)


def ewma(df, x, alpha):
    if alpha >= 1.0:
        return x
    s = pd.Series(x, index=df.index)
    return s.groupby(df["volume_id"].to_numpy()).transform(lambda q: q.ewm(alpha=alpha, adjust=False).mean()).to_numpy()


def coherence(df, e, k):
    """Mean of top-k positive residuals among co-located peers (self excluded)."""
    d = pd.DataFrame({"t": df["tidx"].to_numpy(), "n": df["node_id"].to_numpy(), "e": np.clip(e, 0, None)})
    d["rank"] = d.groupby(["t", "n"])["e"].rank(ascending=False, method="first")
    out = np.zeros(len(d))
    g = d.groupby(["t", "n"])
    # sums of top-(k+1) values per group, then remove self where self is inside it
    topk = d[d["rank"] <= k].groupby(["t", "n"])["e"].sum()
    kth1 = d[d["rank"] == k + 1].set_index(["t", "n"])["e"]
    key = pd.MultiIndex.from_arrays([d["t"], d["n"]])
    sk = topk.reindex(key).to_numpy(); s1 = kth1.reindex(key).fillna(0).to_numpy()
    inside = d["rank"].to_numpy() <= k
    out = np.where(inside, sk - d["e"].to_numpy() + s1, sk) / k
    return out


def cars_scores(z_lat, z_iops, df, alpha, k, beta, use_coh=True):
    el = ewma(df, z_lat, alpha); ei = ewma(df, z_iops, alpha)
    if not use_coh or beta == 0:
        return np.maximum(el, ei), el, ei, np.zeros(len(el))
    C = coherence(df, el, k)
    V = el + beta * np.minimum(C, np.clip(el, 0, None))
    A = ei + beta * np.minimum(C, np.clip(ei, 0, None))
    return np.maximum(V, A), V, A, C


def components(df, fit_mask, mode="shape", P=None, drop_qd=False):
    if mode in ("pred_full", "pred_shape") and P is None:
        P = workload_probs(df, fit_mask, shape_only=(mode == "pred_shape"))
    X = context_matrix(df, fit_mask, mode, P, drop_qd)
    zl1 = residual_z(df, X, fit_mask, "avg_latency_us")
    d2 = df.assign(p99max=df[["read_latency_p99_us", "write_latency_p99_us"]].max(axis=1).to_numpy())
    zl2 = residual_z(d2, X, fit_mask, "p99max")
    zi = residual_z(df, X, fit_mask, "total_iops")
    return 0.5 * (zl1 + zl2), zi


if __name__ == "__main__":
    name = sys.argv[1]
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    df, tr = load(name)
    te = ~tr
    log = {}
    # ---------- hyper-parameter selection on a validation slice of the TRAINING period ----------
    if name == "base":
        vsplit = pd.Timestamp("2026-04-15")
        fit_v = (df["timestamp"] < vsplit).to_numpy(); val = tr & ~fit_v
        grid = []
        for mode in ["shape", "pred_full", "pred_shape"]:
            zl, zi = components(df, fit_v, mode)
            for a, k, b in itertools.product([1.0, 0.7, 0.5], [1, 2, 3], [0.5, 1.0, 2.0]):
                S, *_ = cars_scores(zl, zi, df, a, k, b)
                grid.append(dict(context=mode, alpha=a, k=k, beta=b,
                                 val_AP=average_precision_score(df.loc[val, "y"], S[val])))
            print(mode, "done", flush=True)
        grid = pd.DataFrame(grid).sort_values("val_AP", ascending=False)
        grid.to_csv(f"{OUT}/{name}/cars_grid.csv", index=False)
        best = grid.iloc[0].to_dict()
        json.dump(best, open(f"{OUT}/cars_params.json", "w"))
        print(grid.groupby("context").val_AP.max().round(4)); print(grid.head(5).round(4))
    best = json.load(open(f"{OUT}/cars_params.json"))
    a, k, b, mode = best["alpha"], int(best["k"]), best["beta"], best["context"]

    t0 = time.time()
    zl, zi = components(df, tr, mode)
    S, V, A, C = cars_scores(zl, zi, df, a, k, b)
    t_cars = time.time() - t0
    thr = np.percentile(S[tr], 97)
    np.savez(f"{OUT}/{name}/cars.npz", score=S[te], flag=S[te] > thr)
    np.savez(f"{OUT}/{name}/parts_cars.npz", V=V[te], A=A[te], C=C[te], zl=zl[te], zi=zi[te], thr=thr)
    log.update(threshold=float(thr), fit_and_score_s=t_cars, params=best)

    def save(tag, S):
        thr = np.percentile(S[tr], 97)
        np.savez(f"{OUT}/{name}/{tag}.npz", score=S[te], flag=S[te] > thr)
    save("abl_no_coherence", cars_scores(zl, zi, df, a, k, b, use_coh=False)[0])
    save("abl_no_persistence", cars_scores(zl, zi, df, 1.0, k, b)[0])
    save("abl_latency_only", cars_scores(zl, np.full_like(zi, -9), df, a, k, b)[0])
    if name == "base":
        for m2 in ["pred_full", "pred_shape", "oracle", "minimal"]:
            if m2 == mode: continue
            P = None
            if m2 in ("pred_full", "pred_shape"):
                P = workload_probs(df, tr, shape_only=(m2 == "pred_shape"))
                log[f"workload_acc_{m2}"] = float((P[te].argmax(1) == df.loc[te, "label"].to_numpy()).mean())
                surge = (df["surge"].to_numpy() > 0) & te
                log[f"workload_acc_on_surge_{m2}"] = float((P[surge].argmax(1) == df.loc[surge, "label"].to_numpy()).mean())
                ev = (df["role"].to_numpy() > 0) & te
                log[f"workload_acc_on_event_{m2}"] = float((P[ev].argmax(1) == df.loc[ev, "label"].to_numpy()).mean())
            z1, z2 = components(df, tr, m2, P=P)
            save(f"ctx_{m2}", cars_scores(z1, z2, df, a, k, b)[0])
        z1, z2 = components(df, tr, mode, drop_qd=True)
        save("abl_no_queue_depth", cars_scores(z1, z2, df, a, k, b)[0])

        def raw_z(col):
            x = np.log(df[col].to_numpy(float) + 1)
            s_ = pd.Series(x[tr]).groupby(df.loc[tr, "volume_id"].to_numpy())
            med = s_.median(); mad = s_.apply(lambda q: np.median(np.abs(q - np.median(q)))) * 1.4826
            v = pd.Series(df["volume_id"].to_numpy())
            return (x - v.map(med).to_numpy()) / v.map(mad).to_numpy()
        df["p99max"] = df[["read_latency_p99_us", "write_latency_p99_us"]].max(axis=1)
        zr = 0.5 * (raw_z("avg_latency_us") + raw_z("p99max"))
        save("abl_no_residual_model", cars_scores(zr, raw_z("total_iops"), df, a, k, b)[0])
        sens = []
        for kk, bb in itertools.product([1, 2, 3, 4], [0, 0.5, 1, 2, 4]):
            S2 = cars_scores(zl, zi, df, a, kk, bb)[0]
            sens.append(dict(k=kk, beta=bb, AP=average_precision_score(df.loc[te, "y"], S2[te])))
        pd.DataFrame(sens).to_csv(f"{OUT}/{name}/cars_sensitivity.csv", index=False)
    json.dump(log, open(f"{OUT}/{name}/cars_log.json", "w"), indent=1)
