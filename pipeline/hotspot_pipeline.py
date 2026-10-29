import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPRegressor

FEATURES = [
    "capacity_gb", "read_iops", "write_iops", "throughput_mbps",
    "avg_latency_ms", "p95_latency_ms", "p99_latency_ms", "queue_depth",
    "disk_utilization_pct", "controller_cpu_pct", "controller_memory_pct",
    "cache_hit_pct", "error_count", "timeout_count",
]

METRIC_MAP = {
    "latency": ["avg_latency_ms", "p95_latency_ms", "p99_latency_ms"],
    "slow": ["avg_latency_ms", "p99_latency_ms"],
    "queue": ["queue_depth"],
    "busy": ["disk_utilization_pct", "controller_cpu_pct"],
    "cpu": ["controller_cpu_pct"],
    "memory": ["controller_memory_pct"],
    "cache": ["cache_hit_pct"],
    "errors": ["error_count", "timeout_count"],
    "iops": ["read_iops", "write_iops"],
    "throughput": ["throughput_mbps"],
}


def simulate_telemetry(n_rows=20000, seed=42):
    rng = np.random.default_rng(seed)
    n_nodes = 20
    steps = n_rows // n_nodes
    ts = pd.date_range("2026-03-01", periods=steps, freq="5min")
    rows = []
    for node in range(n_nodes):
        cluster = node // 5
        hour = ts.hour.values
        diurnal = 1 + 0.35 * np.sin((hour - 6) / 24 * 2 * np.pi)
        read = rng.normal(4200, 500, steps) * diurnal
        write = rng.normal(2600, 350, steps) * diurnal
        q = rng.gamma(9, 2.6, steps) * diurnal
        util = np.clip(rng.normal(48, 8, steps) * diurnal, 5, 99)
        cpu = np.clip(0.6 * util + rng.normal(10, 5, steps), 3, 99)
        lat = 0.9 + 0.05 * q + rng.normal(0, 0.15, steps)
        hot = np.zeros(steps, dtype=int)
        for _ in range(rng.integers(1, 3)):
            start = rng.integers(0, steps - 40)
            length = rng.integers(8, 24)
            hot[start:start + length] = 1
        q = np.where(hot == 1, q * rng.uniform(5, 7, steps), q)
        util = np.where(hot == 1, np.clip(util + 40, 0, 100), util)
        cpu = np.where(hot == 1, np.clip(cpu + 30, 0, 100), cpu)
        lat = np.where(hot == 1, lat * rng.uniform(3, 6, steps), lat)
        write = np.where(hot == 1, write * 1.8, write)
        df = pd.DataFrame({
            "timestamp": ts,
            "cluster_id": f"CL{cluster:02d}",
            "node_id": f"N{node:03d}",
            "controller_id": f"CTRL{node % 2}",
            "capacity_gb": rng.normal(8000, 50, steps),
            "read_iops": read,
            "write_iops": write,
            "throughput_mbps": (read + write) * 0.064 + rng.normal(0, 15, steps),
            "avg_latency_ms": lat,
            "p95_latency_ms": lat * 1.9 + rng.normal(0, 0.2, steps),
            "p99_latency_ms": lat * 2.7 + rng.normal(0, 0.3, steps),
            "queue_depth": q,
            "disk_utilization_pct": util,
            "controller_cpu_pct": cpu,
            "controller_memory_pct": np.clip(rng.normal(55, 6, steps) + 0.2 * cpu, 0, 100),
            "cache_hit_pct": np.clip(rng.normal(92, 3, steps) - 25 * hot, 0, 100),
            "error_count": rng.poisson(0.2 + 3 * hot),
            "timeout_count": rng.poisson(0.05 + 2 * hot),
            "true_hotspot": hot,
        })
        rows.append(df)
    return pd.concat(rows).sort_values("timestamp").reset_index(drop=True)


PRIORITY = ["queue_depth", "avg_latency_ms", "disk_utilization_pct", "cache_hit_pct",
            "error_count", "timeout_count", "read_iops", "write_iops"]


def drop_correlated(df, cols, threshold=0.9):
    corr = df[cols].corr().abs()
    ordered = [c for c in PRIORITY if c in cols] + [c for c in cols if c not in PRIORITY]
    kept = []
    for c in ordered:
        if all(corr.loc[c, k] <= threshold for k in kept):
            kept.append(c)
    dropped = [c for c in cols if c not in kept]
    return [c for c in cols if c in kept], dropped


def time_split(df, train_frac=0.8):
    cut = df["timestamp"].quantile(train_frac)
    return df[df["timestamp"] <= cut].copy(), df[df["timestamp"] > cut].copy()


def run_detectors(df, cols, contamination=0.03):
    train, _ = time_split(df)
    scaler = StandardScaler().fit(train[cols])
    X_train = scaler.transform(train[cols])
    X_all = scaler.transform(df[cols])

    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iso.fit(X_train)
    df["iso_score"] = -iso.score_samples(X_all)
    df["iso_anomaly"] = (iso.predict(X_all) == -1).astype(int)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    df["lof_anomaly"] = (lof.fit_predict(X_all) == -1).astype(int)

    db = DBSCAN(eps=2.5, min_samples=10).fit(X_all)
    df["dbscan_anomaly"] = (db.labels_ == -1).astype(int)

    ae = MLPRegressor(hidden_layer_sizes=(16, 8, 16), activation="relu", solver="adam",
                      learning_rate_init=0.001, batch_size=64, max_iter=30, random_state=42)
    ae.fit(X_train, X_train)
    recon_error = np.mean((X_all - ae.predict(X_all)) ** 2, axis=1)
    threshold = np.percentile(recon_error, 97)
    df["ae_error"] = recon_error
    df["ae_anomaly"] = (recon_error > threshold).astype(int)

    votes = df[["iso_anomaly", "lof_anomaly", "dbscan_anomaly", "ae_anomaly"]].sum(axis=1)
    df["votes"] = votes
    df["consensus_anomaly"] = (votes >= 2).astype(int)
    df["severity"] = pd.cut(votes, bins=[-1, 1, 2, 3, 4],
                            labels=["Normal", "Watch", "Hotspot", "Critical"])
    return df, ae, threshold


def explain_row(row, baseline, metric_map=METRIC_MAP):
    reasons = []
    for col in FEATURES:
        mu, sd = baseline.loc["mean", col], baseline.loc["std", col]
        if sd == 0:
            continue
        z = (row[col] - mu) / sd
        if abs(z) >= 3:
            direction = "above" if z > 0 else "below"
            reasons.append((abs(z), f"{col} is {row[col]:.1f} ({z:+.1f} sigma {direction} normal {mu:.1f})"))
    reasons.sort(reverse=True)
    lines = [r for _, r in reasons[:4]]
    action = "Monitor"
    if row["queue_depth"] > baseline.loc["mean", "queue_depth"] * 3:
        action = "Rebalance I/O away from this node or throttle the noisy workload"
    elif row["cache_hit_pct"] < baseline.loc["mean", "cache_hit_pct"] - 10:
        action = "Review cache allocation for the volumes on this controller"
    return {
        "node_id": row["node_id"],
        "timestamp": str(row["timestamp"]),
        "severity": str(row["severity"]),
        "votes": int(row["votes"]),
        "evidence": lines,
        "recommendation": action,
    }


def resolve_metrics(question, metric_map=METRIC_MAP):
    q = question.lower()
    cols = []
    for phrase, mapped in metric_map.items():
        if phrase in q:
            cols.extend(c for c in mapped if c not in cols)
    return cols or ["queue_depth", "avg_latency_ms"]


def simulate_noisy_neighbour(n_rows=20000, seed=11, events_per_cluster=3):
    df = simulate_telemetry(n_rows=n_rows, seed=seed)
    rng = np.random.default_rng(seed + 1)
    df["nn_role"] = "none"
    df["nn_event"] = -1
    times = np.sort(df["timestamp"].unique())
    event_id = 0
    for cluster in sorted(df["cluster_id"].unique()):
        nodes = sorted(df.loc[df["cluster_id"] == cluster, "node_id"].unique())
        used = []
        for _ in range(events_per_cluster):
            while True:
                start = int(rng.integers(len(times) // 10, len(times) - 40))
                if all(abs(start - u) > 60 for u in used):
                    break
            used.append(start)
            length = int(rng.integers(12, 25))
            window = times[start:start + length]
            aggressor = nodes[int(rng.integers(0, len(nodes)))]
            in_win = df["timestamp"].isin(window) & (df["cluster_id"] == cluster)
            agg = in_win & (df["node_id"] == aggressor)
            vic = in_win & (df["node_id"] != aggressor)
            for c in ("read_iops", "write_iops"):
                df.loc[agg, c] *= 3.5
            df.loc[agg, "throughput_mbps"] *= 3.2
            df.loc[agg, "queue_depth"] *= 2.5
            df.loc[agg, "avg_latency_ms"] *= 1.8
            df.loc[agg, "controller_cpu_pct"] = np.clip(df.loc[agg, "controller_cpu_pct"] + 25, 0, 100)
            for c in ("read_iops", "write_iops"):
                df.loc[vic, c] *= 0.85
            df.loc[vic, "queue_depth"] *= rng.uniform(2.5, 3.5)
            df.loc[vic, "avg_latency_ms"] *= rng.uniform(2.5, 3.5)
            df.loc[vic, "cache_hit_pct"] = np.clip(df.loc[vic, "cache_hit_pct"] - 15, 0, 100)
            df.loc[vic, "timeout_count"] += rng.poisson(1.0, vic.sum())
            for m in ("p95_latency_ms", "p99_latency_ms"):
                df.loc[in_win, m] = df.loc[in_win, "avg_latency_ms"] * (1.9 if m == "p95_latency_ms" else 2.7)
            df.loc[agg, "nn_role"] = "aggressor"
            df.loc[vic, "nn_role"] = "victim"
            df.loc[in_win, "nn_event"] = event_id
            event_id += 1
    return df


def workload_baseline(train, robust=True):
    t = train.copy()
    t["hour"] = t["timestamp"].dt.hour
    t["demand"] = t["read_iops"] + t["write_iops"]
    def robust_std(x):
        return 1.4826 * np.median(np.abs(x - np.median(x)))
    g = t.groupby(["node_id", "hour"])[["demand", "avg_latency_ms"]]
    if not robust:
        return g.agg(["mean", "std"])
    return g.agg([("mean", "median"), ("std", robust_std)])


def attribute_noisy_neighbours(df, baseline, demand_z=3.0, quiet_z=1.5, latency_z=3.0, min_ratio=1.5):
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["demand"] = out["read_iops"] + out["write_iops"]
    b = baseline.copy()
    b.columns = ["_".join(c) for c in b.columns]
    out = out.join(b, on=["node_id", "hour"])
    out["demand_z"] = (out["demand"] - out["demand_mean"]) / out["demand_std"]
    out["latency_z"] = (out["avg_latency_ms"] - out["avg_latency_ms_mean"]) / out["avg_latency_ms_std"]
    out["nn_pred"] = "none"
    out["culprit"] = ""
    for (ts, cl), g in out.groupby(["timestamp", "cluster_id"]):
        degraded = (g["consensus_anomaly"] == 1) | (g["latency_z"] >= latency_z)
        victims = g[degraded & (g["demand_z"] < quiet_z)]
        surging = g[(g["demand_z"] >= demand_z) & (g["demand"] >= min_ratio * g["demand_mean"])]
        if len(victims) and len(surging):
            agg_idx = surging["demand_z"].idxmax()
            out.loc[agg_idx, "nn_pred"] = "aggressor"
            out.loc[victims.index, "nn_pred"] = "victim"
            out.loc[victims.index, "culprit"] = out.loc[agg_idx, "node_id"]
        else:
            local = g[g["consensus_anomaly"] == 1].index
            out.loc[local, "nn_pred"] = "node-local hotspot"
    return out


def summarise_events(att, min_intervals=2):
    rows = []
    v = att[att["nn_pred"] == "victim"]
    if v.empty:
        return pd.DataFrame()
    v = v.sort_values("timestamp")
    for (cl, culprit), g in v.groupby(["cluster_id", "culprit"]):
        ts = g["timestamp"].drop_duplicates().sort_values()
        gaps = ts.diff().gt(pd.Timedelta("15min")).cumsum()
        for _, seg in ts.groupby(gaps):
            seg_rows = g[g["timestamp"].isin(seg)]
            agg = att[(att["timestamp"].isin(seg)) & (att["node_id"] == culprit)]
            rows.append({
                "cluster_id": cl, "aggressor": culprit,
                "start": seg.min(), "end": seg.max(), "intervals": len(seg),
                "victims": sorted(seg_rows["node_id"].unique()),
                "peak_demand_ratio": round(float((agg["demand"] / agg["demand_mean"]).max()), 2),
                "victim_latency_ratio": round(float((seg_rows["avg_latency_ms"] / seg_rows["avg_latency_ms_mean"]).mean()), 2),
                "recommendation": f"Throttle or migrate the heaviest workload on {culprit}; "
                                  f"its I/O demand is starving {seg_rows['node_id'].nunique()} neighbour node(s)",
            })
    ev = pd.DataFrame(rows)
    ev = ev[ev["intervals"] >= min_intervals]
    return ev.sort_values("start").reset_index(drop=True)


if __name__ == "__main__":
    data = simulate_telemetry()
    kept, dropped = drop_correlated(data, FEATURES)
    print("Dropped (corr > 0.9):", dropped)
    data, ae_model, thr = run_detectors(data, kept)
    print(data[["iso_anomaly", "lof_anomaly", "dbscan_anomaly", "ae_anomaly", "consensus_anomaly"]].sum())
    print("AE threshold (97th pct):", round(thr, 4))
    data.to_csv("final_hotspot_output.csv", index=False)

    nn = simulate_noisy_neighbour()
    kept_nn, _ = drop_correlated(nn, FEATURES)
    nn, _, _ = run_detectors(nn, kept_nn)
    train_nn, _ = time_split(nn)
    base = workload_baseline(train_nn[train_nn["consensus_anomaly"] == 0])
    attributed = attribute_noisy_neighbours(nn, base)
    events = summarise_events(attributed)
    print("Noisy-neighbour alerts:", len(events))
    print(events[["cluster_id", "aggressor", "start", "end", "intervals"]].to_string(index=False))
    events.to_csv("noisy_neighbour_events.csv", index=False)
