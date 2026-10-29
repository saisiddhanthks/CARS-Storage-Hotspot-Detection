"""Regenerate the team repo's synthetic telemetry with ground-truth labels.

Uses src/data/data_generator.py unchanged, except two functions are replaced by
copies that perform the identical RNG draws but also record which rows were
modified (noisy-neighbour aggressor/victim rows and benign load surges).
Optional severity knobs allow robustness variants.
"""
import sys, json, argparse
import numpy as np, pandas as pd
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PROJECT_REPO, DATA
sys.path.insert(0, os.path.join(PROJECT_REPO, "src", "data"))
import data_generator as dg

LABELS = {}
CFG = dict(lat_lo=1.4, lat_hi=2.2, victim_frac=1.0, iops_lo=1.4, iops_hi=1.9)

def add_time_patterns(df, volume_dominant, seed=dg.SEED + 11):
    rng = np.random.default_rng(seed)
    hour = df["timestamp"].dt.hour.to_numpy()
    day_of_year = df["timestamp"].dt.dayofyear.to_numpy()
    phase_mean = {"DB_OLTP": 13.0, "VM": 12.0, "Backup": 2.0, "AI_Training": 15.0, "AI_Inference": 14.0}
    amp_mean = {"DB_OLTP": 0.45, "VM": 0.35, "Backup": 0.55, "AI_Training": 0.15, "AI_Inference": 0.25}
    volume_phase, volume_amp = {}, {}
    for vol, dom in volume_dominant.items():
        volume_phase[vol] = rng.normal(phase_mean[dom], 3.0) % 24
        volume_amp[vol] = np.clip(rng.normal(amp_mean[dom], 0.12), 0.05, 0.8)
    phase = df["volume_id"].map(volume_phase).to_numpy()
    amp = df["volume_id"].map(volume_amp).to_numpy()
    day_shift = 0.6 * np.sin(2 * np.pi * (day_of_year / 7))
    noise = rng.normal(0, 0.05, len(df))
    tm = 1 + amp * np.sin(2 * np.pi * (hour - phase + day_shift) / 24) + noise
    anomaly_mask = rng.random(len(df)) < 0.02
    tm[anomaly_mask] *= rng.uniform(1.2, 1.6, anomaly_mask.sum())
    tm = np.clip(tm, 0.6, 1.6)
    LABELS["surge"] = anomaly_mask.copy()
    df["read_iops"] = (df["read_iops"] * tm).round(0).astype(int)
    df["write_iops"] = (df["write_iops"] * tm).round(0).astype(int)
    df["total_iops"] = df["read_iops"] + df["write_iops"]
    df["read_throughput_mbps"] = (df["read_throughput_mbps"] * tm).round(2)
    df["write_throughput_mbps"] = (df["write_throughput_mbps"] * tm).round(2)
    ls = 1 + (tm - 1) * 0.15
    for col in ["read_latency_p50_us","read_latency_p95_us","read_latency_p99_us",
                "write_latency_p50_us","write_latency_p95_us","write_latency_p99_us"]:
        df[col] = (df[col] * ls).round(2)
    return df

def inject_noisy_neighbor_events(df, volumes, volume_to_node, n_events=500, duration_minutes=15, seed=dg.SEED + 13):
    rng = np.random.default_rng(seed)
    rng_v = np.random.default_rng(seed + 1000)  # only used for partial-victim variants
    num_timestamps = dg.NUM_DAYS * 24 * (60 // dg.INTERVAL_MINUTES)
    n = len(df)
    role = np.zeros(n, dtype=np.int8)        # 0 none, 1 aggressor, 2 victim
    event_id = np.full(n, -1, dtype=np.int32)
    volume_pos = {v: i for i, v in enumerate(volumes)}
    node_to_vols = {}
    for v in volumes:
        node_to_vols.setdefault(volume_to_node[v], []).append(v)
    starts = rng.integers(0, num_timestamps - duration_minutes, size=n_events)
    noisy_vols = rng.choice(volumes, size=n_events, replace=True)
    lat_cols = ["read_latency_p50_us","read_latency_p95_us","read_latency_p99_us",
                "write_latency_p50_us","write_latency_p95_us","write_latency_p99_us"]
    events = []
    for k, (s, nv) in enumerate(zip(starts, noisy_vols)):
        node = volume_to_node[nv]
        neigh = [v for v in node_to_vols[node] if v != nv]
        t_range = np.arange(s, s + duration_minutes)
        base = t_range * dg.NUM_VOLUMES
        nrows = base + volume_pos[nv]
        # identical draw order to the original generator
        iops_mult = rng.uniform(1.4, 1.9); thr_mult = rng.uniform(1.2, 1.6)
        lat_mult = rng.uniform(1.4, 2.2); thr_down = rng.uniform(0.7, 0.9)
        # optional severity remapping (deterministic transform of the same draws)
        iops_mult = CFG["iops_lo"] + (iops_mult - 1.4) / 0.5 * (CFG["iops_hi"] - CFG["iops_lo"])
        lat_mult = CFG["lat_lo"] + (lat_mult - 1.4) / 0.8 * (CFG["lat_hi"] - CFG["lat_lo"])
        if CFG["victim_frac"] < 1.0:
            m = max(1, int(round(len(neigh) * CFG["victim_frac"])))
            neigh = list(rng_v.choice(neigh, size=m, replace=False))
        npos = np.array([volume_pos[v] for v in neigh], dtype=int)
        vrows = (base[:, None] + npos[None, :]).reshape(-1)
        for c in ["read_iops", "write_iops"]:
            df.loc[nrows, c] = (df.loc[nrows, c] * iops_mult).round(0).astype(int)
        df.loc[nrows, "total_iops"] = df.loc[nrows, "read_iops"] + df.loc[nrows, "write_iops"]
        for c in ["read_throughput_mbps", "write_throughput_mbps"]:
            df.loc[nrows, c] = (df.loc[nrows, c] * thr_mult).round(2)
        for c in lat_cols:
            df.loc[vrows, c] = (df.loc[vrows, c] * lat_mult).round(2)
        for c in ["read_throughput_mbps", "write_throughput_mbps"]:
            df.loc[vrows, c] = (df.loc[vrows, c] * thr_down).round(2)
        role[nrows] = np.where(role[nrows] == 0, 1, role[nrows])
        role[vrows] = np.where(role[vrows] == 0, 2, role[vrows])
        event_id[nrows] = np.where(event_id[nrows] < 0, k, event_id[nrows])
        event_id[vrows] = np.where(event_id[vrows] < 0, k, event_id[vrows])
        events.append(dict(event=k, start=int(s), dur=int(duration_minutes), node=node,
                           aggressor=nv, victims=neigh, iops_mult=float(iops_mult),
                           lat_mult=float(lat_mult)))
    LABELS["role"] = role; LABELS["event_id"] = event_id; LABELS["events"] = events
    return df

dg.add_time_patterns = add_time_patterns
dg.inject_noisy_neighbor_events = inject_noisy_neighbor_events

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lat", default="1.4,2.2")
    ap.add_argument("--iops", default="1.4,1.9")
    ap.add_argument("--victim_frac", type=float, default=1.0)
    a = ap.parse_args()
    CFG["lat_lo"], CFG["lat_hi"] = map(float, a.lat.split(","))
    CFG["iops_lo"], CFG["iops_hi"] = map(float, a.iops.split(","))
    CFG["victim_frac"] = a.victim_frac
    df = dg.generate_time_series_data()
    df["role"] = LABELS["role"]; df["event_id"] = LABELS["event_id"]; df["surge"] = LABELS["surge"]
    for c in ["workload_type","tier","node_id","pool_id","volume_id"]:
        df[c] = df[c].astype(str)
    df.to_parquet(os.path.join(DATA, a.out + ".parquet"), index=False)
    json.dump(dict(cfg=CFG, events=LABELS["events"]), open(os.path.join(DATA, a.out + "_events.json"), "w"))
    print(df.shape, (df.role==1).sum(), (df.role==2).sum(), df.surge.sum())
