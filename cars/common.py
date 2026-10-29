"""Shared loading, splitting and evaluation utilities for the hotspot benchmark."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from paths import DATA, OUT, FIG, PROJECT_REPO  # noqa: F401
SPLIT = pd.Timestamp("2026-04-22")       # days 1-21 train, 22-30 test
FEAT10 = ["total_iops", "avg_latency_us", "total_throughput_mbps", "read_latency_p99_us",
          "write_latency_p99_us", "sequential_ratio", "read_write_ratio", "io_size_entropy",
          "queue_depth", "iops_per_queue"]                       # repo EnsembleDetector.FEATURE_COLS
WORKLOADS = ["DB_OLTP", "VM", "Backup", "AI_Training", "AI_Inference"]


def load(name):
    df = pd.read_parquet(f"{DATA}/{name}_feat.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp", "volume_id"]).reset_index(drop=True)
    df["y"] = (df["role"] > 0).astype(np.int8)
    df["tidx"] = ((df["timestamp"] - df["timestamp"].min()) / pd.Timedelta("5min")).astype(int)
    df["day"] = df["timestamp"].dt.normalize()
    tr = df["timestamp"] < SPLIT
    return df, tr.to_numpy()


def events(name):
    return json.load(open(f"{DATA}/{name}_events.json"))["events"]


def best_f1(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f[:-1]))
    return float(f[i]), float(p[i]), float(r[i]), float(t[i])


def false_alarm_episodes(test, flag):
    """Contiguous runs of flagged rows on negative (non-event) rows, per volume."""
    d = test[["volume_id", "tidx", "y"]].copy()
    d["f"] = flag.astype(bool) & (d["y"].to_numpy() == 0)
    d = d.sort_values(["volume_id", "tidx"])
    f = d["f"].to_numpy()
    same_vol = d["volume_id"].to_numpy()[1:] == d["volume_id"].to_numpy()[:-1]
    starts = f.copy()
    starts[1:] = f[1:] & ~(f[:-1] & same_vol)
    n_vol_days = test["volume_id"].nunique() * test["day"].nunique()
    return int(starts.sum()), starts.sum() / n_vol_days


def event_metrics(test, flag, evs):
    """Event recall and detection delay (minutes) for injected interference events."""
    t0 = int(test["tidx"].min()); t1 = int(test["tidx"].max())
    key = dict(zip(zip(test["tidx"].to_numpy(), test["volume_id"].to_numpy()), flag.astype(bool)))
    det, delays = 0, []
    n = 0; strict = 0; covs = []
    for e in evs:
        if e["start"] < t0 or e["start"] + e["dur"] - 1 > t1:
            continue
        n += 1
        vols = [e["aggressor"]] + list(e["victims"])
        first = None; best_cov = 0.0
        for k in range(e["dur"]):
            hits = [key.get((e["start"] + k, v), False) for v in vols]
            best_cov = max(best_cov, float(np.mean(hits)))
            if first is None and any(hits):
                first = k
        if first is not None:
            det += 1
            delays.append(first * 5)
        strict += best_cov >= 0.5; covs.append(best_cov)
    return dict(n_events=n, event_recall=det / max(n, 1), event_recall_strict=strict / max(n, 1),
                mean_coverage=float(np.mean(covs)) if covs else float("nan"),
                mean_delay_min=float(np.mean(delays)) if delays else float("nan"))


def evaluate(test, score, flag, evs, name):
    y = test["y"].to_numpy()
    surge = (test["surge"].to_numpy() > 0) & (y == 0)
    neg = y == 0
    flag = flag.astype(bool)
    tp = int((flag & (y == 1)).sum()); fp = int((flag & neg).sum()); fn = int((~flag & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    s = np.nan_to_num(score, nan=np.nanmin(score))
    bf, bp, br, bt = best_f1(y, s)
    n_ep, ep_rate = false_alarm_episodes(test, flag)
    em = event_metrics(test, flag, evs)
    role = test["role"].to_numpy()
    return dict(method=name, AP=average_precision_score(y, s), ROC_AUC=roc_auc_score(y, s),
                precision=prec, recall=rec, F1=f1, alarm_rate=float(flag.mean()),
                FPR=fp / max(neg.sum(), 1), surge_FPR=float(flag[surge].mean()),
                aggressor_recall=float(flag[role == 1].mean()), victim_recall=float(flag[role == 2].mean()),
                best_F1=bf, false_alarm_episodes=n_ep, fa_per_volume_day=ep_rate, **em)


def block_bootstrap_ci(test, score, flag, n_boot=300, seed=0):
    """95% CI for AP and F1 by resampling (node, day) blocks."""
    rng = np.random.default_rng(seed)
    y = test["y"].to_numpy(); s = np.nan_to_num(score, nan=np.nanmin(score)); f = flag.astype(bool)
    blocks = (test["node_id"].astype(str) + "|" + test["day"].astype(str)).to_numpy()
    ub, inv = np.unique(blocks, return_inverse=True)
    idx_by = [np.where(inv == b)[0] for b in range(len(ub))]
    aps, f1s = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ub), len(ub))
        ii = np.concatenate([idx_by[p] for p in pick])
        yy = y[ii]
        if yy.sum() == 0:
            continue
        aps.append(average_precision_score(yy, s[ii]))
        tp = (f[ii] & (yy == 1)).sum(); fp = (f[ii] & (yy == 0)).sum(); fn = (~f[ii] & (yy == 1)).sum()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1s.append(2 * p * r / max(p + r, 1e-12))
    return np.percentile(aps, [2.5, 97.5]).tolist(), np.percentile(f1s, [2.5, 97.5]).tolist()
