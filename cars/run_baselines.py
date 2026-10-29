"""Classical baselines: static threshold, rolling 3-sigma (repo), IF (repo cfg), LOF and DBSCAN (notebook cfg)."""
import sys, os, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
sys.path.insert(0, os.path.dirname(__file__))
from common import load, FEAT10, OUT

name = sys.argv[1]
os.makedirs(f"{OUT}/{name}", exist_ok=True)
df, tr = load(name)
te = ~tr
timing = {}

def save(m, score, flag, secs):
    np.savez(f"{OUT}/{name}/{m}.npz", score=np.asarray(score, float), flag=np.asarray(flag, bool))
    timing[m] = secs / te.sum() * 1e6  # microseconds per test row
    print(m, "alarm rate", round(float(np.mean(flag)), 4), flush=True)

# 1. static fleet-wide threshold on tail latency (train 97th percentile)
t = time.time()
p99 = df[["read_latency_p99_us", "write_latency_p99_us"]].max(axis=1).to_numpy()
thr = np.percentile(p99[tr], 97)
save("static_threshold", p99[te], p99[te] > thr, time.time() - t)

# 2. rolling 24 h 3-sigma statistical detector (repo StatisticalHotspotDetector formula)
t = time.time()
mets = ["total_iops", "avg_latency_us", "total_throughput_mbps", "read_latency_p99_us", "write_latency_p99_us"]
g = df.sort_values(["volume_id", "tidx"])
z = {}
for m in mets:
    s = g.groupby("volume_id", observed=True)[m]
    mu = s.transform(lambda x: x.shift(1).rolling(288, min_periods=12).mean())
    sd = s.transform(lambda x: x.shift(1).rolling(288, min_periods=12).std())
    zz = ((g[m] - mu) / sd).where(sd > 1e-6, 0.0).fillna(0.0)
    z[m] = zz.reindex(df.index).to_numpy()
lat_z = np.maximum.reduce([z["avg_latency_us"], z["read_latency_p99_us"], z["write_latency_p99_us"]])
wz = 0.5 * lat_z + 0.3 * z["total_iops"] + 0.2 * z["total_throughput_mbps"]
stat = np.clip(wz / 7.0 * 100, 0, 100)
np.save(f"{OUT}/{name}/stat_all.npy", stat)
save("rolling_3sigma", stat[te], stat[te] >= 40, time.time() - t)

# shared scaling for multivariate detectors (fit on training period only)
sc = StandardScaler().fit(df.loc[tr, FEAT10])
Xtr = sc.transform(df.loc[tr, FEAT10]); Xte = sc.transform(df.loc[te, FEAT10])
np.save(f"{OUT}/{name}/Xte.npy", Xte.astype(np.float32))

# 3. Isolation Forest, repo configuration
t = time.time()
iso = IsolationForest(n_estimators=200, contamination=0.05, max_samples=256, random_state=42, n_jobs=-1).fit(Xtr)
raw_tr = iso.score_samples(Xtr); raw = iso.score_samples(Xte)
lo, hi = np.percentile(raw_tr, 98), np.percentile(raw_tr, 2)          # repo anchors
if_norm = np.clip((lo - raw) / (lo - hi) * 100, 0, 100)
np.save(f"{OUT}/{name}/if_norm.npy", if_norm)
save("isolation_forest", -raw, iso.predict(Xte) == -1, time.time() - t)

# 4. LOF, notebook configuration (n_neighbors=20, contamination=0.03), novelty mode
t = time.time()
rng = np.random.default_rng(0)
sub = rng.choice(len(Xtr), 40000, replace=False)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03, novelty=True, n_jobs=-1).fit(Xtr[sub])
save("lof", -lof.score_samples(Xte), lof.predict(Xte) == -1, time.time() - t)

# 5. DBSCAN, notebook configuration: min_samples = 2*d, eps from k-distance elbow, per test day
t = time.time()
d = Xte.shape[1]; ms = 2 * d
days = df.loc[te, "day"].to_numpy()
db_flag = np.zeros(len(Xte), bool); db_score = np.zeros(len(Xte))
eps_log = []
for dd in np.unique(days):
    ii = np.where(days == dd)[0]; X = Xte[ii]
    dist, _ = NearestNeighbors(n_neighbors=ms).fit(X).kneighbors(X)
    kd = dist[:, ms - 1]; ks = np.sort(kd)
    x = np.linspace(0, 1, len(ks)); yv = (ks - ks.min()) / (ks.max() - ks.min())
    line = np.array([1.0, yv[-1] - yv[0]]); line /= np.linalg.norm(line)
    pts = np.column_stack([x, yv - yv[0]])
    elbow = np.argmax(np.abs(pts[:, 0] * line[1] - pts[:, 1] * line[0]))
    eps = ks[elbow]; eps_log.append(float(eps))
    lab = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit_predict(X)
    db_flag[ii] = lab == -1; db_score[ii] = kd
print("DBSCAN eps per day", np.round(eps_log, 3))
save("dbscan", db_score, db_flag, time.time() - t)

pd.Series(timing).to_json(f"{OUT}/{name}/timing_baselines.json")
pd.Series(eps_log).to_json(f"{OUT}/{name}/dbscan_eps.json")
