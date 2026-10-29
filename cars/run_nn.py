"""Neural baselines: dense autoencoder (notebook architecture) and LSTM autoencoder (repo architecture),
then the repo's weighted ensemble and the notebook's >=2-of-4 consensus vote."""
import sys, os, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(__file__))
from common import load, FEAT10, OUT
from paths import PROJECT_REPO
sys.path.insert(0, os.path.join(PROJECT_REPO, "src", "models", "anomaly"))
torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42)

name = sys.argv[1]
df, tr = load(name); te = ~tr
sc = StandardScaler().fit(df.loc[tr, FEAT10])
X = sc.transform(df[FEAT10]).astype(np.float32)
Xtr, Xte = X[tr], X[te]
timing = {}


def save(m, score, flag, secs=None):
    np.savez(f"{OUT}/{name}/{m}.npz", score=np.asarray(score, float), flag=np.asarray(flag, bool))
    if secs is not None:
        timing[m] = secs / te.sum() * 1e6
    print(m, "alarm rate", round(float(np.mean(flag)), 4), flush=True)


# ---------------- dense autoencoder (notebook: d -> 2d -> d -> d/2 -> d -> 2d -> d) ----------------
class DenseAE(nn.Module):
    def __init__(s, d):
        super().__init__()
        h1, h2, z = 2 * d, d, max(d // 2, 2)
        s.net = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.BatchNorm1d(h1), nn.Dropout(0.1),
                              nn.Linear(h1, h2), nn.ReLU(), nn.BatchNorm1d(h2),
                              nn.Linear(h2, z), nn.ReLU(),
                              nn.Linear(z, h2), nn.ReLU(), nn.BatchNorm1d(h2),
                              nn.Linear(h2, h1), nn.ReLU(), nn.BatchNorm1d(h1),
                              nn.Linear(h1, d))
    def forward(s, x): return s.net(x)


def train(model, Xt, Xv, epochs, bs, lr=1e-3, wd=1e-5, patience=10):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
    best, bad, state, hist = 1e9, 0, None, []
    Xt = torch.tensor(Xt); Xv = torch.tensor(Xv)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt)); tl = 0
        for i in range(0, len(Xt), bs):
            xb = Xt[perm[i:i + bs]]
            opt.zero_grad(); loss = ((model(xb) - xb) ** 2).mean(); loss.backward(); opt.step()
            tl += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([((model(Xv[i:i + 4096]) - Xv[i:i + 4096]) ** 2).mean().item() for i in range(0, len(Xv), 4096)]))
        hist.append((tl / len(Xt), vl)); sch.step(vl)
        if vl < best - 1e-5:
            best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state); model.eval()
    return hist


def err_dense(model, A):
    with torch.no_grad():
        A = torch.tensor(A)
        return np.concatenate([((model(A[i:i + 8192]) - A[i:i + 8192]) ** 2).mean(1).numpy() for i in range(0, len(A), 8192)])


t = time.time()
idx = np.random.permutation(len(Xtr)); nv = len(idx) // 5
ae = DenseAE(Xtr.shape[1])
hist_ae = train(ae, Xtr[idx[nv:]], Xtr[idx[:nv]], epochs=100, bs=128)
e_tr = err_dense(ae, Xtr); t1 = time.time(); e_te = err_dense(ae, Xte)
thr_ae = np.percentile(e_tr, 97)
save("dense_autoencoder", e_te, e_te > thr_ae)
timing["dense_autoencoder"] = (time.time() - t1) / te.sum() * 1e6
json.dump(dict(hist=hist_ae, thr=float(thr_ae), epochs=len(hist_ae)), open(f"{OUT}/{name}/ae_hist.json", "w"))

# ---------------- LSTM autoencoder (repo LSTMAutoencoderModel, seq_len 12) ----------------
from lstm_autoencoder import LSTMAutoencoderModel
L = 12
order = np.lexsort((df["tidx"].to_numpy(), df["volume_id"].to_numpy()))   # volume-major, time-sorted
Xo = X[order]; vo = df["volume_id"].to_numpy()[order]
# windows ending at each row (rows with < L history in the volume are padded by repetition)
vol_start = pd.Series(np.arange(len(vo))).groupby(vo).transform("min").to_numpy()
end = np.arange(len(vo)); beg = np.maximum(end - L + 1, vol_start)
win_idx = np.clip(beg[:, None] + np.arange(L)[None, :], None, end[:, None])
tr_o = tr[order]; te_o = te[order]

lstm = LSTMAutoencoderModel(input_dim=X.shape[1], hidden_dim=64, latent_dim=8)
cand = np.where(tr_o)[0]; rng = np.random.default_rng(0)
sel = rng.choice(cand, 60000, replace=False); nv = 10000
Wt = Xo[win_idx[sel[nv:]]]; Wv = Xo[win_idx[sel[:nv]]]
t = time.time()


def train_seq(model, Wt, Wv, epochs=12, bs=256):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Wt = torch.tensor(Wt); Wv = torch.tensor(Wv); hist = []; best = 1e9; state = None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Wt)); tl = 0
        for i in range(0, len(Wt), bs):
            xb = Wt[perm[i:i + bs]]; opt.zero_grad()
            loss = ((model(xb) - xb) ** 2).mean(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([((model(Wv[i:i + 2048]) - Wv[i:i + 2048]) ** 2).mean().item() for i in range(0, len(Wv), 2048)]))
        hist.append((tl / len(Wt), vl)); print("lstm epoch", ep, round(vl, 4), flush=True)
        if vl < best: best, state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval(); return hist


hist_l = train_seq(lstm, Wt, Wv)


def err_seq(rows):
    out = np.empty(len(rows), np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), 4096):
            w = torch.tensor(Xo[win_idx[rows[i:i + 4096]]])
            out[i:i + 4096] = ((lstm(w) - w) ** 2).mean((1, 2)).numpy()
    return out


rows_tr = rng.choice(cand, 60000, replace=False)
e_ltr = err_seq(rows_tr); thr_l = np.percentile(e_ltr, 95); cap = np.percentile(e_ltr, 99)
t1 = time.time()
rows_te = np.where(te_o)[0]; e_lte_o = err_seq(rows_te)
timing["lstm_autoencoder"] = (time.time() - t1) / te.sum() * 1e6
# map back to the canonical (timestamp, volume) test order
inv = np.empty(len(order), int); inv[order] = np.arange(len(order))
canon_te = np.where(te)[0]
pos = np.searchsorted(rows_te, inv[canon_te])
e_lte = e_lte_o[pos]
save("lstm_autoencoder", e_lte, e_lte > thr_l)
json.dump(dict(hist=hist_l, thr=float(thr_l)), open(f"{OUT}/{name}/lstm_hist.json", "w"))

# ---------------- repo weighted ensemble (0.35 / 0.35 / 0.30, alarm 40, min agreement 1) -------------
stat = np.load(f"{OUT}/{name}/stat_all.npy")[te]
if_norm = np.load(f"{OUT}/{name}/if_norm.npy")
if_flag = np.load(f"{OUT}/{name}/isolation_forest.npz")["flag"]
l_norm = np.clip(e_lte / cap * 100, 0, 100)
ens = 0.35 * stat + 0.35 * if_norm + 0.30 * l_norm
agree = (stat >= 40).astype(int) + if_flag.astype(int) + (e_lte > thr_l).astype(int)
save("repo_ensemble", ens, (ens >= 40) & (agree >= 1))
np.save(f"{OUT}/{name}/ensemble_score_test.npy", ens)

# ---------------- notebook consensus: >=2 of {IF, LOF, DBSCAN, dense AE} ----------------
votes = np.zeros(te.sum()); ranks = np.zeros(te.sum())
for m in ["isolation_forest", "lof", "dbscan", "dense_autoencoder"]:
    z = np.load(f"{OUT}/{name}/{m}.npz"); votes += z["flag"]
    ranks += pd.Series(z["score"]).rank(pct=True).to_numpy()
save("consensus_2of4", votes + ranks / 4.0, votes >= 2)
json.dump(timing, open(f"{OUT}/{name}/timing_nn.json", "w"))
