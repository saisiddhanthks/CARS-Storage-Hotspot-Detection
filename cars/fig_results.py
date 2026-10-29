import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from style import *
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from common import load, events, OUT
from aggregate import NAMES, ABL

from paths import FIG
main = pd.read_csv(f"{OUT}/main_base.csv", index_col=0)
abl = pd.read_csv(f"{OUT}/ablation_base.csv", index_col=0)
df, tr = load("base"); test = df[~tr].reset_index(drop=True); y = test.y.to_numpy()

# ---------- Fig: precision-recall curves ----------
top = main.drop(index="cars").sort_values("AP", ascending=False).index[:3].tolist()
fig, ax = plt.subplots(figsize=(3.45, 2.9))
for m in main.index:
    s = np.load(f"{OUT}/base/{m}.npz")["score"]
    p, r, _ = precision_recall_curve(y, s)
    if m == "cars":
        ax.plot(r, p, color=C[0], lw=2.0, label=f"{NAMES[m]}  AP {main.loc[m,'AP']:.3f}", zorder=5)
    elif m in top:
        i = top.index(m) + 1
        ax.plot(r, p, color=C[i], lw=1.3, label=f"{NAMES[m]}  AP {main.loc[m,'AP']:.3f}")
    else:
        ax.plot(r, p, color=GRAY, lw=0.8, alpha=0.8)
ax.plot([], [], color=GRAY, lw=0.8, label="Other baselines")
ax.axhline(y.mean(), color=INK2, lw=0.6)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
h, l = ax.get_legend_handles_labels(); i = [k for k, t in enumerate(l) if t.startswith("CARS")][0]
h = [h[i]] + h[:i] + h[i+1:]; l = [l[i]] + l[:i] + l[i+1:]
lg = ax.legend(h, l, loc="lower left", fontsize=6.4, handlelength=1.2, bbox_to_anchor=(0.0, 0.06), frameon=True, framealpha=0.92, edgecolor="none")
plt.savefig(f"{FIG}/fig_pr.png"); plt.close()

# ---------- Fig: AP with CI, surge FPR, false alarms ----------
order = main.sort_values("AP").index.tolist()
lab = [NAMES[m] for m in order]
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.75), sharey=True, gridspec_kw=dict(wspace=0.16))
col = [C[0] if m == "cars" else "#8fb4e3" for m in order]
yy = np.arange(len(order))
ax = axs[0]
ax.barh(yy, main.loc[order, "AP"], color=col, height=0.62)
ax.errorbar(main.loc[order, "AP"], yy, xerr=[main.loc[order, "AP"] - main.loc[order, "AP_lo"], main.loc[order, "AP_hi"] - main.loc[order, "AP"]],
            fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
ax.set_yticks(yy); ax.set_yticklabels(lab, fontsize=7.4); ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", ".25", ".50", ".75", "1"]); ax.set_title("(a) Average precision (95% CI)", loc="left", fontsize=8.5)
ax.grid(axis="y", visible=False)
ax = axs[1]
v = main.loc[order, "surge_FPR"] * 100
ax.barh(yy, v, color=["#eb6834" if m == "cars" else "#f3b597" for m in order], height=0.62)
for yi, val in zip(yy, v): ax.text(val + 0.2, yi, f"{val:.1f}", va="center", fontsize=6.6, color=INK2)
ax.set_title("(b) Alarms on benign surges (%)", loc="left", fontsize=8.5); ax.grid(axis="y", visible=False); ax.set_xlim(0, max(v) * 1.25)
ax = axs[2]
v = main.loc[order, "fa_per_volume_day"]
ax.barh(yy, v, color=["#1baf7a" if m == "cars" else "#9fdcc4" for m in order], height=0.62)
for yi, val in zip(yy, v): ax.text(val + 0.1, yi, f"{val:.2f}", va="center", fontsize=6.6, color=INK2)
ax.set_title("(c) False-alarm episodes / volume-day", loc="left", fontsize=8.5); ax.grid(axis="y", visible=False); ax.set_xlim(0, max(v) * 1.22)
plt.savefig(f"{FIG}/fig_main_bars.png"); plt.close()

# ---------- Fig: ablation ----------
ab_order = ["cars", "abl_no_persistence", "abl_no_queue_depth", "ctx_minimal", "abl_latency_only", "abl_no_residual_model", "abl_no_coherence"]
fig, ax = plt.subplots(figsize=(3.45, 2.3))
vals = abl.loc[ab_order, "AP"]; yy = np.arange(len(ab_order))[::-1]
ax.barh(yy, vals, color=[C[0] if m == "cars" else "#8fb4e3" for m in ab_order], height=0.6)
for yi, m in zip(yy, ab_order):
    d = abl.loc[m, "AP"] - abl.loc["cars", "AP"]
    ax.text(abl.loc[m, "AP"] + 0.004, yi, f"{abl.loc[m,'AP']:.3f}" + ("" if m == "cars" else f"  ({d:+.3f})".replace("-", "−")), va="center", fontsize=6.8, color=INK2)
ax.set_yticks(yy); ax.set_yticklabels([ABL[m] for m in ab_order], fontsize=7.4)
ax.set_xlim(0.75, 1.0); ax.set_xlabel("Average precision"); ax.grid(axis="y", visible=False)
plt.savefig(f"{FIG}/fig_ablation.png"); plt.close()

# ---------- Fig: sensitivity heatmap ----------
sens = pd.read_csv(f"{OUT}/base/cars_sensitivity.csv").pivot(index="k", columns="beta", values="AP")
fig, ax = plt.subplots(figsize=(3.0, 2.0))
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#1c5cab", "#0d366b"])
im = ax.imshow(sens.values, cmap=cm, aspect="auto", vmin=0.8, vmax=1.0)
for i in range(sens.shape[0]):
    for j in range(sens.shape[1]):
        v = sens.values[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.8, color="white" if v > 0.93 else INK,
                fontweight="bold" if (sens.index[i] == 3 and sens.columns[j] == 2.0) else "normal")
ax.set_xticks(range(sens.shape[1])); ax.set_xticklabels([f"{b:g}" for b in sens.columns]); ax.set_yticks(range(sens.shape[0])); ax.set_yticklabels(sens.index)
ax.set_xlabel("Coherence weight β"); ax.set_ylabel("Peers averaged k"); ax.grid(False)
[s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5); cb.set_label("Test AP", fontsize=7)
plt.savefig(f"{FIG}/fig_sensitivity.png"); plt.close()

# ---------- Fig: robustness ----------
vs = ["base", "partial", "subtle", "subtle_partial"]; vlab = ["Default", "Partial", "Subtle", "Subtle +\npartial"]
R = {v: pd.read_csv(f"{OUT}/rob_{v}.csv", index_col=0) for v in vs}
show = ["cars", "abl_no_coherence"] + [m for m in main.drop(index="cars").sort_values("AP", ascending=False).index[:3]]
fig, ax = plt.subplots(figsize=(3.45, 2.6))
names2 = dict(NAMES, abl_no_coherence="CARS without coherence")
for i, m in enumerate(show):
    xs = [j for j, v in enumerate(vs) if v in R and m in R[v].index]
    ys = [R[vs[j]].loc[m, "AP"] for j in xs]
    ax.plot(xs, ys, marker="o", ms=4, lw=1.8 if m == "cars" else 1.2, color=C[i], label=names2[m],
            ls="-" if i != 1 else (0, (3, 1.5)))
ax.set_xticks(range(len(vs))); ax.set_xticklabels(vlab, fontsize=7.5); ax.set_ylabel("Average precision"); ax.set_ylim(0, 1)
ax.legend(fontsize=6.4, loc="lower left", handlelength=1.6)
plt.savefig(f"{FIG}/fig_robustness.png"); plt.close()

# ---------- Fig: case study ----------
meta = json.load(open(f"{OUT}/motivation_meta.json")); e = meta["event"]
parts = np.load(f"{OUT}/base/parts_cars.npz"); thr = float(parts["thr"])
test["V"] = parts["V"]; test["A"] = parts["A"]
ifs = np.load(f"{OUT}/base/isolation_forest.npz"); test["IF"] = ifs["score"]
w = test[(test.tidx.between(e["start"] - 15, e["start"] + 15)) & (test.node_id == e["node"])]
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.2))
t = lambda s: (s.tidx - e["start"]) * 5
vols = [(e["aggressor"], "aggressor"), (e["victims"][0], "victim"), (e["victims"][1], "victim")]
for i, (v, r) in enumerate(vols):
    s = w[w.volume_id == v]
    axs[0].plot(t(s), np.maximum(s.V, s.A), color=C[i], lw=1.3, label=f"{r} {v}")
others = [u for u in w.volume_id.unique() if u not in [x for x, _ in vols]]
axs[0].axhline(thr, color=INK2, lw=0.8); axs[0].text(-74, thr + 0.6, "alarm threshold", fontsize=6.6, color=INK2)
axs[0].axvspan(-2.5, 12.5, color="#e34948", alpha=0.10, lw=0)
axs[0].set_title("(a) CARS hotspot score S", loc="left"); axs[0].set_xlabel("Minutes from event onset")
axs[0].legend(fontsize=6.4, loc="upper right", handlelength=1.0)
ifthr = np.percentile(test.IF[ifs["flag"]], 0)
for i, (v, r) in enumerate(vols):
    s = w[w.volume_id == v]
    axs[1].plot(t(s), s.IF, color=C[i], lw=1.3)
axs[1].axhline(ifthr, color=INK2, lw=0.8); axs[1].text(-74, ifthr + 0.004, "alarm threshold", fontsize=6.6, color=INK2)
axs[1].axvspan(-2.5, 12.5, color="#e34948", alpha=0.10, lw=0)
axs[1].set_title("(b) Isolation Forest anomaly score", loc="left"); axs[1].set_xlabel("Minutes from event onset")
plt.tight_layout(w_pad=1.5); plt.savefig(f"{FIG}/fig_case.png"); plt.close()
print("figures done")
