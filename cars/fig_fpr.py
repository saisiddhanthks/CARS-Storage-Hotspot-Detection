import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FIG, OUT
from style import *
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from aggregate import NAMES
r = pd.read_csv(f"{OUT}/fpr_by_workload.csv", index_col=0)
r = r.loc[["static_threshold", "isolation_forest", "repo_ensemble", "consensus_2of4", "dbscan", "rolling_3sigma", "cars"]]
cm = LinearSegmentedColormap.from_list("seq", ["#f7f9fd", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
fig, ax = plt.subplots(figsize=(3.45, 2.35))
im = ax.imshow(r.values, cmap=cm, aspect="auto", vmin=0, vmax=10.5)
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        v = r.values[i, j]
        ax.text(j, i, f"{v:.2f}" if v < 10 else f"{v:.1f}", ha="center", va="center", fontsize=6.8, color="white" if v > 5 else INK,
                fontweight="bold" if r.index[i] == "cars" else "normal")
ax.set_xticks(range(5)); ax.set_xticklabels(["DB_OLTP", "VM", "Backup", "AI_Train.", "AI_Infer."], fontsize=7.2)
ax.set_yticks(range(r.shape[0])); ax.set_yticklabels([NAMES[m] for m in r.index], fontsize=7.2)
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5); cb.set_label("False-positive rate (%)", fontsize=7)
plt.savefig(f"{FIG}/fig_fpr_workload.png"); print("ok")
