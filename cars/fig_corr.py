import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FIG
from style import *
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
names = ["Read IOPS","Write IOPS","Total IOPS","Throughput","Mean latency","p95 latency","p99 latency","Queue depth",
         "Disk utilisation","Controller CPU","Controller memory","Cache hit ratio","I/O wait","Error count","Timeout count"]
# transcribed from the Study 1 notebook's Pearson correlation output (100,000 records)
M = np.array([
[1.00,0.44,0.94,0.35,0.31,0.30,0.30,0.27,0.46,0.45,0.17,0.07,0.32,0.03,0.12],
[0.44,1.00,0.72,0.21,0.18,0.18,0.17,0.24,0.33,0.34,0.06,0.03,0.22,0.02,0.07],
[0.94,0.72,1.00,0.35,0.31,0.30,0.30,0.30,0.48,0.48,0.16,0.06,0.33,0.03,0.12],
[0.35,0.21,0.35,1.00,0.35,0.34,0.33,0.55,0.22,0.21,0.06,-0.03,0.45,0.04,0.14],
[0.31,0.18,0.31,0.35,1.00,0.98,0.96,0.63,0.82,0.76,0.07,0.00,0.93,0.08,0.35],
[0.30,0.18,0.30,0.34,0.98,1.00,0.98,0.62,0.81,0.74,0.07,0.00,0.92,0.08,0.36],
[0.30,0.17,0.30,0.33,0.96,0.98,1.00,0.61,0.79,0.73,0.07,0.00,0.90,0.08,0.37],
[0.27,0.24,0.30,0.55,0.63,0.62,0.61,1.00,0.41,0.40,0.09,-0.03,0.83,0.05,0.24],
[0.46,0.33,0.48,0.22,0.82,0.81,0.79,0.41,1.00,0.92,0.11,0.04,0.74,0.07,0.25],
[0.45,0.34,0.48,0.21,0.76,0.74,0.73,0.40,0.92,1.00,0.10,0.03,0.69,0.07,0.23],
[0.17,0.06,0.16,0.06,0.07,0.07,0.07,0.09,0.11,0.10,1.00,0.06,0.09,0.00,0.03],
[0.07,0.03,0.06,-0.03,0.00,0.00,0.00,-0.03,0.04,0.03,0.06,1.00,-0.01,0.00,0.00],
[0.32,0.22,0.33,0.45,0.93,0.92,0.90,0.83,0.74,0.69,0.09,-0.01,1.00,0.08,0.33],
[0.03,0.02,0.03,0.04,0.08,0.08,0.08,0.05,0.07,0.07,0.00,0.00,0.08,1.00,0.03],
[0.12,0.07,0.12,0.14,0.35,0.36,0.37,0.24,0.25,0.23,0.03,0.00,0.33,0.03,1.00]])
assert np.allclose(M, M.T)
cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f4f3f0", "#2a78d6"])
mask = np.triu(np.ones_like(M, bool), 1)
fig, ax = plt.subplots(figsize=(4.9, 4.2))
A = np.ma.array(M, mask=mask)
im = ax.imshow(A, cmap=cmap, vmin=-1, vmax=1)
for i in range(15):
    for j in range(i + 1):
        v = M[i, j]
        ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=5.6,
                color="white" if v > 0.7 else INK)
ax.set_xticks(range(15)); ax.set_yticks(range(15))
ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6.8); ax.set_yticklabels(names, fontsize=6.8)
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6.5); cb.set_label("Pearson r", fontsize=7.5); cb.outline.set_visible(False)
plt.savefig(f"{FIG}/fig_study1_corr.png"); print("ok")
