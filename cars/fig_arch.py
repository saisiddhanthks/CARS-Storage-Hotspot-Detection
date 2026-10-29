import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FIG
from style import *
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(7.0, 3.85)); ax.set_xlim(0, 100); ax.set_ylim(-5, 51); ax.axis("off")
BW, BH = 29, 13.2
def box(x, y, title, body, fill="white", edge="#8a8984", w=BW, h=BH):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.4", fc=fill, ec=edge, lw=0.9))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=8.6, fontweight="bold")
    ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=7.6, linespacing=1.25, color="#2b2a28")
    return (x, y, w, h)
def arrow(a, b, cs="arc3"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#52514e", connectionstyle=cs, shrinkA=1, shrinkB=1))
xs = [1.5, 35.5, 69.5]; ys = [36.5, 19.0, 1.5]
T = box(xs[0], ys[0], "Telemetry stream", "per volume, every 5 min:\nIOPS, latency p50/p99, queue\ndepth, request size and mix", fill="#f2f1ed")
X = box(xs[1], ys[0], "1  Context  x", "request shape, queue depth,\ntime of day, volume, tier\n(excludes interference outputs)")
M = box(xs[2], ys[0], "2  Expected behaviour", "robust L1 gradient-boosted trees\npredict log latency, log p99 and\nlog IOPS given x   (L = λW)")
Z = box(xs[2], ys[1], "3  Residual evidence", "per-volume median/MAD z-scores\n+ causal EWMA (α)\n→ $e_{lat}$,  $e_{iops}$")
C = box(xs[1], ys[1], "4  Peer coherence", "C = mean of the top-k latency\nresiduals among co-located\npeers (self excluded)")
S = box(xs[0], ys[1], "5  Victim and aggressor scores", "$V = e_{lat} + β\\cdot\\min(C,\\ e_{lat}^{+})$\n$A = e_{iops} + β\\cdot\\min(C,\\ e_{iops}^{+})$\n$S = \\max(V, A)$", fill="#e6effa", edge="#2a78d6")
O = box(xs[0], ys[2], "6  Alarm and attribution", "alarm if S > training 97th pct.\naggressor = argmax A on node\nvictims = other alarmed volumes", fill="#e6effa", edge="#2a78d6")
L = box(xs[1], ys[2], "Grounded explanation", "LLM narrates deterministic\ntool output only; every number\ntraceable to telemetry")
D = box(xs[2], ys[2], "Project policy layer", "persistence, rate limits,\nwhat-if simulation,\nrollback, circuit breaker")
r = lambda b: (b[0] + b[2] + 0.4, b[1] + b[3] / 2); l = lambda b: (b[0] - 0.4, b[1] + b[3] / 2)
top = lambda b: (b[0] + b[2] / 2, b[1] + b[3] + 0.4); bot = lambda b: (b[0] + b[2] / 2, b[1] - 0.4)
arrow(r(T), l(X)); arrow(r(X), l(M)); arrow(bot(M), top(Z)); arrow(l(Z), r(C)); arrow(l(C), r(S))
arrow((Z[0] + 4, Z[1] - 0.4), (S[0] + S[2] - 4, S[1] - 0.4), cs="arc3,rad=-0.18")
arrow(bot(S), top(O)); arrow(r(O), l(L)); arrow((O[0] + O[2] - 3, O[1] - 0.6), (D[0] + 3, D[1] - 0.6), cs="arc3,rad=0.09")
plt.savefig(f"{FIG}/fig_cars_arch.png"); print("ok")
