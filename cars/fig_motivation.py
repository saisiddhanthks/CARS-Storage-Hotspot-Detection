import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FIG
from style import *
import matplotlib.pyplot as plt, numpy as np, pandas as pd, json
from common import load, events
df, tr = load("base"); te = ~tr
evs = [e for e in events("base") if e["start"] > df.tidx[te].min() + 30]
# isolated event (no other event on that node within +/- 2 h), median-ish severity
def isolated(e):
    return all(abs(o["start"] - e["start"]) > 24 or o["node"] != e["node"] for o in evs if o is not e)
cand = [e for e in evs if isolated(e) and 1.7 < e["lat_mult"] < 1.95 and 1.55 < e["iops_mult"] < 1.8]
e = cand[0]; print(e["start"], e["node"], e["aggressor"], e["lat_mult"], e["iops_mult"])
w0, w1 = e["start"] - 15, e["start"] + 15
sub = df[(df.tidx >= w0) & (df.tidx <= w1) & (df.node_id == e["node"])]
vic = e["victims"][:2]
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.25), sharex=False)
t = lambda s: (s.tidx - e["start"]) * 5
for ax, col, ttl in [(axs[0], "total_iops", "(a) IOPS, node_00"), (axs[1], "avg_latency_us", "(b) Mean latency, node_00")]:
    for i, (v, nm) in enumerate([(e["aggressor"], "aggressor " + e["aggressor"]), (vic[0], "victim " + vic[0]), (vic[1], "victim " + vic[1])]):
        s = sub[sub.volume_id == v]
        ax.plot(t(s), s[col] / (1000 if col == "total_iops" else 1000), color=C[i], lw=1.3, label=nm)
    ax.axvspan(-2.5, 12.5, color="#e34948", alpha=0.10, lw=0)
    ax.set_title(ttl, loc="left"); ax.set_xlabel("Minutes from event onset")
axs[0].set_ylabel("IOPS (thousands)"); axs[1].set_ylabel("Latency (ms)")
axs[0].set_ylim(top=axs[0].get_ylim()[1]*1.45); axs[0].legend(loc="upper left", fontsize=6.6, ncol=1, handlelength=1.0, borderaxespad=0.2, labelspacing=0.2)
# benign surge on a volume in the test period: isolated surge row not in an event
s_rows = df[te & (df.surge > 0) & (df.role == 0)]
best = None
for _, r in s_rows.sample(400, random_state=1).iterrows():
    ss = df[(df.volume_id == r.volume_id) & (df.tidx.between(r.tidx - 15, r.tidx + 15))]
    if ss.role.max() == 0 and ss.surge.sum() == 1:
        best = (r, ss); break
r, ss = best
ax = axs[2]; tt = (ss.tidx - r.tidx) * 5
ax.plot(tt, ss.total_iops / ss.total_iops.median(), color=C[0], lw=1.3, label="IOPS")
ax.plot(tt, ss.avg_latency_us / ss.avg_latency_us.median(), color=C[1], lw=1.3, label="Latency")
ax.axvspan(-2.5, 2.5, color=GRAY, alpha=0.25, lw=0)
ax.set_title(f"(c) Benign surge, {r.volume_id}", loc="left"); ax.set_xlabel("Minutes from surge")
ax.set_ylabel("Relative to window median"); ax.set_ylim(top=ax.get_ylim()[1]*1.18); ax.legend(loc="upper left", ncol=2, fontsize=6.6, handlelength=1.0, borderaxespad=0.2)
plt.tight_layout(w_pad=1.0, rect=(0, 0, 1, 1)); plt.savefig(f"{FIG}/fig_motivation.png")
json.dump(dict(event=e, surge_vol=r.volume_id, surge_t=int(r.tidx), surge_wl=r.workload_type), open("out/motivation_meta.json", "w"), default=str)
