import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FIG
from style import *
import matplotlib.pyplot as plt, numpy as np
# Normal vs hotspot means printed in the Colab notebook (Isolation Forest labels, 20,000-row test window)
d = {"read_iops": (2912.404762, 10688.725049), "write_iops": (1417.919875, 3307.235958),
     "throughput_mbps": (215.782038, 4446.994951), "p99_latency_ms": (17.683371, 195.341411),
     "queue_depth": (23.498690, 146.759123), "disk_utilization_pct": (17.526142, 80.437808),
     "controller_cpu_pct": (20.417039, 50.196266), "controller_memory_pct": (58.969324, 64.258734),
     "cache_hit_pct": (71.751771, 70.026558), "io_wait_pct": (6.292193, 52.195325),
     "error_count": (0.018314, 0.139610), "timeout_count": (0.001857, 0.222403)}
lab = {"read_iops": "Read IOPS", "write_iops": "Write IOPS", "throughput_mbps": "Throughput",
       "p99_latency_ms": "p99 latency", "queue_depth": "Queue depth", "disk_utilization_pct": "Disk utilisation",
       "controller_cpu_pct": "Controller CPU", "controller_memory_pct": "Controller memory",
       "cache_hit_pct": "Cache hit ratio", "io_wait_pct": "I/O wait", "error_count": "Error count",
       "timeout_count": "Timeout count"}
r = sorted(((v[1] / v[0], k) for k, v in d.items()))
fig, ax = plt.subplots(figsize=(3.45, 2.9))
y = np.arange(len(r))
vals = [a for a, _ in r]
ax.barh(y, vals, height=0.62, color=[C[0] if a >= 2 else GRAY for a in vals])
ax.set_xscale("log"); ax.axvline(1, color=INK2, lw=0.7)
ax.set_yticks(y); ax.set_yticklabels([lab[k] for _, k in r])
for yi, a in zip(y, vals):
    ax.text(a * 1.12, yi, f"{a:.1f}×" if a < 100 else f"{a:.0f}×", va="center", fontsize=7.2, color=INK2)
ax.set_xlim(0.5, 400); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1×", "10×", "100×"]); ax.set_xlabel("Hotspot mean ÷ normal mean (log scale)")
ax.grid(axis="y", visible=False)
plt.savefig(f"{FIG}/fig_study1_ratios.png"); print("ok")
