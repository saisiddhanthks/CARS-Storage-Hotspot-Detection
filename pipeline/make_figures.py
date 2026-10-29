import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
import hotspot_pipeline as hp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures") + os.sep
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
INK = "#1f2a44"
BLUE = "#dce6f5"
GREEN = "#e1f0e3"
ORANGE = "#fbe8d3"
GREY = "#eeeeee"


def box(ax, x, y, w, h, text, fc=BLUE, fs=9, bold=False, round_=True):
    style = "round,pad=0.02,rounding_size=0.08" if round_ else "square,pad=0.0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=INK, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, fs=8, off=(0, 0.12)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    if text:
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], text, ha="center", fontsize=fs,
                style="italic")


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT + name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture():
    fig, ax = canvas(10, 6.2, (0, 10), (0, 6.6))
    layers = [
        (5.4, "DATA LAYER", ORANGE),
        (4.0, "PROCESSING LAYER", GREY),
        (2.6, "DETECTION LAYER", BLUE),
        (1.2, "DECISION LAYER", GREEN),
    ]
    for y, label, fc in layers:
        ax.add_patch(Rectangle((0.1, y - 0.15), 9.8, 1.2, fc=fc, ec="none", alpha=0.45))
        ax.text(0.25, y + 0.85, label, fontsize=8, fontweight="bold", color=INK)
    box(ax, 0.6, 5.45, 2.6, 0.6, "Storage telemetry\n(nodes, controllers, volumes)", "white")
    box(ax, 3.7, 5.45, 2.6, 0.6, "Ingestion & schema check\n(18 metrics, 5-min grain)", "white")
    box(ax, 6.8, 5.45, 2.6, 0.6, "final_hotspot_output.csv\n(feature store)", "white")
    box(ax, 0.6, 4.05, 2.6, 0.6, "Cleaning & outlier\nreview", "white")
    box(ax, 3.7, 4.05, 2.6, 0.6, "Correlation filter (|r| > 0.9)\n+ standardisation", "white")
    box(ax, 6.8, 4.05, 2.6, 0.6, "Time-based split\n(train 80% / test 20%)", "white")
    for i, t in enumerate(["Isolation\nForest", "Local Outlier\nFactor", "DBSCAN\n(noise points)", "Autoencoder\n(recon. error)"]):
        box(ax, 0.6 + i * 2.25, 2.65, 1.9, 0.6, t, "white")
    box(ax, 0.45, 1.25, 2.1, 0.6, "Consensus voting\n(>= 2 of 4 detectors)", "white", bold=True, fs=8.5)
    box(ax, 2.85, 1.25, 2.1, 0.6, "Noisy-neighbour\nattribution (node x hour)", "white", bold=True, fs=8.5)
    box(ax, 5.25, 1.25, 2.1, 0.6, "Severity + explanation\n(metric_map, LLM)", "white", fs=8.5)
    box(ax, 7.65, 1.25, 2.1, 0.6, "FastAPI service &\nPower BI dashboard", "white", fs=8.5)
    box(ax, 3.7, 0.05, 2.6, 0.55, "Storage administrator", "#ffffff", bold=True)
    arrow(ax, 3.2, 5.75, 3.7, 5.75)
    arrow(ax, 6.3, 5.75, 6.8, 5.75)
    arrow(ax, 8.1, 5.45, 1.9, 4.65)
    arrow(ax, 3.2, 4.35, 3.7, 4.35)
    arrow(ax, 6.3, 4.35, 6.8, 4.35)
    for i in range(4):
        arrow(ax, 8.1, 4.05, 1.55 + i * 2.25, 3.25)
        arrow(ax, 1.55 + i * 2.25, 2.65, 1.5, 1.85)
    arrow(ax, 2.55, 1.55, 2.85, 1.55)
    arrow(ax, 4.95, 1.55, 5.25, 1.55)
    arrow(ax, 7.35, 1.55, 7.65, 1.55)
    arrow(ax, 8.7, 1.25, 6.3, 0.35)
    save(fig, "fig_architecture.png")


def dfd_level0():
    fig, ax = canvas(9, 3.6, (0, 9), (0, 3.6))
    box(ax, 0.2, 1.4, 1.9, 0.8, "Storage\nArray Telemetry", ORANGE, round_=False)
    ax.add_patch(Ellipse((4.5, 1.8), 2.9, 1.8, fc=BLUE, ec=INK, lw=1.4))
    ax.text(4.5, 1.8, "0\nHotspot & Noisy-\nNeighbour Detection\nSystem", ha="center", va="center", fontsize=9, fontweight="bold")
    box(ax, 6.9, 2.3, 1.9, 0.8, "Storage\nAdministrator", ORANGE, round_=False)
    box(ax, 6.9, 0.4, 1.9, 0.8, "Capacity / SRE\nTeam", ORANGE, round_=False)
    arrow(ax, 2.1, 1.8, 3.1, 1.8, "raw metrics", off=(0, 0.12))
    arrow(ax, 5.85, 2.2, 6.9, 2.7, "hotspot alerts,\nexplanations", off=(-0.1, 0.15))
    arrow(ax, 6.9, 2.45, 5.9, 1.95, "questions", off=(0.2, -0.32))
    arrow(ax, 5.85, 1.4, 6.9, 0.8, "dashboard,\nrecommendations", off=(-0.2, -0.45))
    save(fig, "fig_dfd0.png")


def dfd_level1():
    fig, ax = canvas(10, 5.4, (0, 10), (0, 5.4))
    def proc(x, y, t):
        ax.add_patch(Ellipse((x, y), 2.0, 1.0, fc=BLUE, ec=INK, lw=1.2))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.5)
    def store(x, y, t):
        ax.plot([x, x + 2.0], [y + 0.35, y + 0.35], color=INK)
        ax.plot([x, x + 2.0], [y - 0.05, y - 0.05], color=INK)
        ax.plot([x, x], [y - 0.05, y + 0.35], color=INK)
        ax.text(x + 1.0, y + 0.15, t, ha="center", va="center", fontsize=8)
    box(ax, 0.1, 4.2, 1.6, 0.7, "Telemetry\nsource", ORANGE, round_=False)
    proc(3.0, 4.55, "1.0\nIngest &\nvalidate")
    proc(6.0, 4.55, "2.0\nPre-process\n& split")
    store(8.0, 4.4, "D1 Clean features")
    proc(8.9, 2.9, "3.0\nRun four\ndetectors")
    store(5.8, 2.75, "D2 Detector flags")
    proc(3.6, 1.3, "4.0\nConsensus &\nseverity")
    proc(6.0, 1.3, "5.0\nNoisy-neighbour\nattribution")
    proc(8.6, 1.3, "6.0\nExplain &\nrecommend")
    box(ax, 0.1, 0.9, 1.7, 0.8, "Storage\nadministrator", ORANGE, round_=False)
    arrow(ax, 1.7, 4.55, 2.0, 4.55)
    arrow(ax, 4.0, 4.55, 5.0, 4.55, "valid rows")
    arrow(ax, 7.0, 4.55, 8.0, 4.55)
    arrow(ax, 9.0, 4.35, 8.95, 3.4, "scaled X")
    arrow(ax, 7.9, 2.9, 7.8, 2.95)
    arrow(ax, 6.5, 2.75, 4.0, 1.8, "4 flags/row", off=(-0.5, 0.0))
    arrow(ax, 4.6, 1.3, 5.0, 1.3)
    arrow(ax, 7.0, 1.3, 7.6, 1.3)
    ax.text(4.8, 1.45, "hotspots", fontsize=7, ha="center", style="italic")
    ax.text(7.3, 1.45, "culprits", fontsize=7, ha="center", style="italic")
    arrow(ax, 8.6, 0.8, 8.6, 0.45)
    ax.plot([8.6, 1.0], [0.45, 0.45], color=INK, lw=1.2)
    arrow(ax, 1.0, 0.45, 1.0, 0.9)
    ax.text(4.8, 0.25, "alert + culprit + evidence + action", fontsize=8, ha="center", style="italic")
    arrow(ax, 1.8, 1.5, 2.6, 1.5, "query", off=(0, 0.1))
    save(fig, "fig_dfd1.png")


def use_case():
    fig, ax = canvas(9.5, 6, (0, 9.5), (0, 6.2))
    def actor(x, y, name):
        ax.add_patch(plt.Circle((x, y + 0.55), 0.17, fc="white", ec=INK, lw=1.3))
        ax.plot([x, x], [y + 0.38, y - 0.1], color=INK, lw=1.3)
        ax.plot([x - 0.3, x + 0.3], [y + 0.2, y + 0.2], color=INK, lw=1.3)
        ax.plot([x, x - 0.25], [y - 0.1, y - 0.5], color=INK, lw=1.3)
        ax.plot([x, x + 0.25], [y - 0.1, y - 0.5], color=INK, lw=1.3)
        ax.text(x, y - 0.8, name, ha="center", fontsize=9, fontweight="bold")
    ax.add_patch(Rectangle((2.2, 0.2), 5.1, 5.8, fc="#fafbfd", ec=INK, lw=1.3))
    ax.text(4.75, 5.75, "Hotspot & Noisy-Neighbour Detection System", ha="center", fontweight="bold", fontsize=9.5)
    cases = ["Upload / stream telemetry", "Pre-process & train detectors", "Detect anomalies (4 models)",
             "View hotspots & severity", "Identify noisy neighbour (culprit)", "Ask question in plain English",
             "Get explanation & recommendation", "Export results / dashboard"]
    ys = np.linspace(5.1, 0.75, len(cases))
    for t, y in zip(cases, ys):
        ax.add_patch(Ellipse((4.75, y), 4.2, 0.5, fc=BLUE, ec=INK))
        ax.text(4.75, y, t, ha="center", va="center", fontsize=8.5)
    actor(1.0, 3.6, "Data engineer")
    actor(8.5, 3.6, "Storage admin")
    actor(8.5, 1.0, "LLM service\n(Groq API)")
    for i in [0, 1, 2]:
        ax.plot([1.3, 2.75], [3.8, ys[i]], color=INK, lw=0.8)
    for i in [3, 4, 5, 6, 7]:
        ax.plot([8.2, 6.75], [3.8, ys[i]], color=INK, lw=0.8)
    ax.plot([8.2, 6.75], [1.2, ys[6]], color=INK, lw=0.8, ls="--")
    save(fig, "fig_usecase.png")


def class_diagram():
    fig, ax = canvas(10, 6.8, (0, 10), (0.3, 7.2))
    def cls(x, y, w, name, attrs, methods, fs=9):
        lh = 0.22
        h = lh * (len(attrs) + len(methods)) + 0.55
        ax.add_patch(Rectangle((x, y - h), w, h, fc="white", ec=INK, lw=1.2))
        ax.add_patch(Rectangle((x, y - 0.38), w, 0.38, fc=BLUE, ec=INK, lw=1.2))
        ax.text(x + w / 2, y - 0.19, name, ha="center", va="center", fontweight="bold", fontsize=fs)
        yy = y - 0.55
        for t in attrs:
            ax.text(x + 0.08, yy, t, fontsize=7.5, family="DejaVu Sans Mono"); yy -= lh
        ax.plot([x, x + w], [yy + 0.12, yy + 0.12], color=INK, lw=0.8)
        yy -= 0.05
        for m in methods:
            ax.text(x + 0.08, yy, m, fontsize=7.5, family="DejaVu Sans Mono"); yy -= lh
        return y - h
    cls(0.2, 7.0, 3.0, "TelemetryLoader", ["- schema: list", "- path: str"], ["+ load(): DataFrame", "+ validate(df)", "+ simulate(n_rows)"])
    cls(3.6, 7.0, 3.0, "Preprocessor", ["- scaler: StandardScaler", "- threshold: 0.9", "- kept: list"], ["+ drop_correlated()", "+ time_split()", "+ transform(X)"])
    db = cls(7.0, 7.0, 2.8, "<<abstract>> Detector", ["- name: str", "- contamination"], ["+ fit(X_train)", "+ flag(X): int[]"])
    subs = ["IsolationForest", "LOF", "DBSCAN", "Autoencoder"]
    xs = [3.3, 4.95, 6.6, 8.25]
    for x, n in zip(xs, subs):
        cls(x, 4.65, 1.55, n, [], ["+ flag(X)"], fs=8)
    bus = 4.9
    ax.plot([xs[0] + 0.775, xs[-1] + 0.775], [bus, bus], color=INK, lw=1)
    for x in xs:
        ax.plot([x + 0.775, x + 0.775], [4.65, bus], color=INK, lw=1)
    ax.annotate("", xy=(8.4, db), xytext=(8.4, bus), arrowprops=dict(arrowstyle="-|>", fc="white", ec=INK, lw=1))
    cls(0.2, 4.9, 2.7, "ConsensusEngine", ["- detectors: Detector[4]", "- min_votes: 2"], ["+ vote(df): votes", "+ severity(votes)"])
    cls(0.2, 2.6, 3.0, "Explainer", ["- metric_map: dict", "- baseline: DataFrame"], ["+ explain_row(row)", "+ resolve_metrics(q)"])
    cls(3.6, 2.6, 3.0, "HotspotAPI (FastAPI)", ["- data: DataFrame", "- events: DataFrame"], ["+ GET /hotspots", "+ GET /explain/{node}", "+ POST /ask", "+ GET /noisy-neighbours"])
    cls(7.0, 3.6, 2.8, "NoisyNeighbourDetector", ["- baseline: node x hour", "- demand_z: 3.0", "- min_ratio: 1.5"], ["+ workload_baseline()", "+ attribute(df)", "+ summarise_events()"], fs=8.5)
    arrow(ax, 7.0, 3.0, 6.6, 2.2)
    arrow(ax, 7.6, 3.6, 2.9, 4.0)
    arrow(ax, 3.2, 6.6, 3.6, 6.6)
    arrow(ax, 1.7, 5.4, 1.7, 4.9)
    arrow(ax, 3.3, 4.2, 2.9, 4.2)
    arrow(ax, 1.7, 3.3, 1.7, 2.6)
    arrow(ax, 3.6, 2.0, 3.2, 2.0)
    save(fig, "fig_class.png")


def sequence():
    fig, ax = canvas(10, 6, (0, 10), (0, 6.4))
    actors = ["Admin", "FastAPI", "Consensus\nEngine", "Detectors", "Explainer", "LLM (Groq)"]
    xs = np.linspace(0.8, 9.2, len(actors))
    for x, a in zip(xs, actors):
        box(ax, x - 0.65, 5.7, 1.3, 0.55, a, BLUE, fs=8.5, bold=True)
        ax.plot([x, x], [0.2, 5.7], color=INK, ls="--", lw=0.8)
    msgs = [
        (0, 1, "GET /hotspots"), (1, 2, "get flagged rows"), (2, 3, "flags(X) x4"),
        (3, 2, "iso/lof/dbscan/ae flags"), (2, 1, "votes >= 2, severity"), (1, 0, "hotspot list"),
        (0, 1, "GET /explain/N012"), (1, 4, "explain_row(worst)"), (4, 5, "prompt with z-score evidence"),
        (5, 4, "plain-language summary"), (4, 1, "evidence + recommendation"), (1, 0, "JSON response"),
    ]
    y = 5.35
    for a, b, t in msgs:
        style = "-|>" if a < b else "-|>"
        ax.annotate("", xy=(xs[b], y), xytext=(xs[a], y),
                    arrowprops=dict(arrowstyle=style, color=INK, lw=1.1, ls="-" if a < b else "--"))
        ax.text((xs[a] + xs[b]) / 2, y + 0.07, t, ha="center", fontsize=7.8)
        y -= 0.42
    save(fig, "fig_sequence.png")


def gantt():
    tasks = [("Problem study & literature review", 0, 3), ("Telemetry schema & data preparation", 2, 4),
             ("EDA, correlation & feature selection", 4, 3), ("Isolation Forest / LOF / DBSCAN", 6, 4),
             ("Autoencoder & consensus voting", 9, 4), ("Explanation layer & FastAPI", 12, 5),
             ("Power BI dashboard", 15, 3), ("Noisy-neighbour attribution", 22, 8),
             ("Testing & controlled experiments", 24, 8), ("Research paper & report", 26, 10)]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (t, s, d) in enumerate(tasks):
        ax.barh(i, d, left=s, color="#4a6fa5", edgecolor=INK)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t for t, _, _ in tasks], fontsize=8.5)
    ax.invert_yaxis()
    months = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    ax.set_xticks(list(range(0, 36, 4)))
    ax.set_xticklabels(months)
    ax.set_xlim(0, 36)
    ax.set_xlabel("2026")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_gantt.png")


def results():
    df = hp.simulate_telemetry()
    kept, dropped = hp.drop_correlated(df, hp.FEATURES)
    plt.figure(figsize=(8, 6.5))
    corr = df[hp.FEATURES].corr()
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.7},
                xticklabels=True, yticklabels=True)
    plt.xticks(fontsize=7, rotation=60, ha="right"); plt.yticks(fontsize=7)
    save(plt.gcf(), "fig_corr.png")

    df, ae, thr = hp.run_detectors(df, kept)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(ae.loss_curve_, marker="o", ms=3, color="#4a6fa5")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training MSE")
    ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_ae_loss.png")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(np.log10(df["ae_error"]), bins=80, color="#9bb1d4", edgecolor="white")
    ax.axvline(np.log10(thr), color="#c0392b", ls="--", label=f"97th percentile = {thr:.4f}")
    ax.set_xlabel("log10(reconstruction error)"); ax.set_ylabel("Rows"); ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_ae_err.png")

    node = df.loc[df["votes"] == 4, "node_id"].iloc[0]
    s = df[df["node_id"] == node].iloc[:500]
    fig, ax = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    ax[0].plot(s["timestamp"], s["queue_depth"], lw=0.8, color="#4a6fa5")
    flagged = s[s["consensus_anomaly"] == 1]
    ax[0].scatter(flagged["timestamp"], flagged["queue_depth"], color="#c0392b", s=12, label="consensus hotspot", zorder=3)
    ax[0].set_ylabel("Queue depth"); ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(s["timestamp"], s["avg_latency_ms"], lw=0.8, color="#4a6fa5")
    ax[1].scatter(flagged["timestamp"], flagged["avg_latency_ms"], color="#c0392b", s=12, zorder=3)
    ax[1].set_ylabel("Avg latency (ms)")
    for a in ax: a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.25)
    fig.autofmt_xdate()
    save(fig, "fig_timeseries.png")

    methods = ["iso_anomaly", "lof_anomaly", "dbscan_anomaly", "ae_anomaly", "consensus_anomaly"]
    names = ["Isolation\nForest", "LOF", "DBSCAN", "Autoencoder", "Consensus\n(>=2 votes)"]
    rows = []
    for m in methods:
        rows.append([df[m].sum(), precision_score(df.true_hotspot, df[m]),
                     recall_score(df.true_hotspot, df[m]), f1_score(df.true_hotspot, df[m])])
    met = pd.DataFrame(rows, index=names, columns=["flagged", "precision", "recall", "f1"])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(names)); w = 0.26
    for i, (c, col) in enumerate(zip(["precision", "recall", "f1"], ["#4a6fa5", "#9bb1d4", "#e39b52"])):
        ax.bar(x + (i - 1) * w, met[c], w, label=c.capitalize(), color=col)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5); ax.set_ylim(0, 1.1)
    ax.legend(ncol=3, fontsize=8, loc="upper center"); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_metrics.png")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    data = [df.loc[df.consensus_anomaly == 0, "queue_depth"], df.loc[df.consensus_anomaly == 1, "queue_depth"]]
    ax.boxplot(data, tick_labels=["Normal", "Consensus hotspot"], showfliers=False, widths=0.5)
    ax.set_ylabel("Queue depth"); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_queue_box.png")

    fig, ax = plt.subplots(figsize=(6.5, 3))
    vc = df["votes"].value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color="#4a6fa5")
    ax.set_yscale("log"); ax.set_xlabel("Number of detectors agreeing"); ax.set_ylabel("Rows (log scale)")
    for i, v in zip(vc.index, vc.values): ax.text(str(i), v * 1.1, str(v), ha="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_votes.png")

    met.round(3).to_csv(OUT + "metrics.csv")
    stats = df.groupby("consensus_anomaly")[["queue_depth", "avg_latency_ms", "p99_latency_ms",
                                            "disk_utilization_pct", "controller_cpu_pct", "cache_hit_pct",
                                            "error_count", "timeout_count"]].mean().round(2)
    stats.to_csv(OUT + "group_stats.csv")
    vc.to_csv(OUT + "votes.csv")
    print(met.round(3)); print(stats.T); print(vc); print("dropped", dropped, "thr", thr, "node", node)
    print(df.true_hotspot.sum(), df.groupby("severity").size())


import sys
def nn_figs():
    import json
    att = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn_output.csv"), parse_dates=["timestamp"])
    ev = att[(att["nn_event"] >= 0)]
    eid = ev.loc[ev["cluster_id"] == "CL01", "nn_event"].min()
    g = ev[ev["nn_event"] == eid]
    t0, t1 = g["timestamp"].min() - pd.Timedelta("3h"), g["timestamp"].max() + pd.Timedelta("3h")
    agg = g.loc[g["nn_role"] == "aggressor", "node_id"].iloc[0]
    cl = g["cluster_id"].iloc[0]
    w = att[(att["cluster_id"] == cl) & att["timestamp"].between(t0, t1)]
    fig, ax = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)
    for node, d in w.groupby("node_id"):
        demand = d["read_iops"] + d["write_iops"]
        if node == agg:
            ax[0].plot(d["timestamp"], demand, color="#c0392b", lw=1.6, label=f"{node} (aggressor)")
            ax[1].plot(d["timestamp"], d["avg_latency_ms"], color="#c0392b", lw=1.6)
        else:
            ax[0].plot(d["timestamp"], demand, color="#9bb1d4", lw=0.9)
            ax[1].plot(d["timestamp"], d["avg_latency_ms"], color="#4a6fa5", lw=0.9)
    ax[0].plot([], [], color="#9bb1d4", label="other nodes in cluster")
    for a_ in ax:
        a_.axvspan(g["timestamp"].min(), g["timestamp"].max(), color="#f5d6d0", alpha=0.5)
        a_.spines[["top", "right"]].set_visible(False); a_.grid(alpha=0.25)
    ax[0].set_ylabel("Demand (IOPS)"); ax[0].legend(fontsize=8, loc="upper left")
    ax[1].set_ylabel("Avg latency (ms)")
    fig.autofmt_xdate()
    save(fig, "fig_nn_event.png")

    r = json.load(open(OUT + "nn_results.json"))["ablation"]
    names = ["Naive: busiest\nneighbour", "Node-hour\nmean / std", "Node-hour\nmedian / MAD"]
    keys = ["naive", "plain", "robust"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(3); w_ = 0.36
    ax.bar(x - w_ / 2, [r[k]["aggressor"]["f1"] for k in keys], w_, label="Aggressor F1", color="#4a6fa5")
    ax.bar(x + w_ / 2, [r[k]["victim"]["f1"] for k in keys], w_, label="Victim F1", color="#e39b52")
    for i, k in enumerate(keys):
        ax.text(i - w_ / 2, r[k]["aggressor"]["f1"] + 0.02, f'{r[k]["aggressor"]["f1"]:.2f}', ha="center", fontsize=8)
        ax.text(i + w_ / 2, r[k]["victim"]["f1"] + 0.02, f'{r[k]["victim"]["f1"]:.2f}', ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_nn_ablation.png")


sel = {"arch": architecture, "dfd1": dfd_level1, "use": use_case, "class": class_diagram, "nn": nn_figs, "seq": sequence, "gantt": gantt}
todo = [sel[a] for a in sys.argv[1:]] if len(sys.argv) > 1 else [architecture, dfd_level0, dfd_level1, use_case, class_diagram, sequence, gantt, results, nn_figs]
for f in todo:
    f()
