import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from hotspot_pipeline import FEATURES, explain_row, resolve_metrics

app = FastAPI(title="Hotspot and Noisy Neighbour Detection API")
data = pd.read_csv("final_hotspot_output.csv", parse_dates=["timestamp"])
baseline = data[data["consensus_anomaly"] == 0][FEATURES].agg(["mean", "std"])
events = pd.read_csv("noisy_neighbour_events.csv")


class Question(BaseModel):
    text: str
    node_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "rows": len(data)}


@app.get("/hotspots")
def hotspots(limit: int = 10):
    hot = data[data["consensus_anomaly"] == 1].sort_values(["votes", "iso_score"], ascending=False)
    return hot.head(limit)[["timestamp", "cluster_id", "node_id", "votes", "severity",
                            "queue_depth", "avg_latency_ms"]].astype(str).to_dict("records")


@app.get("/explain/{node_id}")
def explain(node_id: str):
    rows = data[(data["node_id"] == node_id) & (data["consensus_anomaly"] == 1)]
    if rows.empty:
        raise HTTPException(404, f"No hotspot recorded for {node_id}")
    worst = rows.sort_values("iso_score", ascending=False).iloc[0]
    return explain_row(worst, baseline)


@app.post("/ask")
def ask(q: Question):
    cols = resolve_metrics(q.text)
    scope = data if q.node_id is None else data[data["node_id"] == q.node_id]
    summary = scope.groupby("consensus_anomaly")[cols].mean().round(2)
    return {"metrics_used": cols, "normal_vs_hotspot": summary.to_dict()}


@app.get("/noisy-neighbours")
def noisy_neighbours(cluster_id: str | None = None):
    ev = events if cluster_id is None else events[events["cluster_id"] == cluster_id]
    return ev.astype(str).to_dict("records")
