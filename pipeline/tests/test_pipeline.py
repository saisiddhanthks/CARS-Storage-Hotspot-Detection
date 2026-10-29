import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import hotspot_pipeline as hp

@pytest.fixture(scope="module")
def data():
    df = hp.simulate_telemetry(n_rows=4000, seed=7)
    kept, _ = hp.drop_correlated(df, hp.FEATURES)
    df, _, thr = hp.run_detectors(df, kept)
    return df, kept, thr

def test_schema_complete():
    df = hp.simulate_telemetry(n_rows=2000)
    for c in ["timestamp", "cluster_id", "node_id", "controller_id"] + hp.FEATURES:
        assert c in df.columns
    assert df[hp.FEATURES].isna().sum().sum() == 0

def test_correlation_filter_keeps_queue_depth():
    df = hp.simulate_telemetry(n_rows=2000)
    kept, dropped = hp.drop_correlated(df, hp.FEATURES)
    assert "queue_depth" in kept
    corr = df[kept].corr().abs().values.copy()
    np.fill_diagonal(corr, 0)
    assert corr.max() <= 0.9

def test_time_split_has_no_leakage():
    df = hp.simulate_telemetry(n_rows=2000)
    tr, te = hp.time_split(df)
    assert tr["timestamp"].max() < te["timestamp"].min()

def test_autoencoder_threshold_flags_three_percent(data):
    df, _, _ = data
    assert abs(df["ae_anomaly"].mean() - 0.03) < 0.002

def test_consensus_requires_two_votes(data):
    df, _, _ = data
    assert (df.loc[df["consensus_anomaly"] == 1, "votes"] >= 2).all()
    assert (df.loc[df["consensus_anomaly"] == 0, "votes"] <= 1).all()

def test_hotspots_have_higher_queue_depth(data):
    df, _, _ = data
    g = df.groupby("consensus_anomaly")["queue_depth"].mean()
    assert g[1] > 2 * g[0]

def test_metric_map_resolution():
    assert hp.resolve_metrics("why is latency high") == ["avg_latency_ms", "p95_latency_ms", "p99_latency_ms"]
    assert hp.resolve_metrics("show me cache and errors") == ["cache_hit_pct", "error_count", "timeout_count"]
    assert hp.resolve_metrics("hello") == ["queue_depth", "avg_latency_ms"]

def test_api_endpoints():
    import api
    c = TestClient(api.app)
    assert c.get("/health").json()["rows"] == 20000
    hot = c.get("/hotspots", params={"limit": 3}).json()
    assert len(hot) == 3
    exp = c.get(f"/explain/{hot[0]['node_id']}").json()
    assert exp["evidence"] and exp["recommendation"]
    assert c.get("/explain/N999").status_code == 404
    assert "queue_depth" in c.post("/ask", json={"text": "queue"}).json()["metrics_used"]
    nn = c.get("/noisy-neighbours").json()
    assert nn and {"aggressor", "victims", "recommendation"} <= set(nn[0])


@pytest.fixture(scope="module")
def nn_data():
    df = hp.simulate_noisy_neighbour(n_rows=8000, seed=5, events_per_cluster=2)
    kept, _ = hp.drop_correlated(df, hp.FEATURES)
    df, _, _ = hp.run_detectors(df, kept)
    train, _ = hp.time_split(df)
    base = hp.workload_baseline(train[train["consensus_anomaly"] == 0])
    return hp.attribute_noisy_neighbours(df, base)

def test_noisy_neighbour_culprit_found(nn_data):
    hits = 0
    events = nn_data[nn_data["nn_event"] >= 0].groupby("nn_event")
    for _, g in events:
        true_agg = g.loc[g["nn_role"] == "aggressor", "node_id"].iloc[0]
        hits += true_agg in g.loc[g["nn_pred"] == "aggressor", "node_id"].unique()
    assert hits >= 0.75 * events.ngroups

def test_victims_point_to_aggressor_in_same_cluster(nn_data):
    v = nn_data[nn_data["nn_pred"] == "victim"]
    cl = dict(zip(nn_data["node_id"], nn_data["cluster_id"]))
    assert len(v) > 0
    assert all(cl[c] == k for c, k in zip(v["culprit"], v["cluster_id"]))
    assert (v["node_id"] != v["culprit"]).all()
