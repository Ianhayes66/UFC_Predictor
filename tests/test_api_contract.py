"""API contract tests for core endpoints."""

from __future__ import annotations

# ruff: noqa: S101
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from ufc_winprob.api.main import create_app
from ufc_winprob.api.schemas import PredictionResponse


def _prepare_market_fixture() -> Path:
    path = Path("data/processed/market_odds.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "bout_id": "evt-1-main",
                "sportsbook": "MockBook",
                "book": "MockBook",
                "american_odds": -120,
                "implied_probability": 0.55,
                "normalized_probability": 0.52,
                "overround": 0.04,
                "z_shin": 0.1,
                "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
                "stale": False,
            }
        ]
    )
    frame.to_parquet(path, index=False)
    return path


def test_markets_returns_all_fields() -> None:
    _prepare_market_fixture()
    app = create_app()
    client = TestClient(app)

    response = client.get("/markets/")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload
    entry = payload[0]
    expected_fields = {
        "bout_id",
        "book",
        "sportsbook",
        "price",
        "implied_probability",
        "normalized_probability",
        "overround",
        "last_updated",
        "z_shin",
        "stale",
    }
    assert expected_fields.issubset(entry.keys())
    assert isinstance(entry["bout_id"], str)
    assert isinstance(entry["book"], str)
    assert isinstance(entry["sportsbook"], str)
    assert isinstance(entry["price"], float)
    assert isinstance(entry["implied_probability"], float)
    assert isinstance(entry["normalized_probability"], float)
    assert isinstance(entry["overround"], float)
    assert isinstance(entry["last_updated"], str)
    assert isinstance(entry["z_shin"], float)
    assert isinstance(entry["stale"], bool)


def test_predict_validation_errors() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post("/predict", json={"fighter": "Name"})
    assert response.status_code == 422

    response = client.post(
        "/predict",
        json={"bout_id": "evt", "fighter": "Name", "opponent": 123},
    )
    assert response.status_code == 422


def test_predict_endpoint_returns_response_schema(trained_model: dict[str, Path]) -> None:
    app = create_app()
    client = TestClient(app)
    payload = {
        "bout_id": "evt-1-main",
        "fighter": "Fighter A",
        "opponent": "Fighter B",
        "division": "LW",
        "american_odds": -135,
        "fighter_age": 32,
        "opponent_age": 29,
        "fighter_height": 72,
        "opponent_height": 71,
        "fighter_reach": 74,
        "opponent_reach": 70,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    expected_keys = set(PredictionResponse.model_fields)
    assert expected_keys.issubset(data.keys())
    assert data["bout_id"] == payload["bout_id"]
    assert data["fighter"] == payload["fighter"]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["probability_low"] <= data["probability"] <= data["probability_high"]
    assert data["market_probability"] is not None


def test_metrics_endpoint_exposes_counters() -> None:
    app = create_app()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    for metric_name in (
        "ufc_api_requests_total",
        "ufc_api_request_latency_seconds",
        "ufc_api_exceptions_total",
    ):
        assert metric_name in body
