"""Tests for the /predict API endpoint."""

from __future__ import annotations

# ruff: noqa: S101
from collections.abc import Mapping

from fastapi.testclient import TestClient

from ufc_winprob.api.main import create_app
from ufc_winprob.api.schemas import PredictionResponse


def test_predict_returns_prediction(trained_model: Mapping[str, object]) -> None:
    assert "model" in trained_model
    app = create_app()
    client = TestClient(app)
    payload = {
        "bout_id": "evt-1-main",
        "fighter": "Fighter A",
        "opponent": "Fighter B",
        "division": "LW",
        "american_odds": -150,
        "fighter_age": 32,
        "opponent_age": 30,
        "fighter_height": 72,
        "opponent_height": 70,
        "fighter_reach": 74,
        "opponent_reach": 72,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    validated = PredictionResponse.model_validate(body)
    assert validated.bout_id == payload["bout_id"]
    assert validated.fighter == payload["fighter"]
    assert 0.0 <= validated.probability <= 1.0
    assert validated.probability_low <= validated.probability <= validated.probability_high
    assert validated.market_probability is not None


def test_predict_rejects_negative_values(trained_model: Mapping[str, object]) -> None:
    assert "model" in trained_model
    app = create_app()
    client = TestClient(app)
    payload = {
        "bout_id": "evt-1-main",
        "fighter": "Fighter A",
        "opponent": "Fighter B",
        "fighter_age": -5,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
