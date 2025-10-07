from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from ufc_winprob.api.main import create_app


def _prepare_market_fixture() -> None:
    path = Path("data/processed/market_odds.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "bout_id": "evt-1-main",
                "sportsbook": "MockBook",
                "american_odds": -120,
                "implied_probability": 0.55,
                "normalized_probability": 0.52,
                "shin_probability": 0.51,
                "z_shin": 0.1,
                "overround": 0.04,
                "timestamp": "2024-01-01T12:00:00+00:00",
                "stale": False,
                "implied_rank": 1,
                "normalized_rank": 1,
                "shin_rank": 1,
            }
        ]
    )
    frame.to_parquet(path, index=False)


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
    assert isinstance(entry["price"], float)
    assert isinstance(entry["implied_probability"], float)
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
