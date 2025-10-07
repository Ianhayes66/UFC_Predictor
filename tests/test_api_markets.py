"""Tests for the /markets endpoint schema stability."""

from __future__ import annotations

# ruff: noqa: S101
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ufc_winprob.api.main import create_app
from ufc_winprob.api.routers import markets as markets_router
from ufc_winprob.api.schemas import MarketResponse


def _assert_field_types(entry: dict[str, Any]) -> None:
    assert isinstance(entry["bout_id"], str)
    assert isinstance(entry["book"], str)
    assert isinstance(entry["sportsbook"], str)
    assert isinstance(entry["price"], float)
    assert isinstance(entry["implied_probability"], float)
    assert isinstance(entry["normalized_probability"], float)
    assert isinstance(entry["overround"], float)
    assert isinstance(entry["z_shin"], float)
    assert isinstance(entry["stale"], bool)

    last_updated = entry["last_updated"]
    assert isinstance(last_updated, str)
    parsed = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.astimezone(UTC).tzinfo == UTC


def test_markets_endpoint_returns_complete_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """/markets returns a non-empty list containing all required fields."""
    synthetic_path = tmp_path / "nonexistent.parquet"
    monkeypatch.setattr(markets_router, "_DATA_PATH", synthetic_path)

    app = create_app()
    client = TestClient(app)

    response = client.get("/markets/")
    assert response.status_code == 200

    payload = response.json()
    assert isinstance(payload, list)
    assert payload

    entry = payload[0]
    required_keys = set(MarketResponse.model_fields)
    assert required_keys.issubset(entry.keys())
    _assert_field_types(entry)
