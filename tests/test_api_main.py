# ruff: noqa: S101
"""Tests for the FastAPI application entrypoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ufc_winprob.api.main import create_app


def get_test_client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_health_endpoint_returns_status_and_time() -> None:
    client = get_test_client()

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"status", "time"}
    assert payload["status"] == "ok"
    assert isinstance(payload["time"], str) and payload["time"]


def test_metrics_endpoint_exposes_api_metrics() -> None:
    client = get_test_client()

    # Trigger at least one request so counters are initialised.
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "ufc_api_requests_total" in body
    assert "ufc_api_request_latency_seconds" in body
    assert "ufc_api_exceptions_total" in body
