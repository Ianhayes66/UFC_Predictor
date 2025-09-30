from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_predict():
    payload = {
        "reach_diff": 3.0,
        "age_diff": -2.0,
        "height_diff": 1.0,
        "strike_acc_diff": 4.0,
        "takedown_acc_diff": -1.0
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "win_probability" in body
    assert 0.0 <= body["win_probability"] <= 1.0
