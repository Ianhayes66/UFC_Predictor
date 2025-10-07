from __future__ import annotations

from pathlib import Path

from ufc_winprob.models.training import train


def test_training_produces_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TZ", "UTC")
    artifacts = train()
    assert artifacts.model_path.exists()
    assert artifacts.calibrator_path.exists()
    assert artifacts.metrics_path.exists()
