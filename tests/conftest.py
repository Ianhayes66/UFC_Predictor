"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from ufc_winprob.features.feature_builder import synthetic_dataset
from ufc_winprob.models.training import train


@pytest.fixture(scope="session")
def synthetic_matrix():
    return synthetic_dataset(n=64)


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> Generator[dict[str, Path], None, None]:
    tmp_path = tmp_path_factory.mktemp("models")
    original_model_dir = Path("data/models")
    original_model_dir.mkdir(parents=True, exist_ok=True)
    artifacts = train()
    yield {
        "model": Path(artifacts.model_path),
        "calibrator": Path(artifacts.calibrator_path),
    }
    for path in [artifacts.model_path, artifacts.calibrator_path]:
        if path.exists():
            path.unlink()
