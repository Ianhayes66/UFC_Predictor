from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ufc_winprob.data_quality import (
    DataQualityError,
    validate_interim_to_processed,
    validate_raw_to_interim,
)
from ufc_winprob.pipelines import build_dataset, update_upcoming


def test_raw_to_interim_suite_passes() -> None:
    frame = pd.read_csv(Path("data/tests/upcoming_cards.csv"))
    validate_raw_to_interim(frame)


def test_raw_to_interim_suite_fails_missing_column() -> None:
    frame = pd.DataFrame({"event_id": ["evt-1"], "fighter_a": ["A"]})
    with pytest.raises(DataQualityError):
        validate_raw_to_interim(frame)


def test_interim_to_processed_suite_passes() -> None:
    frame = pd.read_csv(Path("data/tests/market_snapshots.csv"))
    validate_interim_to_processed(frame)


def test_interim_to_processed_suite_enforces_probability_bounds() -> None:
    frame = pd.read_csv(Path("data/tests/market_snapshots.csv"))
    frame.loc[0, "implied_probability"] = 1.2
    with pytest.raises(DataQualityError):
        validate_interim_to_processed(frame)


def test_build_dataset_fails_when_raw_validation_fails(monkeypatch) -> None:
    def _boom(_: pd.DataFrame) -> None:
        raise DataQualityError("invalid raw data")

    monkeypatch.setattr(build_dataset, "validate_raw_to_interim", _boom)
    with pytest.raises(DataQualityError):
        build_dataset.build()


def test_update_upcoming_stops_on_processed_validation(monkeypatch) -> None:
    def _fail(_: pd.DataFrame) -> None:
        raise DataQualityError("processed check failed")

    monkeypatch.setattr(update_upcoming, "validate_interim_to_processed", _fail)
    with pytest.raises(DataQualityError):
        update_upcoming.run()
