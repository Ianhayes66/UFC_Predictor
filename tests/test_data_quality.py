from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ufc_winprob.data_quality import (
    DataQualityError,
    validate_interim_to_processed,
    validate_raw_to_interim,
)


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
