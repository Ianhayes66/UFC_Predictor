from __future__ import annotations

import pandas as pd
import pytest

from ufc_winprob.data_quality.great_expectations import (
    ValidationError,
    validate_interim_to_processed,
    validate_raw_to_interim,
)


def test_validate_raw_to_interim_accepts_valid_frame() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["evt-1", "evt-2"],
            "name": ["UFC Sample Night", "UFC Sample Night 2"],
            "date": ["2024-04-01T20:00:00+00:00", "2024-04-08T20:00:00+00:00"],
            "fighter_a": ["Fighter One", "Fighter Three"],
            "fighter_b": ["Fighter Two", "Fighter Four"],
        }
    )

    validate_raw_to_interim(frame)


def test_validate_raw_to_interim_raises_on_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["evt-1", "evt-1"],
            "name": ["UFC", "UFC"],
            "date": ["2024-04-01T20:00:00+00:00", "2024-04-02T20:00:00+00:00"],
            "fighter_a": ["A", "B"],
            "fighter_b": ["C", "D"],
        }
    )

    with pytest.raises(ValidationError):
        validate_raw_to_interim(frame)


def test_validate_raw_to_interim_raises_on_bad_dates() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["evt-1"],
            "name": ["UFC"],
            "date": ["not-a-date"],
            "fighter_a": ["A"],
            "fighter_b": ["B"],
        }
    )

    with pytest.raises(ValidationError):
        validate_raw_to_interim(frame)


def _valid_processed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bout_id": ["b-1", "b-2"],
            "sportsbook": ["MockBook", "SharpBook"],
            "american_odds": [-120, 110],
            "implied_probability": [0.545, 0.476],
            "normalized_probability": [0.520, 0.480],
            "shin_probability": [0.518, None],
            "stale": [False, False],
            "implied_rank": [2, 1],
            "normalized_rank": [2, 1],
            "shin_rank": [2, 1],
        }
    )


def test_validate_interim_to_processed_accepts_valid_frame() -> None:
    frame = _valid_processed_frame()

    validate_interim_to_processed(frame)


def test_validate_interim_to_processed_rejects_probability_out_of_bounds() -> None:
    frame = _valid_processed_frame()
    frame.loc[0, "normalized_probability"] = 1.5

    with pytest.raises(ValidationError):
        validate_interim_to_processed(frame)


def test_validate_interim_to_processed_rejects_non_boolean_stale() -> None:
    frame = _valid_processed_frame()
    frame.loc[0, "stale"] = "no"

    with pytest.raises(ValidationError):
        validate_interim_to_processed(frame)


def test_validate_interim_to_processed_rejects_inconsistent_ranks() -> None:
    frame = _valid_processed_frame()
    frame.loc[0, "shin_rank"] = 5

    with pytest.raises(ValidationError):
        validate_interim_to_processed(frame)
