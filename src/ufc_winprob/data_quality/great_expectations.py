"""Great Expectations-inspired validation helpers.

This module provides light-weight, dependency-free checks that mimic the
behaviour of common Great Expectations suites used within the project.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-untyped]


class ValidationError(ValueError):
    """Raised when a dataframe does not satisfy a validation gate."""


# Backwards compatibility for previous public API.
DataQualityError = ValidationError


@dataclass(frozen=True)
class _CheckConfig:
    required_columns: tuple[str, ...]
    non_null_columns: tuple[str, ...]


def _require_columns(df: pd.DataFrame, columns: Iterable[str], stage: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValidationError(f"{stage}: missing required column(s): {', '.join(missing)}")


def _ensure_non_null(df: pd.DataFrame, columns: Iterable[str], stage: str) -> None:
    for column in columns:
        if df[column].isnull().any():
            raise ValidationError(f"{stage}: column '{column}' contains null values")


def _ensure_unique(df: pd.DataFrame, column: str, stage: str) -> None:
    if df[column].duplicated().any():
        raise ValidationError(f"{stage}: column '{column}' must be unique")


def _ensure_datetime(
    series: pd.Series, *, stage: str, lower: str = "1993-01-01", upper: str = "2100-01-01"
) -> None:
    converted = pd.to_datetime(series, errors="coerce", utc=True)
    if converted.isna().any():
        raise ValidationError(f"{stage}: invalid datetime values detected")
    lower_bound = pd.Timestamp(lower, tz="UTC")
    upper_bound = pd.Timestamp(upper, tz="UTC")
    if (converted < lower_bound).any() or (converted > upper_bound).any():
        message = (
            f"{stage}: datetime values must be between {lower_bound.date()} "
            f"and {upper_bound.date()}"
        )
        raise ValidationError(message)


def _ensure_probability_range(
    series: pd.Series, *, stage: str, column: str, allow_null: bool = False
) -> None:
    values = pd.to_numeric(series, errors="coerce")
    if not allow_null and values.isna().any():
        raise ValidationError(f"{stage}: column '{column}' contains non-numeric values")
    bounded = values[(values.notna()) & ((values < 0.0) | (values > 1.0))]
    if not bounded.empty:
        raise ValidationError(
            f"{stage}: column '{column}' contains values outside the [0, 1] interval"
        )


def _ensure_numeric_range(
    series: pd.Series,
    *,
    stage: str,
    column: str,
    minimum: float,
    maximum: float,
) -> None:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValidationError(f"{stage}: column '{column}' contains non-numeric values")
    if ((values < minimum) | (values > maximum)).any():
        raise ValidationError(f"{stage}: column '{column}' must be between {minimum} and {maximum}")


def _ensure_rank_integrity(df: pd.DataFrame, stage: str) -> None:
    rank_columns = ("implied_rank", "normalized_rank", "shin_rank")
    missing_columns = [column for column in rank_columns if column not in df.columns]
    if missing_columns:
        return
    for column in rank_columns:
        ranks = pd.to_numeric(df[column], errors="coerce")
        if ranks.isna().any():
            raise ValidationError(f"{stage}: column '{column}' contains non-numeric values")
        if (ranks < 1).any():
            raise ValidationError(f"{stage}: column '{column}' must be >= 1")
    implied = pd.to_numeric(df["implied_rank"], errors="coerce")
    normalized = pd.to_numeric(df["normalized_rank"], errors="coerce")
    shin = pd.to_numeric(df["shin_rank"], errors="coerce")
    if ((implied - normalized).abs() > 1).any() or ((implied - shin).abs() > 1).any():
        raise ValidationError(f"{stage}: ranking columns disagree by more than one position")


def _ensure_boolean(series: pd.Series, *, stage: str, column: str) -> None:
    if not series.dropna().map(lambda value: isinstance(value, bool | np.bool_)).all():
        raise ValidationError(f"{stage}: column '{column}' must contain boolean values")


def validate_raw_to_interim(df: pd.DataFrame) -> None:
    """Validate raw ingestion output before entering the interim layer."""
    stage = "raw_to_interim"
    required = _CheckConfig(
        required_columns=("event_id", "name", "date", "fighter_a", "fighter_b"),
        non_null_columns=("event_id", "name", "date", "fighter_a", "fighter_b"),
    )
    _require_columns(df, required.required_columns, stage)
    if df.empty:
        raise ValidationError(f"{stage}: dataframe must contain at least one row")
    _ensure_non_null(df, required.non_null_columns, stage)
    _ensure_unique(df, "event_id", stage)
    _ensure_datetime(df["date"], stage=stage)


def validate_interim_to_processed(df: pd.DataFrame) -> None:
    """Validate data leaving the interim layer for the processed store."""
    stage = "interim_to_processed"
    required = _CheckConfig(
        required_columns=(
            "bout_id",
            "sportsbook",
            "american_odds",
            "implied_probability",
            "normalized_probability",
            "shin_probability",
            "stale",
            "implied_rank",
            "normalized_rank",
            "shin_rank",
        ),
        non_null_columns=(
            "bout_id",
            "sportsbook",
            "american_odds",
            "implied_probability",
            "normalized_probability",
            "stale",
            "implied_rank",
            "normalized_rank",
        ),
    )
    _require_columns(df, required.required_columns, stage)
    if df.empty:
        raise ValidationError(f"{stage}: dataframe must contain at least one row")
    _ensure_non_null(df, required.non_null_columns, stage)
    _ensure_probability_range(df["implied_probability"], stage=stage, column="implied_probability")
    _ensure_probability_range(
        df["normalized_probability"], stage=stage, column="normalized_probability"
    )
    _ensure_probability_range(
        df["shin_probability"], stage=stage, column="shin_probability", allow_null=True
    )
    _ensure_numeric_range(
        df["american_odds"],
        stage=stage,
        column="american_odds",
        minimum=-2000,
        maximum=2000,
    )
    _ensure_boolean(df["stale"], stage=stage, column="stale")
    _ensure_rank_integrity(df, stage)


__all__ = [
    "DataQualityError",
    "ValidationError",
    "validate_interim_to_processed",
    "validate_raw_to_interim",
]
