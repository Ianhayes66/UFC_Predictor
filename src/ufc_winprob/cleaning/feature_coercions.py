"""Feature coercion helpers."""

from __future__ import annotations

import pandas as pd


NUMERIC_COLUMNS = [
    "height_cm",
    "reach_cm",
    "age",
    "elo_delta",
    "activity_gap",
    "market_prob",
]


def coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    coerced = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in coerced.columns:
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


__all__ = ["NUMERIC_COLUMNS", "coerce_numeric"]
