"""Bet selection logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..data.schemas import Recommendation
from ..utils.odds_utils import american_to_decimal


@dataclass
class SelectionConfig:
    min_ev: float = 0.02
    max_kelly: float = 0.1


def kelly_fraction(prob: float, odds: float) -> float:
    edge = prob * (odds - 1) - (1 - prob)
    if odds <= 1:
        return 0.0
    fraction = edge / (odds - 1)
    return float(np.clip(fraction, 0.0, 1.0))


def rank_recommendations(predictions: pd.DataFrame, config: SelectionConfig | None = None) -> pd.DataFrame:
    config = config or SelectionConfig()
    records: list[dict[str, object]] = []
    for _, row in predictions.iterrows():
        decimal_odds = american_to_decimal(row.get("american_odds", 100.0))
        prob = row["probability"]
        ev = prob * (decimal_odds - 1) - (1 - prob)
        kelly = min(kelly_fraction(prob, decimal_odds), config.max_kelly)
        if ev < config.min_ev:
            continue
        records.append(
            {
                "bout_id": row.get("bout_id", "synthetic"),
                "fighter": row.get("fighter", "unknown"),
                "decimal_odds": decimal_odds,
                "probability": prob,
                "expected_value": ev,
                "kelly": kelly,
            }
        )
    return pd.DataFrame(records).sort_values("expected_value", ascending=False)


__all__ = ["SelectionConfig", "rank_recommendations", "kelly_fraction"]
