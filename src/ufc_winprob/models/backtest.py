"""Backtesting utilities for betting strategies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..utils.metrics import roi

BACKTEST_PATH = Path("data/processed/backtest_summary.json")
BACKTEST_REPORT_PATH = Path("reports/backtest_summary.json")


@dataclass
class BacktestResult:
    roi: float
    bets: int

    def save(self) -> Path:
        BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        BACKTEST_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = pd.Series({"roi": self.roi, "bets": self.bets})
        payload.to_json(BACKTEST_PATH, indent=2)
        payload.to_json(BACKTEST_REPORT_PATH, indent=2)
        return BACKTEST_PATH


def run_backtest(
    probabilities: Iterable[float], outcomes: Iterable[int], prices: Iterable[float]
) -> BacktestResult:
    prob = np.asarray(list(probabilities), dtype=float)
    true = np.asarray(list(outcomes), dtype=int)
    odds = np.asarray(list(prices), dtype=float)
    mask = prob > 0.5
    filtered_prob = prob[mask]
    filtered_true = true[mask]
    filtered_odds = odds[mask]
    if len(filtered_prob) == 0:
        return BacktestResult(roi=0.0, bets=0)
    value = roi(filtered_true, filtered_prob, filtered_odds)
    return BacktestResult(roi=value, bets=len(filtered_prob))


def backtest() -> BacktestResult:
    if not Path("data/processed/upcoming_predictions.parquet").exists():
        raise FileNotFoundError("Predictions missing. Run make predict first.")
    df = pd.read_parquet("data/processed/upcoming_predictions.parquet")
    result = run_backtest(
        df["probability"],
        np.random.binomial(1, df["probability"]),
        1 / df["probability"].clip(lower=0.05),
    )
    result.save()
    return result


if __name__ == "__main__":
    backtest()
