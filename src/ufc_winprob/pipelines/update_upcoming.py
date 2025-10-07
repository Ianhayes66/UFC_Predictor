"""Pipeline for updating upcoming fights and odds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..ingestion.schedule_upcoming import load_upcoming_cards
from ..models.predict import predict
from ..models.selection import rank_recommendations


def run() -> dict[str, Path]:
    cards = load_upcoming_cards()
    cards.to_parquet("data/processed/upcoming_cards.parquet", index=False)
    predictions = predict()
    leaderboard = rank_recommendations(predictions.assign(american_odds=120))
    leaderboard_path = Path("data/processed/ev_leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    return {
        "cards": Path("data/processed/upcoming_cards.parquet"),
        "predictions": Path("data/processed/upcoming_predictions.parquet"),
        "leaderboard": leaderboard_path,
    }


if __name__ == "__main__":
    run()
