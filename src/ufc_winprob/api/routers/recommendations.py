"""Recommendations router providing EV ranked bets."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import APIRouter

from ..schemas import RecommendationResponse

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/", response_model=List[RecommendationResponse])
def recommendations() -> List[RecommendationResponse]:
    path = Path("data/processed/ev_leaderboard.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        RecommendationResponse(
            bout_id=row.get("bout_id", "synthetic"),
            fighter=row.get("fighter", "unknown"),
            sportsbook="MockBook",
            price=float(row.get("decimal_odds", 2.0)),
            probability=float(row["probability"]),
            expected_value=float(row["expected_value"]),
            kelly=float(row["kelly"]),
            stale=False,
        )
        for _, row in df.iterrows()
    ]


__all__ = ["router"]
