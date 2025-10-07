"""Markets router exposing odds snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter
import pandas as pd

from ..schemas import MarketResponse

router = APIRouter(prefix="/markets", tags=["markets"])


@router.get("/", response_model=List[MarketResponse])
def markets() -> List[MarketResponse]:
    path = Path("data/processed/market_odds.parquet")
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    responses: List[MarketResponse] = []
    for row in frame.itertuples(index=False):
        responses.append(
            MarketResponse(
                bout_id=str(row.bout_id),
                sportsbook=str(row.sportsbook),
                price=float(row.american_odds),
                implied_probability=float(row.implied_probability),
                normalized_probability=float(row.normalized_probability),
                overround=float(row.overround),
                last_updated=row.timestamp.to_pydatetime(),
                z_shin=float(row.z_shin) if "z_shin" in frame.columns else 0.0,
                stale=bool(row.stale) if "stale" in frame.columns else False,
            )
        )
    return responses


__all__ = ["router"]
