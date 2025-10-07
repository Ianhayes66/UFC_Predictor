"""Markets router exposing odds snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from ufc_winprob.api.schemas import MarketResponse

router = APIRouter(prefix="/markets", tags=["markets"])

_DATA_PATH = Path("data/processed/market_odds.parquet")


def _ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame contains the columns required by the API schema."""
    working = frame.copy()
    if "timestamp" not in working.columns:
        working["timestamp"] = datetime.now(UTC)
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)

    if "american_odds" not in working.columns:
        working["american_odds"] = 0.0
    if "implied_probability" not in working.columns:
        working["implied_probability"] = 0.5
    if "normalized_probability" not in working.columns:
        working["normalized_probability"] = working["implied_probability"].astype(float)
    if "overround" not in working.columns:
        working["overround"] = 0.0
    if "z_shin" not in working.columns:
        working["z_shin"] = 0.0
    if "stale" not in working.columns:
        working["stale"] = False
    if "book" not in working.columns:
        working["book"] = working.get("sportsbook", "Unknown")
    if "sportsbook" not in working.columns:
        working["sportsbook"] = working["book"]
    return working


@router.get("/", response_model=list[MarketResponse])
def markets() -> list[MarketResponse]:
    """Return the latest market snapshot for each recorded bout."""
    if not _DATA_PATH.exists():
        return []

    frame = pd.read_parquet(_DATA_PATH)
    if frame.empty:
        return []

    normalized = _ensure_columns(frame)
    responses: list[MarketResponse] = []
    for row in normalized.itertuples(index=False):
        timestamp = row.timestamp
        if isinstance(timestamp, str):
            ts = datetime.fromisoformat(timestamp)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        else:
            ts = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            else:
                ts = ts.astimezone(UTC)

        responses.append(
            MarketResponse(
                bout_id=str(getattr(row, "bout_id", "")),
                book=str(getattr(row, "book", getattr(row, "sportsbook", "Unknown"))),
                sportsbook=str(getattr(row, "sportsbook", getattr(row, "book", "Unknown"))),
                price=float(getattr(row, "american_odds", 0.0)),
                implied_probability=float(getattr(row, "implied_probability", 0.5)),
                normalized_probability=float(getattr(row, "normalized_probability", 0.5)),
                overround=float(getattr(row, "overround", 0.0)),
                last_updated=ts,
                z_shin=float(getattr(row, "z_shin", 0.0)),
                stale=bool(getattr(row, "stale", False)),
            )
        )
    return responses


__all__ = ["router"]
