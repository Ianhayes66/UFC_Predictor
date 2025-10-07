"""Markets router exposing odds snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter

from ..schemas import MarketResponse

router = APIRouter(prefix="/markets", tags=["markets"])


@router.get("/", response_model=List[MarketResponse])
def markets() -> List[MarketResponse]:
    now = datetime.now(timezone.utc)
    return [
        MarketResponse(
            bout_id="synthetic-1",
            sportsbook="MockBook",
            price=-120.0,
            implied_probability=0.54,
            normalized_probability=0.52,
            overround=0.04,
            last_updated=now,
        )
    ]


__all__ = ["router"]
