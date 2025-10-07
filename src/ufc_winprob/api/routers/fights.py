"""Fights router providing upcoming bouts."""

from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import APIRouter

router = APIRouter(prefix="/upcoming", tags=["upcoming"])


@router.get("/")
def upcoming() -> List[dict[str, object]]:
    now = datetime.utcnow().isoformat()
    return [
        {
            "bout_id": "synthetic-1",
            "event": "UFC Fight Night",
            "fighters": ["A. Example", "B. Example"],
            "weight_class": "LW",
            "event_time": now,
        }
    ]


__all__ = ["router"]
