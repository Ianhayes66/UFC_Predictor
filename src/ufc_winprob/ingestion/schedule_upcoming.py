"""Upcoming event scheduler using UFCStats fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd

from ..data.schemas import Bout, Event, Fighter
from ..utils.time_utils import as_utc

UPCOMING_PATH = Path("data/raw/upcoming")
UPCOMING_PATH.mkdir(parents=True, exist_ok=True)


def load_upcoming_cards() -> pd.DataFrame:
    events: List[dict[str, object]] = []
    now = datetime.now(timezone.utc)
    for idx in range(3):
        events.append(
            {
                "event_id": f"upc-{idx}",
                "name": f"UFC Fight Night {idx}",
                "date": now + timedelta(days=7 * (idx + 1)),
                "weight_class": "LW",
                "fighter_a": f"fighter-{idx}",
                "fighter_b": f"fighter-{idx+1}",
            }
        )
    frame = pd.DataFrame(events)
    frame.to_json(UPCOMING_PATH / "upcoming_cards.json", orient="records", date_format="iso")
    return frame


__all__ = ["load_upcoming_cards"]
