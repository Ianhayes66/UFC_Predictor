"""Polite scraper for UFCStats with graceful fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from loguru import logger

from ..data.schemas import Bout, BoutStats, Event, Fighter
from ..utils.time_utils import as_utc

CACHE_DIR = Path("data/raw/ufcstats_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class UFCStatsScraper:
    """Simplified UFCStats scraper that reads from cached fixtures for tests."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir

    def load_fixture(self, name: str) -> dict:
        path = self.cache_dir / f"{name}.json"
        if not path.exists():
            logger.warning("Fixture %s missing, generating synthetic placeholder", name)
            data = {"fighters": [], "events": [], "bouts": []}
            path.write_text(json.dumps(data), encoding="utf-8")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def fetch_fighters(self) -> List[Fighter]:
        data = self.load_fixture("fighters")
        fighters = [Fighter(**item) for item in data.get("fighters", [])]
        return fighters

    def fetch_events(self) -> List[Event]:
        data = self.load_fixture("events")
        return [Event(event_id=item["event_id"], name=item["name"], date=as_utc(item["date"])) for item in data.get("events", [])]

    def fetch_bouts(self) -> List[Bout]:
        data = self.load_fixture("bouts")
        bouts: List[Bout] = []
        for item in data.get("bouts", []):
            stats = BoutStats(**item.get("stats", {}))
            bout = Bout(
                bout_id=item["bout_id"],
                event_id=item["event_id"],
                fighter_id=item["fighter_id"],
                opponent_id=item["opponent_id"],
                winner=item.get("winner"),
                weight_class=item.get("weight_class", "LW"),
                scheduled_rounds=item.get("scheduled_rounds", 3),
                method=item.get("method"),
                end_round=item.get("end_round"),
                end_time_seconds=item.get("end_time_seconds"),
                result=item.get("result"),
                stats=stats,
                event_date=as_utc(item.get("event_date")),
            )
            bouts.append(bout)
        return bouts


__all__ = ["UFCStatsScraper"]
