"""Optional Tapology scraper with polite fallback."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from loguru import logger

from ..data.schemas import Event
from ..utils.time_utils import as_utc


@dataclass
class TapologyScraper:
    enabled: bool = False

    def fetch_upcoming_events(self) -> List[Event]:
        if not self.enabled:
            logger.info("Tapology scraper disabled via configuration")
            return []
        logger.info("Fetching Tapology events (simulated)")
        rng = random.Random(42)
        events: List[Event] = []
        for idx in range(3):
            event = Event(
                event_id=f"tap-{idx}",
                name=f"Tapology Event {idx}",
                date=as_utc(f"2024-12-{10+idx}T00:00:00Z"),
                location=rng.choice(["Las Vegas", "Abu Dhabi", "New York"]),
            )
            events.append(event)
        return events


__all__ = ["TapologyScraper"]
