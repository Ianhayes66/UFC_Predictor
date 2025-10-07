"""Odds API client with mockable backends."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List

from loguru import logger

from ..data.schemas import OddsSnapshot
from ..utils.odds_utils import american_to_implied, normalize_probabilities_shin, overround


@dataclass
class OddsAPIClient:
    sportsbooks: Iterable[str]
    use_live_api: bool = False

    def fetch_odds(self, bout_id: str, prices: Dict[str, float] | None = None) -> List[OddsSnapshot]:
        logger.info("Fetching odds for %s (live=%s)", bout_id, self.use_live_api)
        snapshots: List[OddsSnapshot] = []
        if prices is None:
            rng = random.Random(bout_id)
            prices = {book: rng.choice([-150, -110, 120, 150]) for book in self.sportsbooks}
        implied = [american_to_implied(value) for value in prices.values()]
        normalized = normalize_probabilities_shin(implied)
        for (book, price), norm_prob in zip(prices.items(), normalized):
            snapshot = OddsSnapshot(
                bout_id=bout_id,
                sportsbook=book,
                timestamp=datetime.now(timezone.utc),
                american_odds=float(price),
                implied_probability=american_to_implied(price),
                overround=overround(implied),
                normalized_probability=norm_prob,
            )
            snapshots.append(snapshot)
        return snapshots


__all__ = ["OddsAPIClient"]
