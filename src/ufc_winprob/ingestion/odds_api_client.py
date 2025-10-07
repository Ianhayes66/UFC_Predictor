"""Odds API client with mockable backends and enhanced metrics."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import httpx
import numpy as np
from loguru import logger
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..data.schemas import OddsSnapshot
from ..settings import get_settings
from ..utils.odds_utils import american_to_implied, overround, shin_adjustment

ODDS_CACHE = Path("data/interim/odds")
ODDS_CACHE.mkdir(parents=True, exist_ok=True)


@dataclass
class OddsAPIClient:
    sportsbooks: Iterable[str]
    use_live_api: bool | None = None
    stale_after_minutes: int = 30
    client: httpx.Client | None = None
    _aggregated: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        settings = get_settings()
        self.sportsbooks = list(self.sportsbooks)
        self.api_key = settings.odds_api_key or ""
        env_live = settings.use_live_odds and bool(self.api_key)
        self.use_live_api = env_live if self.use_live_api is None else self.use_live_api
        timeout = httpx.Timeout(10.0, read=20.0)
        self.client = self.client or httpx.Client(timeout=timeout)
        self.base_url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
        self.market = settings.providers.odds_market
        self._stale_delta = timedelta(minutes=self.stale_after_minutes)

    def close(self) -> None:
        if self.client:
            self.client.close()

    @property
    def aggregated(self) -> Dict[str, float]:
        return self._aggregated

    def fetch_odds(self, bout_id: str, prices: Dict[str, float] | None = None) -> List[OddsSnapshot]:
        logger.info("Fetching odds for %s (live=%s)", bout_id, self.use_live_api)
        if self.use_live_api:
            try:
                response = self._fetch_live_odds(bout_id)
                snapshots = self._parse_live_response(bout_id, response)
                self._persist(bout_id, snapshots)
                return snapshots
            except (RetryError, httpx.HTTPError) as exc:  # pragma: no cover - network failures
                logger.warning("Live odds fetch failed (%s); falling back to mock", exc)
        snapshots = self._mock_odds(bout_id, prices)
        self._persist(bout_id, snapshots)
        return snapshots

    def _persist(self, bout_id: str, snapshots: List[OddsSnapshot]) -> None:
        path = ODDS_CACHE / f"{bout_id}.json"
        payload = {
            "bout_id": bout_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "snapshots": [snap.model_dump() for snap in snapshots],
            "median_probability": float(np.median([snap.normalized_probability for snap in snapshots])) if snapshots else 0.0,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if snapshots:
            self._aggregated[bout_id] = payload["median_probability"]

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    def _fetch_live_odds(self, bout_id: str) -> dict:
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": self.market,
            "event_id": bout_id,
        }
        response = self.client.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

    def _parse_live_response(self, bout_id: str, payload: dict) -> List[OddsSnapshot]:
        now = datetime.now(timezone.utc)
        snapshots: List[OddsSnapshot] = []
        bookmakers = payload if isinstance(payload, list) else payload.get("bookmakers", [])
        for book in bookmakers:
            key = book.get("key", "unknown")
            markets = book.get("markets", [])
            if not markets:
                continue
            prices = self._extract_prices(markets[0])
            snapshot = self._build_snapshot(bout_id, key, prices, now)
            snapshots.append(snapshot)
        return snapshots

    def _extract_prices(self, market: dict) -> Tuple[float, float]:
        outcomes = market.get("outcomes", [])
        if len(outcomes) < 2:
            raise ValueError("Expected two outcomes for odds market")
        price_a = float(outcomes[0].get("price", 0))
        price_b = float(outcomes[1].get("price", 0))
        if price_a == 0 or price_b == 0:
            raise ValueError("Invalid odds returned")
        return price_a, price_b

    def _mock_odds(self, bout_id: str, prices: Dict[str, float] | None = None) -> List[OddsSnapshot]:
        rng = random.Random(bout_id)
        snapshots: List[OddsSnapshot] = []
        now = datetime.now(timezone.utc)
        for book in self.sportsbooks:
            price = prices[book] if prices and book in prices else rng.choice([-150, -110, 120, 150, 175])
            opponent_price = -price if price > 0 else abs(price) + 10
            snapshot = self._build_snapshot(bout_id, book, (price, opponent_price), now)
            snapshots.append(snapshot)
        return snapshots

    def _build_snapshot(
        self, bout_id: str, sportsbook: str, prices: Tuple[float, float], timestamp: datetime
    ) -> OddsSnapshot:
        implied = [american_to_implied(price) for price in prices]
        adj, z_value = shin_adjustment(implied)
        snapshot = OddsSnapshot(
            bout_id=bout_id,
            sportsbook=sportsbook,
            timestamp=timestamp,
            american_odds=float(prices[0]),
            implied_probability=float(implied[0]),
            overround=overround(implied),
            normalized_probability=float(adj[0]),
            shin_probability=float(adj[0]),
            z_shin=z_value,
            stale=(datetime.now(timezone.utc) - timestamp) > self._stale_delta,
        )
        return snapshot


__all__ = ["OddsAPIClient"]
