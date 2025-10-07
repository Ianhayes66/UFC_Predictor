"""Polite scraper for UFCStats with caching and rate limiting."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
import random
from pathlib import Path
from typing import List
from urllib.robotparser import RobotFileParser

import httpx
from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..data.schemas import Bout, BoutStats, Event, Fighter
from ..settings import get_settings
from ..utils.time_utils import as_utc

RAW_CACHE = Path("data/raw/ufcstats")
RAW_CACHE.mkdir(parents=True, exist_ok=True)
INTERIM_CACHE = Path("data/interim/ufcstats")
INTERIM_CACHE.mkdir(parents=True, exist_ok=True)


class ScrapingDisabled(RuntimeError):
    """Raised when scraping is disabled via configuration or robots.txt."""


@dataclass
class RateLimiter:
    interval_seconds: float
    jitter_seconds: float = 0.0
    last_call: float = 0.0

    def wait(self) -> None:
        if self.interval_seconds <= 0:
            return
        now = time.monotonic()
        delta = now - self.last_call
        remaining = self.interval_seconds - delta
        if remaining > 0:
            jitter = random.uniform(0, self.jitter_seconds)
            time.sleep(remaining + jitter)
        self.last_call = time.monotonic()


class UFCStatsScraper:
    """HTTPX powered scraper with caching and robots.txt awareness."""

    def __init__(
        self,
        base_url: str = "https://www.ufcstats.com",
        session: httpx.Client | None = None,
        cache_dir: Path = INTERIM_CACHE,
    ) -> None:
        settings = get_settings()
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        timeout = httpx.Timeout(10.0, read=20.0)
        headers = {"User-Agent": settings.providers.user_agent}
        self.client = session or httpx.Client(timeout=timeout, headers=headers)
        self.rate_limiter = RateLimiter(
            settings.providers.rate_limit_seconds,
            jitter_seconds=settings.providers.rate_limit_jitter_seconds,
        )
        self.disable_if_disallowed = settings.providers.disable_if_robots_disallow
        self._robots: RobotFileParser | None = None

    def close(self) -> None:
        self.client.close()

    def _ensure_robots(self) -> RobotFileParser:
        if self._robots is not None:
            return self._robots
        robots_url = f"{self.base_url}/robots.txt"
        try:
            response = self.client.get(robots_url)
            response.raise_for_status()
            parser = RobotFileParser()
            parser.parse(response.text.splitlines())
            self._robots = parser
        except httpx.HTTPError as exc:  # pragma: no cover - network failure fallback
            logger.warning("Failed to fetch robots.txt: %s", exc)
            parser = RobotFileParser()
            parser.parse([])
            self._robots = parser
        return self._robots

    def _check_allowed(self, path: str) -> None:
        parser = self._ensure_robots()
        allowed = parser.can_fetch("*", path)
        if not allowed and self.disable_if_disallowed:
            logger.warning("Robots.txt disallows %s; skipping fetch", path)
            raise ScrapingDisabled(path)

    def _raw_cache_path(self, slug: str) -> Path:
        return RAW_CACHE / f"{slug}.html"

    def _interim_path(self, slug: str) -> Path:
        return self.cache_dir / f"{slug}.json"

    def _write_raw(self, slug: str, content: str) -> None:
        path = self._raw_cache_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _write_json(self, slug: str, payload: dict) -> None:
        path = self._interim_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    def _get(self, path: str) -> str:
        self._check_allowed(path)
        self.rate_limiter.wait()
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        response = self.client.get(url)
        response.raise_for_status()
        return response.text

    def _load_cached(self, slug: str) -> dict | None:
        interim = self._interim_path(slug)
        if interim.exists():
            return json.loads(interim.read_text(encoding="utf-8"))
        return None

    def _persist_snapshot(self, slug: str, data: dict) -> dict:
        self._write_json(slug, data)
        return data

    def fetch_fighters(self) -> List[Fighter]:
        slug = "fighters"
        cached = self._load_cached(slug)
        if cached:
            return [Fighter(**item) for item in cached.get("fighters", [])]
        try:
            html = self._get("/statistics/fighters")
            self._write_raw(slug, html)
            # Placeholder parsing: rely on cached fixtures when available
            data = {"fighters": []}
            return []
        except (RetryError, ScrapingDisabled):
            logger.info("Falling back to cached fighters for %s", slug)
            fixture_path = RAW_CACHE / "ufcstats_cache" / f"{slug}.json"
            if fixture_path.exists():
                payload = json.loads(fixture_path.read_text(encoding="utf-8"))
                self._persist_snapshot(slug, payload)
                return [Fighter(**item) for item in payload.get("fighters", [])]
        return []

    def fetch_events(self) -> List[Event]:
        slug = "events"
        cached = self._load_cached(slug)
        if cached:
            return [
                Event(event_id=item["event_id"], name=item["name"], date=as_utc(item["date"]))
                for item in cached.get("events", [])
            ]
        try:
            html = self._get("/statistics/events")
            self._write_raw(slug, html)
        except (RetryError, ScrapingDisabled):
            logger.info("Using cached events for %s", slug)
            fixture_path = RAW_CACHE / "ufcstats_cache" / f"{slug}.json"
            if fixture_path.exists():
                payload = json.loads(fixture_path.read_text(encoding="utf-8"))
                self._persist_snapshot(slug, payload)
                return [
                    Event(event_id=item["event_id"], name=item["name"], date=as_utc(item["date"]))
                    for item in payload.get("events", [])
                ]
        return []

    def fetch_bouts(self) -> List[Bout]:
        slug = "bouts"
        cached = self._load_cached(slug)
        if cached:
            return self._parse_bouts(cached)
        try:
            html = self._get("/statistics/bouts")
            self._write_raw(slug, html)
        except (RetryError, ScrapingDisabled):
            logger.info("Using cached bouts for %s", slug)
            fixture_path = RAW_CACHE / "ufcstats_cache" / f"{slug}.json"
            if fixture_path.exists():
                payload = json.loads(fixture_path.read_text(encoding="utf-8"))
                self._persist_snapshot(slug, payload)
                return self._parse_bouts(payload)
        return []

    def _parse_bouts(self, payload: dict) -> List[Bout]:
        bouts: List[Bout] = []
        for item in payload.get("bouts", []):
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


__all__ = ["UFCStatsScraper", "ScrapingDisabled"]
