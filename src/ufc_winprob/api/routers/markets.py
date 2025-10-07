"""Markets router exposing odds snapshots with a stable schema."""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from ufc_winprob.api.schemas import MarketResponse

router = APIRouter(prefix="/markets", tags=["markets"])

_DATA_PATH = Path("data/processed/market_odds.parquet")


def _american_odds_to_probability(odds: float) -> float:
    """Convert American odds into a probability."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def _synthetic_frame() -> pd.DataFrame:
    """Generate a small synthetic odds snapshot with two sportsbooks."""
    now = datetime.now(UTC)
    synthetic_rows = [
        {
            "bout_id": "synthetic-bout",
            "sportsbook": "ExampleBookA",
            "book": "ExampleBookA",
            "price": -110.0,
            "implied_probability": _american_odds_to_probability(-110.0),
            "normalized_probability": _american_odds_to_probability(-110.0),
            "overround": 0.02,
            "z_shin": 0.0,
            "stale": False,
            "last_updated": now,
        },
        {
            "bout_id": "synthetic-bout",
            "sportsbook": "ExampleBookB",
            "book": "ExampleBookB",
            "price": 105.0,
            "implied_probability": _american_odds_to_probability(105.0),
            "normalized_probability": _american_odds_to_probability(105.0),
            "overround": 0.02,
            "z_shin": 0.0,
            "stale": False,
            "last_updated": now,
        },
    ]
    return pd.DataFrame(synthetic_rows)


def _load_market_frame(path: Path) -> pd.DataFrame:
    """Load the market snapshot from disk or fall back to synthetic data."""
    if path.exists():
        try:
            frame = pd.read_parquet(path)
        except Exception:
            frame = pd.DataFrame()
        if not frame.empty:
            return frame
    return _synthetic_frame()


def _coerce_float(value: object, default: float) -> float:
    """Safely coerce any value to a float with a default."""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def _coerce_bool(value: object, default: bool) -> bool:
    """Safely coerce any value to a boolean with a default."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "yes"}:
            return True
        if lowered in {"false", "f", "0", "no"}:
            return False
        return default
    if isinstance(value, int | float):
        return bool(value)
    return default


def _coerce_str(value: object, default: str) -> str:
    """Safely coerce any value to a string with a default."""
    if value is None:
        return default
    text = str(value)
    return text if text else default


def _coerce_datetime(value: object, default: datetime) -> datetime:
    """Safely coerce any value to an aware UTC datetime."""
    candidate: datetime | None
    if value is None:
        candidate = None
    elif isinstance(value, datetime):
        candidate = value
    elif isinstance(value, pd.Timestamp):
        candidate = value.to_pydatetime()
    else:
        try:
            converted = pd.to_datetime(value, utc=True)
        except (TypeError, ValueError):
            candidate = None
        else:
            if isinstance(converted, pd.Timestamp):
                candidate = converted.to_pydatetime()
            elif isinstance(converted, datetime):
                candidate = converted
            else:
                candidate = None
    if candidate is None:
        candidate = default
    if candidate.tzinfo is None:
        return candidate.replace(tzinfo=UTC)
    return candidate.astimezone(UTC)


def _iter_market_records(frame: pd.DataFrame) -> Iterable[MarketResponse]:
    """Yield MarketResponse objects from a raw dataframe."""
    records = frame.to_dict(orient="records")
    for raw in records:
        fallback_timestamp = datetime.now(UTC)
        timestamp_source = raw.get("last_updated")
        if timestamp_source is None:
            timestamp_source = raw.get("timestamp")
        last_updated = _coerce_datetime(timestamp_source, fallback_timestamp)

        sportsbook = _coerce_str(raw.get("sportsbook"), "Unknown")
        book = _coerce_str(raw.get("book"), sportsbook)

        price_source = raw.get("price")
        if price_source is None:
            price_source = raw.get("american_odds")
        price = _coerce_float(price_source, 0.0)

        implied_probability = _coerce_float(
            raw.get("implied_probability"),
            _american_odds_to_probability(price),
        )
        normalized_probability = _coerce_float(
            raw.get("normalized_probability"),
            implied_probability,
        )
        overround = _coerce_float(raw.get("overround"), 0.0)
        z_shin = _coerce_float(raw.get("z_shin"), 0.0)
        stale = _coerce_bool(raw.get("stale"), False)
        bout_id = _coerce_str(raw.get("bout_id"), "unknown-bout")

        yield MarketResponse(
            bout_id=bout_id,
            book=book,
            sportsbook=sportsbook,
            price=price,
            implied_probability=implied_probability,
            normalized_probability=normalized_probability,
            overround=overround,
            last_updated=last_updated,
            z_shin=z_shin,
            stale=stale,
        )


@router.get("/", response_model=list[MarketResponse])
def markets() -> list[MarketResponse]:
    """Return the latest market snapshot for each recorded bout."""
    frame = _load_market_frame(_DATA_PATH)
    responses = list(_iter_market_records(frame))
    return responses


__all__ = ["router"]
