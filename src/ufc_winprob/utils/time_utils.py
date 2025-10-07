"""Time handling utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Union

import pandas as pd

DateLike = Union[str, datetime, pd.Timestamp]


def as_utc(value: DateLike) -> datetime:
    """Convert the given value into a timezone-aware UTC datetime."""

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, str):
        dt = pd.to_datetime(value, utc=True).to_pydatetime()
    else:  # pragma: no cover - defensive
        raise TypeError(f"Unsupported date type: {type(value)!r}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def now_utc() -> datetime:
    """Return the current UTC datetime."""

    return datetime.now(timezone.utc)


def days_between(start: DateLike, end: DateLike) -> float:
    """Return the number of days between two instants."""

    start_dt = as_utc(start)
    end_dt = as_utc(end)
    delta = end_dt - start_dt
    return delta.total_seconds() / 86400.0


__all__ = ["as_utc", "now_utc", "days_between"]
