"""API schemas using Pydantic models."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class PredictionResponse(BaseModel):
    bout_id: str
    fighter: str
    probability: float
    probability_low: float
    probability_high: float
    market_probability: Optional[float]


class RecommendationResponse(BaseModel):
    bout_id: str
    fighter: str
    sportsbook: str
    price: float
    probability: float
    expected_value: float
    kelly: float
    stale: bool


class MarketResponse(BaseModel):
    bout_id: str
    book: str
    sportsbook: str
    price: float
    implied_probability: float
    normalized_probability: float
    overround: float
    last_updated: datetime
    z_shin: float
    stale: bool


class PredictRequest(BaseModel):
    bout_id: str
    fighter: str
    opponent: str
    american_odds: Optional[float] = None
    fighter_age: Optional[float] = None
    opponent_age: Optional[float] = None
    fighter_height: Optional[float] = None
    opponent_height: Optional[float] = None
    fighter_reach: Optional[float] = None
    opponent_reach: Optional[float] = None


__all__ = [
    "HealthResponse",
    "PredictionResponse",
    "RecommendationResponse",
    "MarketResponse",
    "PredictRequest",
]
