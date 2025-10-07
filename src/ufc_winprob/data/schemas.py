"""Pydantic schemas for core domain entities."""

from __future__ import annotations

from datetime import date, datetime
from hashlib import sha1
from typing import List, Optional

from pydantic import BaseModel, Field, computed_field


class Fighter(BaseModel):
    fighter_id: str
    name: str
    dob: Optional[date] = None
    height_cm: Optional[float] = None
    reach_cm: Optional[float] = None
    stance: Optional[str] = None
    division: Optional[str] = None
    gym: Optional[str] = None
    country: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)

    @computed_field
    def age(self) -> Optional[int]:  # type: ignore[override]
        if not self.dob:
            return None
        return int((datetime.utcnow().date() - self.dob).days // 365)

    @classmethod
    def create(cls, name: str, dob: Optional[date] = None, **kwargs: object) -> "Fighter":
        identifier = sha1((name + (dob.isoformat() if dob else "")).encode("utf-8")).hexdigest()[:12]
        return cls(fighter_id=identifier, name=name, dob=dob, **kwargs)


class Event(BaseModel):
    event_id: str
    name: str
    date: datetime
    location: Optional[str] = None


class BoutStats(BaseModel):
    sig_strikes_landed: int = 0
    sig_strikes_attempted: int = 0
    takedowns: int = 0
    control_seconds: float = 0.0
    knockdowns: int = 0
    submission_attempts: int = 0


class Bout(BaseModel):
    bout_id: str
    event_id: str
    fighter_id: str
    opponent_id: str
    winner: Optional[str]
    weight_class: str
    scheduled_rounds: int
    method: Optional[str]
    end_round: Optional[int]
    end_time_seconds: Optional[int]
    result: Optional[str]
    stats: BoutStats = Field(default_factory=BoutStats)
    event_date: datetime


class OddsSnapshot(BaseModel):
    bout_id: str
    sportsbook: str
    timestamp: datetime
    american_odds: float
    implied_probability: float
    overround: float
    normalized_probability: float


class StyleProfile(BaseModel):
    fighter_id: str
    components: List[str]
    strengths: List[float]


class ComponentElo(BaseModel):
    fighter_id: str
    division: str
    components: List[str]
    ratings: List[float]
    uncertainties: List[float]
    last_updated: datetime

    def as_dict(self) -> dict[str, object]:
        return self.model_dump()


class Prediction(BaseModel):
    bout_id: str
    fighter_id: str
    opponent_id: str
    division: str
    prob_win: float
    prob_lose: float
    prob_draw: float = 0.0
    prob_interval: tuple[float, float] = (0.0, 1.0)
    market_probability: Optional[float] = None
    edge: Optional[float] = None


class Recommendation(BaseModel):
    bout_id: str
    fighter_id: str
    sportsbook: str
    price: float
    probability: float
    expected_value: float
    kelly_fraction: float
    stale: bool


__all__ = [
    "Fighter",
    "Event",
    "Bout",
    "BoutStats",
    "OddsSnapshot",
    "StyleProfile",
    "ComponentElo",
    "Prediction",
    "Recommendation",
]
