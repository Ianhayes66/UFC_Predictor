"""Style taxonomy definitions for component Elo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

COMPONENT_WEIGHTS: Dict[str, float] = {
    "striking": 1.0,
    "grappling": 0.9,
    "wrestling": 0.95,
    "submissions": 0.85,
    "cardio": 0.9,
    "durability": 0.8,
    "iq": 0.75,
    "aggression": 0.7,
}


@dataclass(frozen=True)
class StyleInteraction:
    offensive: str
    defensive: str
    multiplier: float


STYLE_INTERACTIONS = [
    StyleInteraction("striking", "durability", 1.05),
    StyleInteraction("grappling", "cardio", 1.03),
    StyleInteraction("wrestling", "aggression", 1.02),
]


__all__ = ["COMPONENT_WEIGHTS", "STYLE_INTERACTIONS", "StyleInteraction"]
