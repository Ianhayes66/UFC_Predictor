"""Odds conversion utilities including Shin overround removal."""

from __future__ import annotations

from math import sqrt
from typing import Iterable, List, Sequence

import numpy as np


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""

    if odds > 0:
        return 1 + odds / 100
    if odds < 0:
        return 1 + 100 / abs(odds)
    raise ValueError("American odds cannot be zero")


def decimal_to_american(odds: float) -> float:
    """Convert decimal odds to American odds."""

    if odds <= 1:
        raise ValueError("Decimal odds must be greater than 1")
    if odds >= 2:
        return (odds - 1) * 100
    return -100 / (odds - 1)


def american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability."""

    if odds > 0:
        return 100 / (odds + 100)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    raise ValueError("American odds cannot be zero")


def implied_to_american(prob: float) -> float:
    """Convert implied probability to American odds."""

    if not 0 < prob < 1:
        raise ValueError("Probability must be between 0 and 1")
    if prob >= 0.5:
        return -prob / (1 - prob) * 100
    return (1 - prob) / prob * 100


def normalize_probabilities(probabilities: Sequence[float]) -> List[float]:
    total = float(sum(probabilities))
    if total == 0:
        raise ValueError("Probability list sums to zero")
    return [p / total for p in probabilities]


def normalize_probabilities_shin(probabilities: Sequence[float]) -> List[float]:
    adjusted, _ = shin_adjustment(probabilities)
    return adjusted


def shin_adjustment(probabilities: Sequence[float]) -> tuple[List[float], float]:
    """Return Shin-normalized probabilities and imbalance parameter."""

    probs = np.array(probabilities, dtype=float)
    if np.any(probs <= 0):
        raise ValueError("Probabilities must be positive for Shin normalization")
    if probs.size == 0:
        return [], 0.0

    q = probs / probs.sum()

    def equation(z: float) -> float:
        term = np.sqrt(z**2 + 4 * (1 - z) * (q**2)).sum()
        return term - 2

    z_low, z_high = 0.0, 0.25
    for _ in range(100):
        z_mid = (z_low + z_high) / 2
        if equation(z_mid) > 0:
            z_high = z_mid
        else:
            z_low = z_mid
    z_star = (z_low + z_high) / 2

    adjusted = (np.sqrt(z_star**2 + 4 * (1 - z_star) * (q**2)) - z_star) / (2 * (1 - z_star))
    adjusted = adjusted / adjusted.sum()
    return adjusted.tolist(), float(z_star)


def overround(probabilities: Iterable[float]) -> float:
    """Return the overround given implied probabilities."""

    total = float(sum(probabilities))
    return total - 1.0


__all__ = [
    "american_to_decimal",
    "decimal_to_american",
    "american_to_implied",
    "implied_to_american",
    "normalize_probabilities",
    "normalize_probabilities_shin",
    "shin_adjustment",
    "overround",
]
