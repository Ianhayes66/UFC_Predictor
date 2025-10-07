"""Component-wise Elo rating implementation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..data.schemas import ComponentElo
from ..utils.time_utils import as_utc
from .style_taxonomy import COMPONENT_WEIGHTS

DEFAULT_RATING = 1500.0
DEFAULT_UNCERTAINTY = 250.0


@dataclass
class EloVector:
    components: List[str]
    ratings: np.ndarray
    uncertainties: np.ndarray

    def as_component_elo(self, fighter_id: str, division: str) -> ComponentElo:
        return ComponentElo(
            fighter_id=fighter_id,
            division=division,
            components=self.components,
            ratings=self.ratings.tolist(),
            uncertainties=self.uncertainties.tolist(),
            last_updated=as_utc("1970-01-01T00:00:00Z"),
        )


def logistic_expectation(delta: float) -> float:
    return 1.0 / (1.0 + exp(-delta / 400.0))


def elo_expectation(vector_a: EloVector, vector_b: EloVector) -> float:
    weights = np.array([COMPONENT_WEIGHTS.get(component, 1.0) for component in vector_a.components])
    delta = ((vector_a.ratings - vector_b.ratings) * weights).sum()
    return logistic_expectation(delta)


def update_elo(
    vector_a: EloVector,
    vector_b: EloVector,
    result: float,
    k_factor: float,
) -> Tuple[EloVector, EloVector]:
    expected_a = elo_expectation(vector_a, vector_b)
    expected_b = 1.0 - expected_a

    diff_a = result - expected_a
    diff_b = (1 - result) - expected_b

    weights = np.array([COMPONENT_WEIGHTS.get(component, 1.0) for component in vector_a.components])

    new_ratings_a = vector_a.ratings + k_factor * diff_a * weights
    new_ratings_b = vector_b.ratings + k_factor * diff_b * weights

    new_uncertainty_a = np.maximum(vector_a.uncertainties * 0.98, 50.0)
    new_uncertainty_b = np.maximum(vector_b.uncertainties * 0.98, 50.0)

    return (
        EloVector(vector_a.components, new_ratings_a, new_uncertainty_a),
        EloVector(vector_b.components, new_ratings_b, new_uncertainty_b),
    )


def initial_vector(components: Iterable[str]) -> EloVector:
    comps = list(components)
    ratings = np.full(len(comps), DEFAULT_RATING)
    uncertainties = np.full(len(comps), DEFAULT_UNCERTAINTY)
    return EloVector(comps, ratings, uncertainties)


def expected_margin(vector_a: EloVector, vector_b: EloVector) -> float:
    expected = elo_expectation(vector_a, vector_b)
    return (expected - 0.5) * 10


__all__ = [
    "EloVector",
    "ComponentElo",
    "initial_vector",
    "update_elo",
    "elo_expectation",
    "expected_margin",
]
