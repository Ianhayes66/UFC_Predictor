"""High-level Elo update routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from ..data.schemas import Bout
from .age_curve import age_adjustment
from .component_elo import EloVector, initial_vector, update_elo


@dataclass
class EloState:
    ratings: Dict[str, EloVector]
    components: Iterable[str]

    def get_vector(self, fighter_id: str) -> EloVector:
        if fighter_id not in self.ratings:
            self.ratings[fighter_id] = initial_vector(self.components)
        return self.ratings[fighter_id]

    def update(self, bout: Bout, result: float, division: str, ages: Tuple[float, float]) -> None:
        vector_a = self.get_vector(bout.fighter_id)
        vector_b = self.get_vector(bout.opponent_id)
        age_effect_a = age_adjustment(ages[0], division)
        age_effect_b = age_adjustment(ages[1], division)
        k_factor = 24 * (1 + age_effect_a - age_effect_b)
        updated_a, updated_b = update_elo(vector_a, vector_b, result, k_factor)
        self.ratings[bout.fighter_id] = updated_a
        self.ratings[bout.opponent_id] = updated_b


__all__ = ["EloState"]
