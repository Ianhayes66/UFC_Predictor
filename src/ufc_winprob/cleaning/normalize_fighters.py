"""Fighter normalization routines."""

from __future__ import annotations

import unicodedata
from typing import Iterable, List

from ..data.schemas import Fighter


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return " ".join(part.capitalize() for part in text.split())


def normalize_fighters(fighters: Iterable[Fighter]) -> List[Fighter]:
    normalized: List[Fighter] = []
    for fighter in fighters:
        normalized.append(fighter.model_copy(update={"name": normalize_name(fighter.name)}))
    return normalized


__all__ = ["normalize_fighters", "normalize_name"]
