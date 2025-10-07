"""Bout normalization routines."""

from __future__ import annotations

from typing import Iterable, List

from ..data.schemas import Bout


def normalize_bouts(bouts: Iterable[Bout]) -> List[Bout]:
    normalized: List[Bout] = []
    for bout in bouts:
        weight_class = bout.weight_class.upper()
        normalized.append(bout.model_copy(update={"weight_class": weight_class}))
    return normalized


__all__ = ["normalize_bouts"]
