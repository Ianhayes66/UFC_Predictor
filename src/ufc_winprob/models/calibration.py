"""Calibration helpers for converting raw model scores to probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class Calibrator:
    model: IsotonicRegression

    def transform(self, scores: Iterable[float]) -> np.ndarray:
        return self.model.transform(np.asarray(list(scores), dtype=float))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        model = joblib.load(path)
        return cls(model)


def train_calibrator(scores: Iterable[float], targets: Iterable[int]) -> Calibrator:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(np.asarray(list(scores), dtype=float), np.asarray(list(targets), dtype=float))
    return Calibrator(model=model)


__all__ = ["Calibrator", "train_calibrator"]
