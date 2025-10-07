"""Division-aware age curve modelling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from ..utils.time_utils import as_utc

AGE_CURVE_DIR = Path("data/models/age_vectors")
AGE_CURVE_DIR.mkdir(parents=True, exist_ok=True)

DIVISION_ANCHORS = {
    "HW": 33,
    "LHW": 32,
    "MW": 31,
    "WW": 30,
    "LW": 29,
    "FW": 28,
    "BW": 27,
    "FLW": 26,
}


@dataclass
class AgeCurveModel:
    division: str
    coefficients: np.ndarray

    def effect(self, age: float) -> float:
        center = DIVISION_ANCHORS.get(self.division, 30)
        x = (age - center) / 5
        poly = np.polyval(self.coefficients, x)
        return float(np.clip(poly, -0.5, 0.5))

    def save(self) -> Path:
        path = AGE_CURVE_DIR / f"{self.division}.json"
        data = {
            "division": self.division,
            "coefficients": self.coefficients.tolist(),
        }
        path.write_text(pd.Series(data).to_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, division: str) -> "AgeCurveModel":
        path = AGE_CURVE_DIR / f"{division}.json"
        if not path.exists():
            model = cls.fit_from_anchor(division)
            model.save()
        else:
            series = pd.read_json(path)
            model = cls(division=division, coefficients=np.asarray(series["coefficients"].tolist()))
        return model

    @classmethod
    def fit_from_anchor(cls, division: str) -> "AgeCurveModel":
        center = DIVISION_ANCHORS.get(division, 30)
        ages = np.linspace(center - 10, center + 10, num=5)
        effects = -((ages - center) ** 2) / 150 + 0.1
        coefficients = np.polyfit((ages - center) / 5, effects, deg=2)
        return cls(division=division, coefficients=coefficients)


def age_adjustment(age: float, division: str) -> float:
    model = AgeCurveModel.load(division)
    return model.effect(age)


__all__ = ["AgeCurveModel", "age_adjustment", "DIVISION_ANCHORS"]
