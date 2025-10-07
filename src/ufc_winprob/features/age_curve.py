"""Division-aware age curve modelling using spline-based GAMs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from patsy import dmatrix

AGE_CURVE_DIR = Path("data/models/age_vectors")
AGE_CURVE_DIR.mkdir(parents=True, exist_ok=True)

DIVISION_ANCHORS: Dict[str, float] = {
    "HW": 33.0,
    "LHW": 32.0,
    "MW": 31.0,
    "WW": 30.0,
    "LW": 29.0,
    "FW": 28.0,
    "BW": 27.0,
    "FLW": 26.0,
}

AGE_KNOTS = [20.0, 25.0, 30.0, 33.0, 36.0, 39.0, 42.0]
_BS_FORMULA = (
    "bs(age, knots=knots[1:-1], degree=3, include_intercept=True, "
    "lower_bound=knots[0], upper_bound=knots[-1])"
)


@dataclass
class AgeCurveModel:
    """Represents a fitted division-specific age effect curve."""

    division: str
    coefficients: Dict[str, float]
    knots: Iterable[float]
    baseline_probability: float

    def _design(self, age: float) -> pd.Series:
        basis = dmatrix(
            _BS_FORMULA,
            {"age": [age], "knots": list(self.knots)},
            return_type="dataframe",
        )
        return basis.iloc[0]

    def probability(self, age: float) -> float:
        design_row = self._design(age)
        logit = 0.0
        for name, value in self.coefficients.items():
            logit += design_row.get(name, 0.0) * value
        return float(_expit(logit))

    def effect(self, age: float) -> float:
        prob = self.probability(age)
        effect = prob - self.baseline_probability
        return float(np.clip(effect, -0.5, 0.5))

    def save(self) -> Path:
        path = AGE_CURVE_DIR / f"{self.division}.json"
        payload = {
            "division": self.division,
            "coefficients": self.coefficients,
            "knots": list(self.knots),
            "baseline_probability": self.baseline_probability,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, division: str) -> "AgeCurveModel":
        path = AGE_CURVE_DIR / f"{division}.json"
        if not path.exists():
            model = cls.fit_from_history(division)
            model.save()
            return model
        data = json.loads(path.read_text(encoding="utf-8"))
        coeff_raw = data.get("coefficients", {})
        if isinstance(coeff_raw, list):
            model = cls.fit_from_history(division)
            model.save()
            return model
        coeffs = {k: float(v) for k, v in coeff_raw.items()}
        if not coeffs:
            coeffs = {"Intercept": 0.0}
        return cls(
            division=data["division"],
            coefficients=coeffs,
            knots=data.get("knots", AGE_KNOTS),
            baseline_probability=float(data.get("baseline_probability", 0.5)),
        )

    @classmethod
    def fit_from_history(cls, division: str) -> "AgeCurveModel":
        history = _load_age_outcomes(division)
        if history.empty:
            raise ValueError(f"No history available for division {division}")
        logger.info("Fitting GAM age curve for %s using %d rows", division, len(history))
        basis = dmatrix(
            _BS_FORMULA,
            {"age": history["age"], "knots": AGE_KNOTS},
            return_type="dataframe",
        )
        model = sm.GLM(
            history["wins"] / history["total"],
            basis,
            family=sm.families.Binomial(),
            freq_weights=history["total"],
        ).fit()
        anchor = DIVISION_ANCHORS.get(division, 30.0)
        anchor_design = dmatrix(
            _BS_FORMULA,
            {"age": [anchor], "knots": AGE_KNOTS},
            return_type="dataframe",
        ).iloc[0]
        baseline_logit = float(sum(anchor_design.get(name, 0.0) * value for name, value in model.params.items()))
        baseline_prob = float(_expit(baseline_logit))
        coefficients = {str(name): float(value) for name, value in model.params.to_dict().items()}
        return cls(
            division=division,
            coefficients=coefficients,
            knots=AGE_KNOTS,
            baseline_probability=baseline_prob,
        )

    def plot(self, path: Path) -> Path:
        ages = np.linspace(min(self.knots), max(self.knots), num=200)
        effects = [self.effect(age) for age in ages]
        df = pd.DataFrame({"age": ages, "effect": effects})
        ax = df.plot(x="age", y="effect", title=f"Age effect for {self.division}")
        ax.set_ylabel("Effect (probability delta)")
        ax.figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(path)
        ax.figure.clf()
        return path


def _load_age_outcomes(division: str) -> pd.DataFrame:
    dataset_path = Path("data/processed/training_features.parquet")
    if dataset_path.exists():
        frame = pd.read_parquet(dataset_path)
        if "division" in frame.columns:
            groups = []
            filtered = frame[frame["division"] == division]
            if filtered.empty:
                return pd.DataFrame()
            for _, row in filtered.iterrows():
                groups.append({"age": float(row.get("age_a", 30)), "wins": int(row["target"]), "total": 1})
                groups.append({"age": float(row.get("age_b", 30)), "wins": int(1 - row["target"]), "total": 1})
            history = pd.DataFrame(groups)
            history = history.groupby(pd.cut(history["age"], bins=np.linspace(20, 42, 12))).agg({"age": "mean", "wins": "sum", "total": "sum"}).dropna()
            history["age"] = history["age"].fillna(DIVISION_ANCHORS.get(division, 30.0))
            return history.reset_index(drop=True)
    # Synthetic fallback
    anchor = DIVISION_ANCHORS.get(division, 30.0)
    ages = np.linspace(20, 42, num=30)
    logits = 0.5 - 0.02 * (ages - anchor) ** 2
    probs = _expit(logits)
    total = np.full_like(probs, 40, dtype=int)
    wins = np.round(probs * total).astype(int)
    return pd.DataFrame({"age": ages, "wins": wins, "total": total})


@lru_cache(maxsize=16)
def load_age_model(division: str) -> AgeCurveModel:
    return AgeCurveModel.load(division)


def age_curve_effect(age: float, division: str) -> float:
    model = load_age_model(division)
    return model.effect(age)


def _expit(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


__all__ = ["AgeCurveModel", "age_curve_effect", "DIVISION_ANCHORS", "AGE_KNOTS", "load_age_model"]
