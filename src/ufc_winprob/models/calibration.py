"""Calibration helpers for converting raw model scores to probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class Calibrator:
    """Per-division probability calibrator."""

    method: str
    models: Mapping[str, object]
    default_division: str

    def transform(self, scores: Iterable[float], divisions: Iterable[str]) -> np.ndarray:
        values = np.asarray(list(scores), dtype=float)
        divs = list(divisions)
        calibrated = np.zeros_like(values, dtype=float)
        for idx, (score, division) in enumerate(zip(values, divs)):
            model = self.models.get(division) or self.models[self.default_division]
            calibrated[idx] = _apply_model(model, self.method, np.asarray([score], dtype=float))[0]
        return np.clip(calibrated, 1e-6, 1 - 1e-6)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"method": self.method, "models": dict(self.models), "default_division": self.default_division}
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        payload: Dict[str, object] = joblib.load(path)
        return cls(method=str(payload["method"]), models=payload["models"], default_division=str(payload["default_division"]))


def train_calibrator(
    scores: Iterable[float],
    targets: Iterable[int],
    divisions: Iterable[str],
    method: str = "isotonic",
) -> Calibrator:
    values = np.asarray(list(scores), dtype=float)
    target_arr = np.asarray(list(targets), dtype=float)
    division_list = list(divisions)
    unique_divisions = sorted(set(division_list)) or ["GLOBAL"]
    models: Dict[str, object] = {}
    for division in unique_divisions:
        mask = [div == division for div in division_list]
        if sum(mask) < 3:
            continue
        model = _train_single(values[mask], target_arr[mask], method)
        models[division] = model
    if not models:
        models["GLOBAL"] = _train_single(values, target_arr, method)
    default_division = next(iter(models))
    if "GLOBAL" not in models:
        models["GLOBAL"] = _train_single(values, target_arr, method)
    return Calibrator(method=method, models=models, default_division=default_division)


def _train_single(scores: np.ndarray, targets: np.ndarray, method: str) -> object:
    if method == "platt":
        clf = LogisticRegression(max_iter=500)
        clf.fit(scores.reshape(-1, 1), targets)
        return clf
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(scores, targets)
        return iso
    raise ValueError(f"Unsupported calibration method: {method}")


def _apply_model(model: object, method: str, scores: np.ndarray) -> np.ndarray:
    if method == "platt":
        assert isinstance(model, LogisticRegression)
        probs = model.predict_proba(scores.reshape(-1, 1))[:, 1]
        return probs
    if method == "isotonic":
        assert isinstance(model, IsotonicRegression)
        return model.transform(scores)
    raise ValueError(f"Unsupported calibration method: {method}")


__all__ = ["Calibrator", "train_calibrator"]
