"""Utility functions for evaluating probabilistic forecasts and betting returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class MetricResult:
    """Container for a metric value with optional confidence intervals."""

    value: float
    lower: float | None = None
    upper: float | None = None


def _to_1d_float(array: ArrayLike) -> np.ndarray:
    """Convert an arbitrary array-like structure to a one-dimensional float array."""
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _validate_binary(name: str, array: np.ndarray) -> None:
    """Ensure that an array only contains binary outcomes (0 or 1)."""
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must only contain values in [0, 1].")
    unique_values = np.unique(array)
    if not np.all(np.isin(unique_values, (0.0, 1.0))):
        raise ValueError(f"{name} must be binary (0 or 1).")


def _validate_probabilities(name: str, array: np.ndarray) -> None:
    """Ensure that an array represents valid probabilities."""
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must only contain probabilities in [0, 1].")


def _validate_shapes(*arrays: np.ndarray) -> None:
    """Ensure that all arrays share the same length."""
    lengths = {array.shape[0] for array in arrays}
    if len(lengths) != 1:
        raise ValueError("All inputs must share the same length.")


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Compute the Brier score for binary probabilistic predictions."""
    true = _to_1d_float(y_true)
    prob = _to_1d_float(y_prob)
    _validate_shapes(true, prob)
    _validate_binary("y_true", true)
    _validate_probabilities("y_prob", prob)
    return float(np.mean((prob - true) ** 2))


def expected_calibration_error(y_true: ArrayLike, y_prob: ArrayLike, bins: int = 20) -> float:
    """Compute the Expected Calibration Error (ECE) for probabilistic forecasts."""
    if bins <= 0:
        raise ValueError("bins must be a positive integer.")
    true = _to_1d_float(y_true)
    prob = _to_1d_float(y_prob)
    _validate_shapes(true, prob)
    _validate_binary("y_true", true)
    _validate_probabilities("y_prob", prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = true.shape[0]
    ece = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (prob >= lower) & (prob <= upper)
        else:
            mask = (prob >= lower) & (prob < upper)
        if not np.any(mask):
            continue
        bucket_true = true[mask]
        bucket_prob = prob[mask]
        accuracy = float(bucket_true.mean())
        confidence = float(bucket_prob.mean())
        weight = bucket_true.size / total
        ece += abs(accuracy - confidence) * weight
    return float(ece)


def reliability_bins(
    y_true: ArrayLike, y_prob: ArrayLike, bins: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Compute average confidence and accuracy per probability bin."""
    if bins <= 0:
        raise ValueError("bins must be a positive integer.")
    true = _to_1d_float(y_true)
    prob = _to_1d_float(y_prob)
    _validate_shapes(true, prob)
    _validate_binary("y_true", true)
    _validate_probabilities("y_prob", prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    accuracy: list[float] = []
    confidence: list[float] = []
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (prob >= lower) & (prob <= upper)
        else:
            mask = (prob >= lower) & (prob < upper)
        if not np.any(mask):
            continue
        accuracy.append(float(true[mask].mean()))
        confidence.append(float(prob[mask].mean()))
    return np.asarray(confidence, dtype=float), np.asarray(accuracy, dtype=float)


def roi(y_true: ArrayLike, y_prob: ArrayLike, prices: ArrayLike) -> float:
    """Compute the weighted average return on investment for a betting strategy."""
    true = _to_1d_float(y_true)
    prob = _to_1d_float(y_prob)
    odds = _to_1d_float(prices)
    _validate_shapes(true, prob, odds)
    _validate_binary("y_true", true)
    _validate_probabilities("y_prob", prob)
    if np.any(odds <= 0.0):
        raise ValueError("prices (odds) must be strictly positive.")
    profits = np.where(true > 0.5, odds - 1.0, -1.0)
    stakes = np.clip(prob, 0.0, None)
    total_stake = float(np.sum(stakes))
    if np.isclose(total_stake, 0.0):
        return 0.0
    return float(np.dot(profits, stakes) / total_stake)


def bin_counts(probs: ArrayLike, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Count probabilities within equally spaced bins across the [0, 1] range."""
    if bins <= 0:
        raise ValueError("bins must be a positive integer.")
    probabilities = _to_1d_float(probs)
    _validate_probabilities("probs", probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(probabilities, bins=edges)
    return counts.astype(int), edges


__all__ = [
    "MetricResult",
    "bin_counts",
    "brier_score",
    "expected_calibration_error",
    "reliability_bins",
    "roi",
]
