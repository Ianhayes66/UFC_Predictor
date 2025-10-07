"""Metrics utilities for calibration and ROI evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    """Container for metric values with optional confidence interval."""

    value: float
    lower: float | None = None
    upper: float | None = None


def brier_score(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    true = np.asarray(list(y_true), dtype=float)
    prob = np.asarray(list(y_prob), dtype=float)
    return float(np.mean((prob - true) ** 2))


def expected_calibration_error(y_true: Iterable[int], y_prob: Iterable[float], bins: int = 20) -> float:
    true = np.asarray(list(y_true), dtype=float)
    prob = np.asarray(list(y_prob), dtype=float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (prob >= lower) & (prob < upper)
        if not np.any(mask):
            continue
        bucket_true = true[mask]
        bucket_prob = prob[mask]
        acc = bucket_true.mean()
        conf = bucket_prob.mean()
        ece += np.abs(acc - conf) * (len(bucket_true) / len(true))
    return float(ece)


def roi(y_true: Iterable[int], y_prob: Iterable[float], prices: Iterable[float]) -> float:
    true = np.asarray(list(y_true), dtype=float)
    prob = np.asarray(list(y_prob), dtype=float)
    odds = np.asarray(list(prices), dtype=float)
    returns = true * (odds - 1) - (1 - true)
    weights = prob
    return float(np.sum(returns * weights) / np.sum(np.abs(weights)))


def bin_counts(probabilities: Iterable[float], bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(list(probabilities), dtype=float)
    hist, edges = np.histogram(probs, bins=bins, range=(0, 1))
    return hist, edges


__all__ = ["MetricResult", "brier_score", "expected_calibration_error", "roi", "bin_counts"]
