from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ufc_winprob.utils.metrics import (
    bin_counts,
    brier_score,
    expected_calibration_error,
    roi,
)


def test_brier_score_matches_manual_computation() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2])
    expected = np.mean((y_prob - y_true) ** 2)
    score = brier_score(y_true, y_prob)
    assert_allclose(score, expected)
    if score < 0.0:
        pytest.fail("Brier score must be non-negative.")


def test_expected_calibration_error_range_and_perfect_case() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_prob_perfect = np.array([0.0, 1.0, 0.0, 1.0])
    assert_allclose(expected_calibration_error(y_true, y_prob_perfect), 0.0)
    y_prob_imperfect = np.array([0.25, 0.75, 0.25, 0.75])
    value = expected_calibration_error(y_true, y_prob_imperfect, bins=2)
    if not (0.0 <= value <= 1.0):
        pytest.fail("ECE should lie within [0, 1].")


def test_roi_weighted_average_returns() -> None:
    y_true = np.array([1, 0, 1])
    y_prob = np.array([0.5, 0.2, 0.3])
    prices = np.array([2.0, 2.0, 2.0])
    profits = np.array([1.0, -1.0, 1.0])
    expected = np.average(profits, weights=y_prob)
    assert_allclose(roi(y_true, y_prob, prices), expected)


def test_bin_counts_cover_probability_range() -> None:
    probs = np.array([0.1, 0.2, 0.35, 0.9])
    counts, edges = bin_counts(probs, bins=4)
    if counts.sum() != len(probs):
        pytest.fail("Counts should match the number of probabilities.")
    if edges.shape != (5,):
        pytest.fail("Bin edges should include the endpoints.")
    assert_allclose(edges[0], 0.0)
    assert_allclose(edges[-1], 1.0)
