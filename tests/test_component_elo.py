from __future__ import annotations

import numpy as np

from ufc_winprob.features.component_elo import EloVector, elo_expectation, initial_vector, update_elo


def test_initial_vector() -> None:
    vec = initial_vector(["striking", "grappling"])
    assert np.allclose(vec.ratings, 1500)
    assert np.allclose(vec.uncertainties, 250)


def test_elo_expectation_symmetry() -> None:
    a = initial_vector(["striking", "grappling"])
    b = initial_vector(["striking", "grappling"])
    expectation = elo_expectation(a, b)
    assert expectation == 0.5


def test_update_moves_ratings() -> None:
    a = initial_vector(["striking", "grappling"])
    b = initial_vector(["striking", "grappling"])
    updated_a, updated_b = update_elo(a, b, result=1.0, k_factor=24.0)
    assert updated_a.ratings.mean() > a.ratings.mean()
    assert updated_b.ratings.mean() < b.ratings.mean()
