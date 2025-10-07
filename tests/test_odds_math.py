from __future__ import annotations

import math

import numpy as np
from hypothesis import given, strategies as st

from ufc_winprob.utils.odds_utils import (
    american_to_decimal,
    american_to_implied,
    implied_to_american,
    normalize_probabilities_shin,
)


@given(st.floats(min_value=0.05, max_value=0.95))
def test_probability_conversion_round_trip(prob: float) -> None:
    american = implied_to_american(prob)
    recon = american_to_implied(american)
    assert math.isclose(prob, recon, rel_tol=1e-3)


def test_shin_normalization() -> None:
    implied = [0.55, 0.52]
    normalized = normalize_probabilities_shin(implied)
    assert math.isclose(sum(normalized), 1.0, rel_tol=1e-6)
