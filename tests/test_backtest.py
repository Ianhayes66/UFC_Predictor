from __future__ import annotations

import numpy as np

from ufc_winprob.models.backtest import run_backtest


def test_backtest_roi_in_range() -> None:
    prob = np.linspace(0.4, 0.7, 10)
    outcomes = (prob > 0.5).astype(int)
    prices = np.full_like(prob, 2.0)
    result = run_backtest(prob, outcomes, prices)
    assert -1.0 <= result.roi <= 1.0
