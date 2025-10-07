from __future__ import annotations

import pandas as pd

from ufc_winprob.models.selection import SelectionConfig, kelly_fraction, rank_recommendations


def test_kelly_fraction_monotonic() -> None:
    low = kelly_fraction(0.55, 2.0)
    high = kelly_fraction(0.65, 2.0)
    assert high >= low


def test_rank_recommendations_filters_by_ev() -> None:
    df = pd.DataFrame(
        {
            "bout_id": ["a", "b"],
            "fighter": ["A", "B"],
            "probability": [0.6, 0.4],
            "american_odds": [150, 150],
        }
    )
    ranked = rank_recommendations(df, SelectionConfig(min_ev=0.01))
    assert "expected_value" in ranked.columns
    assert (ranked["expected_value"] >= 0.01).all()
