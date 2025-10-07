from __future__ import annotations

from ufc_winprob.features.age_curve import AgeCurveModel, age_adjustment


def test_age_curve_effects_within_bounds() -> None:
    model = AgeCurveModel.fit_from_anchor("LW")
    for age in range(20, 41):
        effect = model.effect(age)
        assert -0.5 <= effect <= 0.5


def test_age_adjustment_round_trip() -> None:
    effect = age_adjustment(30, "LW")
    assert isinstance(effect, float)
