from __future__ import annotations

from ufc_winprob.features.age_curve import AgeCurveModel, age_curve_effect, load_age_model


def test_age_curve_effects_within_bounds() -> None:
    model = AgeCurveModel.fit_from_history("LW")
    for age in range(20, 41):
        effect = model.effect(age)
        assert -0.5 <= effect <= 0.5


def test_age_adjustment_round_trip() -> None:
    effect = age_curve_effect(30, "LW")
    assert isinstance(effect, float)


def test_age_curve_monotonic_mid_thirties() -> None:
    model = load_age_model("LW")
    younger = model.effect(28)
    prime = model.effect(32)
    veteran = model.effect(38)
    assert prime <= younger + 0.05
    assert veteran <= prime + 0.05
