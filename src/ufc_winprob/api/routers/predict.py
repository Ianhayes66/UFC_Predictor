"""Predict endpoint for ad-hoc probability requests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException

from ufc_winprob.api.schemas import PredictionResponse, PredictRequest

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    import pandas as pd

    from ufc_winprob.models.calibration import Calibrator

router = APIRouter(tags=["probabilities"])

_DEFAULT_DIVISION = "LW"
_DEFAULT_RATING = 1500.0
_DEFAULT_AGE = 30.0
_DEFAULT_HEIGHT = 70.0
_DEFAULT_REACH = 70.0
_DEFAULT_ACTIVITY_GAP = 60.0
_DEFAULT_SCHEDULED_ROUNDS = 3.0
_CONFIDENCE_DELTA = 0.05


def _validate_payload(payload: PredictRequest) -> None:
    """Ensure numeric payload fields are positive when provided."""
    numeric_fields = (
        "fighter_age",
        "opponent_age",
        "fighter_height",
        "opponent_height",
        "fighter_reach",
        "opponent_reach",
    )
    for field in numeric_fields:
        value = getattr(payload, field)
        if value is not None and value <= 0:
            raise HTTPException(status_code=422, detail=f"{field} must be positive")


def _neutral_features(feature_names: Iterable[str], market_probability: float) -> dict[str, float]:
    """Return neutral defaults compatible with the trained model."""
    features: dict[str, float] = {}
    for name in feature_names:
        if name.startswith("fighter_") or name.startswith("opponent_"):
            features[name] = _DEFAULT_RATING
        elif name == "market_prob":
            features[name] = market_probability
        elif name == "scheduled_rounds":
            features[name] = _DEFAULT_SCHEDULED_ROUNDS
        elif name == "activity_gap":
            features[name] = _DEFAULT_ACTIVITY_GAP
        else:
            features[name] = 0.0
    return features


def _set_pair_features(
    features: dict[str, float],
    prefix: str,
    fighter_value: float,
    opponent_value: float,
) -> None:
    """Populate mirrored and differential features when available."""
    primary = f"{prefix}_a"
    secondary = f"{prefix}_b"
    diff_name = f"{prefix}_diff"
    if primary in features:
        features[primary] = fighter_value
    if secondary in features:
        features[secondary] = opponent_value
    if diff_name in features:
        features[diff_name] = fighter_value - opponent_value


def _set_single_feature(features: dict[str, float], name: str, value: float) -> None:
    if name in features:
        features[name] = value


def _build_feature_frame(
    payload: PredictRequest,
    feature_names: Iterable[str],
    market_probability: float,
    division: str,
) -> pd.DataFrame:
    """Construct the single-row feature frame expected by the classifier."""
    import numpy as np
    import pandas as pd

    from ufc_winprob.features.age_curve import age_curve_effect

    fighter_age = float(payload.fighter_age or _DEFAULT_AGE)
    opponent_age = float(payload.opponent_age or _DEFAULT_AGE)
    fighter_height = float(payload.fighter_height or _DEFAULT_HEIGHT)
    opponent_height = float(payload.opponent_height or _DEFAULT_HEIGHT)
    fighter_reach = float(payload.fighter_reach or _DEFAULT_REACH)
    opponent_reach = float(payload.opponent_reach or _DEFAULT_REACH)

    features = _neutral_features(feature_names, market_probability)

    _set_pair_features(features, "age", fighter_age, opponent_age)
    _set_pair_features(features, "height", fighter_height, opponent_height)
    _set_pair_features(features, "reach", fighter_reach, opponent_reach)

    age_effect_fighter = float(age_curve_effect(fighter_age, division=division))
    age_effect_opponent = float(age_curve_effect(opponent_age, division=division))
    _set_pair_features(features, "age_effect", age_effect_fighter, age_effect_opponent)

    _set_single_feature(features, "fighter_age", fighter_age)
    _set_single_feature(features, "opponent_age", opponent_age)
    _set_single_feature(features, "fighter_height", fighter_height)
    _set_single_feature(features, "opponent_height", opponent_height)
    _set_single_feature(features, "fighter_reach", fighter_reach)
    _set_single_feature(features, "opponent_reach", opponent_reach)

    columns = list(feature_names)
    frame = pd.DataFrame([[features[name] for name in columns]], columns=columns)
    return frame.astype(np.float64, copy=False)


def _load_model_artifacts() -> tuple[Any, Calibrator]:
    """Load classifier and calibrator artifacts from disk."""
    from ufc_winprob.models.calibration import Calibrator
    from ufc_winprob.models.persistence import MODEL_DIR, load_artifact

    try:
        classifier: Any = load_artifact("classifier.joblib")
        calibrator = Calibrator.load(MODEL_DIR / "calibrator.joblib")
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=503, detail="Model artifacts unavailable") from exc
    return classifier, calibrator


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest) -> PredictionResponse:
    """Generate a win probability for the requested fighter."""
    _validate_payload(payload)

    american_odds = payload.american_odds
    market_probability: float | None = None
    if american_odds is not None:
        from ufc_winprob.utils.odds_utils import american_to_implied

        try:
            market_probability = float(american_to_implied(american_odds))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    classifier, calibrator = _load_model_artifacts()
    feature_names = getattr(classifier, "feature_names_in_", None)
    if feature_names is None:
        raise HTTPException(status_code=500, detail="Model missing feature metadata")

    division = (payload.division or calibrator.default_division or _DEFAULT_DIVISION).upper()
    base_market_probability = market_probability if market_probability is not None else 0.5
    frame = _build_feature_frame(payload, feature_names, base_market_probability, division)

    try:
        probabilities = classifier.predict_proba(frame)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail="Model inference failed") from exc

    raw_probability = float(probabilities[0, 1])
    calibrated = float(calibrator.transform([raw_probability], [division])[0])
    probability_low = max(0.0, calibrated - _CONFIDENCE_DELTA)
    probability_high = min(1.0, calibrated + _CONFIDENCE_DELTA)

    return PredictionResponse(
        bout_id=payload.bout_id,
        fighter=payload.fighter,
        probability=calibrated,
        probability_low=probability_low,
        probability_high=probability_high,
        market_probability=market_probability,
    )


__all__ = ["PredictRequest", "PredictionResponse", "router"]
