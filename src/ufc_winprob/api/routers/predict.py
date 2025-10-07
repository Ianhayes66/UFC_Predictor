"""Predict endpoint for ad-hoc probability requests."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ufc_winprob.api.schemas import PredictionResponse, PredictRequest
from ufc_winprob.features.age_curve import age_curve_effect
from ufc_winprob.models.calibration import Calibrator
from ufc_winprob.models.persistence import MODEL_DIR, load_artifact
from ufc_winprob.utils.odds_utils import american_to_implied

router = APIRouter(tags=["probabilities"])

_DEFAULT_RATING = 1500.0
_DEFAULT_HEIGHT = 72.0
_DEFAULT_REACH = 72.0
_DEFAULT_AGE = 30.0
_DEFAULT_ACTIVITY_GAP = 60.0
_DEFAULT_DIVISION = "LW"


def _neutral_feature_vector(feature_names: Iterable[str]) -> dict[str, float]:
    """Return a neutral feature dictionary compatible with the trained model."""
    features: dict[str, float] = {}
    for name in feature_names:
        if name.startswith("fighter_") or name.startswith("opponent_"):
            features[name] = _DEFAULT_RATING
        elif name == "scheduled_rounds":
            features[name] = 3.0
        elif name == "is_main_event":
            features[name] = 0.0
        elif name == "activity_gap":
            features[name] = _DEFAULT_ACTIVITY_GAP
        elif name == "market_prob":
            features[name] = 0.5
        elif name.endswith("_diff"):
            features[name] = 0.0
        else:
            features[name] = 0.0
    return features


def _apply_metric_features(
    features: dict[str, float],
    fighter_value: float,
    opponent_value: float,
    prefix: str,
) -> None:
    """Populate mirrored features and their difference when available."""
    primary = f"{prefix}_a"
    secondary = f"{prefix}_b"
    diff_name = f"{prefix}_diff"
    if primary in features:
        features[primary] = fighter_value
    if secondary in features:
        features[secondary] = opponent_value
    if diff_name in features:
        features[diff_name] = fighter_value - opponent_value


def _build_feature_frame(
    payload: PredictRequest,
    feature_index: pd.Index,
    market_probability: float,
    division: str,
) -> pd.DataFrame:
    """Construct the single-row feature frame expected by the classifier."""
    features = _neutral_feature_vector(feature_index)
    fighter_age = float(payload.fighter_age or _DEFAULT_AGE)
    opponent_age = float(payload.opponent_age or _DEFAULT_AGE)
    fighter_height = float(payload.fighter_height or _DEFAULT_HEIGHT)
    opponent_height = float(payload.opponent_height or _DEFAULT_HEIGHT)
    fighter_reach = float(payload.fighter_reach or _DEFAULT_REACH)
    opponent_reach = float(payload.opponent_reach or _DEFAULT_REACH)

    _apply_metric_features(features, fighter_age, opponent_age, "age")
    _apply_metric_features(features, fighter_height, opponent_height, "height")
    _apply_metric_features(features, fighter_reach, opponent_reach, "reach")

    age_effect_a = age_curve_effect(fighter_age, division)
    age_effect_b = age_curve_effect(opponent_age, division)
    if "age_effect_a" in features:
        features["age_effect_a"] = age_effect_a
    if "age_effect_b" in features:
        features["age_effect_b"] = age_effect_b
    if "age_effect_diff" in features:
        features["age_effect_diff"] = age_effect_a - age_effect_b

    if "market_prob" in features:
        features["market_prob"] = market_probability

    frame = pd.DataFrame([features], columns=feature_index)
    return frame.astype(np.float64, copy=False)


def _load_model_artifacts() -> tuple[object, Calibrator]:
    try:
        classifier = load_artifact("classifier.joblib")
        calibrator = Calibrator.load(MODEL_DIR / "calibrator.joblib")
    except FileNotFoundError as exc:  # pragma: no cover - validated in tests
        raise HTTPException(status_code=503, detail="Model artifacts unavailable") from exc
    return classifier, calibrator


@router.post("/predict", response_model=PredictionResponse)
def predict_single(payload: PredictRequest) -> PredictionResponse:
    """Generate a win probability for the requested fighter."""
    american_odds = payload.american_odds
    market_probability: float | None = None
    if american_odds is not None:
        try:
            market_probability = american_to_implied(american_odds)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    classifier, calibrator = _load_model_artifacts()

    feature_names = getattr(classifier, "feature_names_in_", None)
    if feature_names is None:
        raise HTTPException(status_code=500, detail="Model missing feature metadata")

    division = (payload.division or calibrator.default_division or _DEFAULT_DIVISION).upper()
    base_prob = market_probability if market_probability is not None else 0.5
    feature_index = pd.Index(feature_names)
    frame = _build_feature_frame(payload, feature_index, base_prob, division)

    try:
        probabilities = classifier.predict_proba(frame)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail="Model inference failed") from exc

    raw_probability = float(probabilities[0, 1])
    calibrated = float(calibrator.transform([raw_probability], [division])[0])
    probability_low = float(np.clip(calibrated - 0.05, 0.0, 1.0))
    probability_high = float(np.clip(calibrated + 0.05, 0.0, 1.0))

    return PredictionResponse(
        bout_id=payload.bout_id,
        fighter=payload.fighter,
        probability=calibrated,
        probability_low=probability_low,
        probability_high=probability_high,
        market_probability=market_probability,
    )


__all__ = ["router"]
