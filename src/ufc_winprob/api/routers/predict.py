"""Predict endpoint for ad-hoc probability requests."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from ufc_winprob.api.schemas import PredictionResponse, PredictRequest
from ufc_winprob.features.age_curve import age_curve_effect
from ufc_winprob.models.calibration import Calibrator
from ufc_winprob.models.persistence import MODEL_DIR, load_artifact
from ufc_winprob.utils.odds_utils import american_to_implied

router = APIRouter(tags=["probabilities"])


def _base_feature_vector(feature_names: pd.Index) -> dict[str, float]:
    """Construct a neutral feature vector compatible with the trained model."""
    base: dict[str, float] = dict.fromkeys(feature_names, 0.0)
    for name in feature_names:
        if name.startswith("fighter_") or name.startswith("opponent_"):
            base[name] = 1500.0
        elif name == "scheduled_rounds":
            base[name] = 3.0
        elif name == "is_main_event":
            base[name] = 0.0
        elif name == "activity_gap":
            base[name] = 60.0
        elif name == "market_prob":
            base[name] = 0.5
    return base


def _build_feature_frame(
    payload: PredictRequest, feature_names: pd.Index, market_prob: float
) -> pd.DataFrame:
    """Populate a single-row feature frame for inference."""
    features = _base_feature_vector(feature_names)
    fighter_age = payload.fighter_age or 30.0
    opponent_age = payload.opponent_age or 30.0
    fighter_height = payload.fighter_height or 72.0
    opponent_height = payload.opponent_height or 72.0
    fighter_reach = payload.fighter_reach or 72.0
    opponent_reach = payload.opponent_reach or 72.0

    def assign(name: str, value: float) -> None:
        if name in features:
            features[name] = float(value)

    assign("age_a", fighter_age)
    assign("age_b", opponent_age)
    assign("age_diff", fighter_age - opponent_age)
    assign("age_effect_a", age_curve_effect(fighter_age, "LW"))
    assign("age_effect_b", age_curve_effect(opponent_age, "LW"))
    assign("height_a", fighter_height)
    assign("height_b", opponent_height)
    assign("height_diff", fighter_height - opponent_height)
    assign("reach_a", fighter_reach)
    assign("reach_b", opponent_reach)
    assign("reach_diff", fighter_reach - opponent_reach)
    assign("market_prob", market_prob)

    return pd.DataFrame([features], columns=feature_names)


@router.post("/predict", response_model=PredictionResponse)
def predict_single(payload: PredictRequest) -> PredictionResponse:
    market_probability = None
    if payload.american_odds is not None:
        try:
            market_probability = american_to_implied(payload.american_odds)
        except ValueError:
            market_probability = None

    try:
        classifier = load_artifact("classifier.joblib")
        calibrator = Calibrator.load(MODEL_DIR / "calibrator.joblib")
    except FileNotFoundError as exc:  # pragma: no cover - guarded by tests
        raise HTTPException(status_code=503, detail="Model artifacts unavailable") from exc

    feature_names = getattr(classifier, "feature_names_in_", None)
    if feature_names is None:
        raise HTTPException(status_code=500, detail="Model is missing feature metadata")

    base_prob = market_probability if market_probability is not None else 0.5
    feature_index = pd.Index(feature_names)
    frame = _build_feature_frame(payload, feature_index, base_prob)

    try:
        raw_score = float(classifier.predict_proba(frame)[0, 1])
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail="Model inference failed") from exc

    calibrated = float(calibrator.transform([raw_score], ["GLOBAL"])[0])
    probability_low = max(calibrated - 0.05, 0.0)
    probability_high = min(calibrated + 0.05, 1.0)

    return PredictionResponse(
        bout_id=payload.bout_id,
        fighter=payload.fighter,
        probability=calibrated,
        probability_low=probability_low,
        probability_high=probability_high,
        market_probability=market_probability,
    )


__all__ = ["router"]
