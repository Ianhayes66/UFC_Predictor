"""Predict endpoint for ad-hoc probability requests."""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas import PredictRequest, PredictionResponse
from ...utils.odds_utils import american_to_implied

router = APIRouter(tags=["probabilities"])


@router.post("/predict", response_model=PredictionResponse)
def predict_single(payload: PredictRequest) -> PredictionResponse:
    base_prob = 0.5
    if payload.american_odds is not None:
        try:
            base_prob = american_to_implied(payload.american_odds)
        except ValueError:
            base_prob = 0.5
    return PredictionResponse(
        bout_id=payload.bout_id,
        fighter=payload.fighter,
        probability=base_prob,
        probability_low=max(base_prob - 0.05, 0.0),
        probability_high=min(base_prob + 0.05, 1.0),
        market_probability=None,
    )


__all__ = ["router"]
