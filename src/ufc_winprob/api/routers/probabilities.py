"""Probabilities router."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import APIRouter

from ..schemas import PredictionResponse

router = APIRouter(prefix="/probabilities", tags=["probabilities"])


@router.get("/", response_model=List[PredictionResponse])
def probabilities() -> List[PredictionResponse]:
    path = Path("data/processed/upcoming_predictions.parquet")
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return [
        PredictionResponse(
            bout_id=row.get("bout_id", str(idx)),
            fighter=row.get("fighter", f"Fighter {idx}"),
            probability=float(row["probability"]),
            probability_low=float(row["prob_low"]),
            probability_high=float(row["prob_high"]),
            market_probability=float(row.get("market_prob", 0.5)) if not pd.isna(row.get("market_prob")) else None,
        )
        for idx, row in df.iterrows()
    ]


__all__ = ["router"]
