"""Batch prediction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..data.schemas import Prediction
from ..features.feature_builder import FeatureMatrix, synthetic_dataset
from .calibration import Calibrator
from .persistence import MODEL_DIR, load_artifact

PREDICTIONS_PATH = Path("data/processed/upcoming_predictions.parquet")


def load_inference_features() -> FeatureMatrix:
    dataset_path = Path("data/processed/upcoming_features.parquet")
    if dataset_path.exists():
        frame = pd.read_parquet(dataset_path)
        target = frame.pop("target") if "target" in frame.columns else pd.Series(dtype=int)
        return FeatureMatrix(features=frame, target=target)
    return synthetic_dataset(n=32)


def predict() -> pd.DataFrame:
    clf = load_artifact("classifier.joblib")
    calibrator_model = load_artifact("calibrator.joblib")
    calibrator = Calibrator(model=calibrator_model)

    matrix = load_inference_features()
    features = matrix.features
    raw_scores = clf.predict_proba(features)[:, 1]
    calibrated = calibrator.transform(raw_scores)

    df = features.copy()
    df["probability"] = calibrated
    df["prob_low"] = np.clip(calibrated - 0.05, 0, 1)
    df["prob_high"] = np.clip(calibrated + 0.05, 0, 1)
    try:
        df.to_parquet(PREDICTIONS_PATH, index=False)
    except ImportError:
        csv_path = PREDICTIONS_PATH.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    predict()
