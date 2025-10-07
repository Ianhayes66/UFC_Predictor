"""Model training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from ..evaluation import evaluate_predictions, save_metrics
from ..features.feature_builder import FeatureMatrix, synthetic_dataset
from .calibration import train_calibrator
from .persistence import MODEL_DIR, save_artifact

MODEL_PATH = MODEL_DIR / "classifier.joblib"
CALIBRATOR_PATH = MODEL_DIR / "calibrator.joblib"
METRICS_PATH = Path("data/processed/metrics.csv")


@dataclass
class TrainingArtifacts:
    model_path: Path
    calibrator_path: Path
    metrics_path: Path


def load_training_data() -> FeatureMatrix:
    dataset_path = Path("data/processed/training_features.parquet")
    if dataset_path.exists():
        frame = pd.read_parquet(dataset_path)
        target = frame.pop("target")
        return FeatureMatrix(features=frame, target=target)
    return synthetic_dataset()


def train() -> TrainingArtifacts:
    matrix = load_training_data()
    features = matrix.features
    target = matrix.target
    splitter = TimeSeriesSplit(n_splits=3)
    last_train_idx, last_valid_idx = None, None
    for train_index, valid_index in splitter.split(features):
        last_train_idx, last_valid_idx = train_index, valid_index
    if last_train_idx is None or last_valid_idx is None:
        raise RuntimeError("Insufficient data for time-series split")
    x_train, x_valid = features.iloc[last_train_idx], features.iloc[last_valid_idx]
    y_train, y_valid = target.iloc[last_train_idx], target.iloc[last_valid_idx]

    clf = LogisticRegression(max_iter=200, n_jobs=1)
    clf.fit(x_train, y_train)

    valid_scores = clf.predict_proba(x_valid)[:, 1]
    calibrator = train_calibrator(valid_scores, y_valid)
    calibrated = calibrator.transform(valid_scores)

    report = evaluate_predictions(y_valid, calibrated)
    save_metrics(report, METRICS_PATH)

    model_path = save_artifact(clf, MODEL_PATH.name)
    calibrator_path = save_artifact(calibrator.model, CALIBRATOR_PATH.name)

    return TrainingArtifacts(model_path=model_path, calibrator_path=calibrator_path, metrics_path=METRICS_PATH)


if __name__ == "__main__":
    train()
