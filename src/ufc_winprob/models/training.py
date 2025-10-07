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

from ..evaluation import (
    evaluate_predictions,
    plot_calibration_curve,
    reliability_by_division,
    save_metrics,
)
from ..settings import get_settings
from ..features.age_curve import load_age_model
from ..features.feature_builder import FeatureMatrix, synthetic_dataset
from .calibration import train_calibrator
from .persistence import MODEL_DIR, save_artifact

MODEL_PATH = MODEL_DIR / "classifier.joblib"
CALIBRATOR_PATH = MODEL_DIR / "calibrator.joblib"
METRICS_PATH = Path("data/processed/metrics.csv")
METRICS_DIVISION_PATH = Path("data/processed/metrics_by_division.csv")
CALIBRATION_PLOT_DIR = Path("data/processed/plots")


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
        meta_cols = [col for col in ["bout_id", "event_id", "division"] if col in frame.columns]
        metadata = frame[meta_cols].copy() if meta_cols else None
        features = frame.drop(columns=meta_cols, errors="ignore")
        return FeatureMatrix(features=features, target=target, metadata=metadata)
    return synthetic_dataset()


def train() -> TrainingArtifacts:
    settings = get_settings()
    matrix = load_training_data()
    features = matrix.features
    target = matrix.target
    if matrix.metadata is not None and "division" in matrix.metadata.columns:
        divisions = matrix.metadata["division"].fillna("GLOBAL")
    else:
        divisions = pd.Series(["GLOBAL"] * len(target), index=target.index)
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
    valid_divisions = divisions.iloc[last_valid_idx]
    calibrator = train_calibrator(valid_scores, y_valid, valid_divisions, method=settings.model.calibration)
    calibrated = calibrator.transform(valid_scores, valid_divisions)

    report = evaluate_predictions(y_valid, calibrated)
    save_metrics(report, METRICS_PATH)
    evaluation_frame = pd.DataFrame(
        {
            "division": valid_divisions.values,
            "target": y_valid.values,
            "probability": calibrated,
        }
    )
    reliability = reliability_by_division(
        evaluation_frame, "division", "target", "probability"
    )
    METRICS_DIVISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    reliability.to_csv(METRICS_DIVISION_PATH, index=False)
    for division, group in evaluation_frame.groupby("division"):
        plot_calibration_curve(
            group,
            "probability",
            "target",
            f"Calibration - {division}",
            CALIBRATION_PLOT_DIR / f"calibration_{division}.png",
        )

    unique_divisions = sorted(set(divisions))
    for division in unique_divisions:
        model = load_age_model(division)
        model.plot(CALIBRATION_PLOT_DIR / f"age_curves_{division}.png")

    model_path = save_artifact(clf, MODEL_PATH.name)
    calibrator_path = CALIBRATOR_PATH
    calibrator.save(calibrator_path)

    return TrainingArtifacts(model_path=model_path, calibrator_path=calibrator_path, metrics_path=METRICS_PATH)


if __name__ == "__main__":
    train()
