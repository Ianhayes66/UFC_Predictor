"""Evaluation helpers for models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .utils.metrics import MetricResult, brier_score, expected_calibration_error


@dataclass
class EvaluationReport:
    """Container for evaluation metrics."""

    auc: float
    logloss: float
    brier: float
    ece: float

    def as_dict(self) -> dict[str, float]:
        return {
            "auc": self.auc,
            "logloss": self.logloss,
            "brier": self.brier,
            "ece": self.ece,
        }


def evaluate_predictions(y_true: Iterable[int], y_prob: Iterable[float]) -> EvaluationReport:
    true = np.asarray(list(y_true), dtype=int)
    prob = np.asarray(list(y_prob), dtype=float)
    return EvaluationReport(
        auc=float(roc_auc_score(true, prob)),
        logloss=float(log_loss(true, prob, eps=1e-9)),
        brier=brier_score(true, prob),
        ece=expected_calibration_error(true, prob),
    )


def save_metrics(report: EvaluationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([report.as_dict()])
    df.to_csv(path, index=False)


def per_division_metrics(frame: pd.DataFrame, division_col: str, target_col: str, prob_col: str) -> pd.DataFrame:
    records: List[dict[str, float]] = []
    for division, group in frame.groupby(division_col):
        report = evaluate_predictions(group[target_col], group[prob_col])
        row = report.as_dict()
        row[division_col] = division
        records.append(row)
    return pd.DataFrame(records)


__all__ = ["EvaluationReport", "evaluate_predictions", "save_metrics", "per_division_metrics"]
