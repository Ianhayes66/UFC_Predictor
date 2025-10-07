"""Evaluation helpers for models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score

from .utils.metrics import MetricResult, brier_score, expected_calibration_error, reliability_bins


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
    clipped = np.clip(prob, 1e-9, 1 - 1e-9)
    return EvaluationReport(
        auc=float(roc_auc_score(true, clipped)),
        logloss=float(log_loss(true, clipped)),
        brier=brier_score(true, clipped),
        ece=expected_calibration_error(true, clipped),
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


def reliability_by_division(
    frame: pd.DataFrame,
    division_col: str,
    target_col: str,
    prob_col: str,
    bins: int = 20,
    bootstrap_samples: int = 200,
) -> pd.DataFrame:
    rows: List[dict[str, float]] = []
    rng = np.random.default_rng(42)
    for division, group in frame.groupby(division_col):
        probabilities = group[prob_col].to_numpy(dtype=float)
        outcomes = group[target_col].to_numpy(dtype=int)
        ece = expected_calibration_error(outcomes, probabilities, bins=bins)
        boot = []
        if len(group) > 1:
            for _ in range(bootstrap_samples):
                indices = rng.integers(0, len(group), size=len(group))
                boot.append(
                    expected_calibration_error(outcomes[indices], probabilities[indices], bins=bins)
                )
        ci_lower = float(np.quantile(boot, 0.05)) if boot else ece
        ci_upper = float(np.quantile(boot, 0.95)) if boot else ece
        rows.append(
            {
                division_col: division,
                "ece": float(ece),
                "ece_lower": ci_lower,
                "ece_upper": ci_upper,
                "count": float(len(group)),
            }
        )
    return pd.DataFrame(rows)


def plot_calibration_curve(group: pd.DataFrame, prob_col: str, target_col: str, title: str, path: Path) -> Path:
    prob_mean, acc_mean = reliability_bins(group[target_col], group[prob_col])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect")
    ax.plot(prob_mean, acc_mean, marker="o", label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


__all__ = [
    "EvaluationReport",
    "evaluate_predictions",
    "save_metrics",
    "per_division_metrics",
    "reliability_by_division",
    "plot_calibration_curve",
]
