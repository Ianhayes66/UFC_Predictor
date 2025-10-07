"""Daily refresh orchestration."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import pandas as pd
from loguru import logger

from ..models.backtest import BACKTEST_PATH, backtest
from ..models.predict import PREDICTIONS_PATH, predict
from ..models.training import train
from ..observability import pipeline_run
from .build_dataset import build
from .update_upcoming import run


TRAINING_FEATURES_PATH = Path("data/processed/training_features.parquet")
UPCOMING_FEATURES_PATH = Path("data/processed/upcoming_features.parquet")
METRICS_BY_DIVISION_PATH = Path("data/processed/metrics_by_division.csv")


def daily_refresh(use_live_odds: bool | None = None) -> Dict[str, str]:
    logger.info("Starting daily refresh (live_odds=%s)", use_live_odds)
    with pipeline_run("daily_refresh") as tracker:
        with tracker.step("build_datasets"):
            build()
            if TRAINING_FEATURES_PATH.exists():
                training_rows = len(pd.read_parquet(TRAINING_FEATURES_PATH))
                tracker.rows_out(training_rows)
                tracker.rows_out(training_rows, step="build_datasets")

        with tracker.step("train_models"):
            if TRAINING_FEATURES_PATH.exists():
                feature_rows = len(pd.read_parquet(TRAINING_FEATURES_PATH))
                tracker.rows_in(feature_rows)
                tracker.rows_in(feature_rows, step="train_models")
            artifacts = train()
            if METRICS_BY_DIVISION_PATH.exists():
                metrics_rows = len(pd.read_csv(METRICS_BY_DIVISION_PATH))
                tracker.rows_out(metrics_rows)
                tracker.rows_out(metrics_rows, step="train_models")

        with tracker.step("predict_upcoming"):
            if UPCOMING_FEATURES_PATH.exists():
                feature_rows = len(pd.read_parquet(UPCOMING_FEATURES_PATH))
                tracker.rows_in(feature_rows)
                tracker.rows_in(feature_rows, step="predict_upcoming")
            predictions = predict()
            tracker.rows_out(len(predictions))
            tracker.rows_out(len(predictions), step="predict_upcoming")

        with tracker.step("update_upcoming"):
            if PREDICTIONS_PATH.exists():
                prediction_rows = len(pd.read_parquet(PREDICTIONS_PATH))
                tracker.rows_in(prediction_rows)
                tracker.rows_in(prediction_rows, step="update_upcoming")
            outputs = run(use_live_odds=use_live_odds)
            leaderboard_path = outputs.get("leaderboard")
            if leaderboard_path and leaderboard_path.exists():
                leaderboard_rows = len(pd.read_csv(leaderboard_path))
                tracker.rows_out(leaderboard_rows)
                tracker.rows_out(leaderboard_rows, step="update_upcoming")

        with tracker.step("backtest"):
            if PREDICTIONS_PATH.exists():
                prediction_rows = len(pd.read_parquet(PREDICTIONS_PATH))
                tracker.rows_in(prediction_rows)
                tracker.rows_in(prediction_rows, step="backtest")
            result = backtest()
            tracker.rows_out(result.bets)
            tracker.rows_out(result.bets, step="backtest")

    return {
        "model": str(artifacts.model_path),
        "calibrator": str(artifacts.calibrator_path),
        "metrics": str(artifacts.metrics_path),
        "predictions": outputs["predictions"].as_posix(),
        "leaderboard": outputs["leaderboard"].as_posix(),
        "market": outputs["market"].as_posix(),
        "backtest": BACKTEST_PATH.as_posix(),
    }


def main() -> None:
    parser = ArgumentParser(description="Run daily refresh pipeline")
    parser.add_argument(
        "--use-live-odds",
        action="store_true",
        help="Pull live odds from providers when set.",
    )
    args = parser.parse_args()
    daily_refresh(use_live_odds=args.use_live_odds)


if __name__ == "__main__":
    main()
