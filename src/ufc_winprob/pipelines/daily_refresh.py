"""Daily refresh orchestration."""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Dict

from loguru import logger

from ..models.backtest import BACKTEST_PATH, backtest
from ..models.predict import predict
from ..models.training import train
from ..observability import pipeline_run
from .build_dataset import build
from .update_upcoming import run


def daily_refresh(use_live_odds: bool | None = None) -> Dict[str, str]:
    logger.info("Starting daily refresh (live_odds=%s)", use_live_odds)
    with pipeline_run("daily_refresh") as tracker:
        with tracker.step("build_datasets"):
            build()

        with tracker.step("train_models"):
            artifacts = train()

        with tracker.step("predict_upcoming"):
            predictions = predict()
            tracker.rows_out(len(predictions))
            tracker.rows_out(len(predictions), step="predict_upcoming")

        with tracker.step("update_upcoming"):
            outputs = run(use_live_odds=use_live_odds)

        with tracker.step("backtest"):
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
