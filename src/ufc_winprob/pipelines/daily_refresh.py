"""Daily refresh orchestration."""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Dict

from loguru import logger

from ..models.predict import predict
from ..models.training import train
from .build_dataset import build
from .update_upcoming import run


def daily_refresh(mock: bool = False) -> Dict[str, str]:
    logger.info("Starting daily refresh (mock=%s)", mock)
    build()
    artifacts = train()
    predictions = predict()
    outputs = run()
    return {
        "model": str(artifacts.model_path),
        "calibrator": str(artifacts.calibrator_path),
        "metrics": str(artifacts.metrics_path),
        "predictions": outputs["predictions"].as_posix(),
        "leaderboard": outputs["leaderboard"].as_posix(),
    }


def main() -> None:
    parser = ArgumentParser(description="Run daily refresh pipeline")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    daily_refresh(mock=args.mock)


if __name__ == "__main__":
    main()
