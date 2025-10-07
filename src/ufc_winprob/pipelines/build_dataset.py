"""Dataset construction pipeline."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_quality import validate_interim_to_processed, validate_raw_to_interim
from ..features.feature_builder import synthetic_dataset
from ..ingestion.schedule_upcoming import load_upcoming_cards
from ..observability import pipeline_run


def _prepare_processed_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create a lightweight DataFrame to run processed-level validations."""

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        baseline = pd.Series([0.5] * len(df), name="probability", dtype=float)
    else:
        scaled = df[numeric_cols].mean(axis=1) / (df[numeric_cols].std(axis=1).fillna(1.0) + 1.0)
        baseline = 1 / (1 + np.exp(-scaled))
    processed = pd.DataFrame(
        {
            "implied_probability": baseline.clip(0.01, 0.99),
            "normalized_probability": baseline.clip(0.01, 0.99),
            "shin_probability": baseline.clip(0.01, 0.99),
            "z_shin": 0.0,
        }
    )
    processed["implied_rank"] = processed["implied_probability"].rank(method="dense")
    processed["normalized_rank"] = processed["normalized_probability"].rank(method="dense")
    processed["shin_rank"] = processed["shin_probability"].rank(method="dense")
    return processed


def build(stage: str | None = None) -> None:
    with pipeline_run("build_dataset") as tracker:
        with tracker.step("ingest_upcoming"):
            raw_cards = load_upcoming_cards()
            tracker.rows_in(len(raw_cards))
            tracker.rows_in(len(raw_cards), step="ingest_upcoming")
            validate_raw_to_interim(raw_cards)

        with tracker.step("generate_features"):
            matrix = synthetic_dataset(n=128)
            df = matrix.features.copy()
            if matrix.metadata is not None:
                df = pd.concat(
                    [matrix.metadata.reset_index(drop=True), df.reset_index(drop=True)], axis=1
                )
            df["target"] = matrix.target
            quality_frame = _prepare_processed_frame(df)
            validate_interim_to_processed(quality_frame)
            output = Path("data/processed/training_features.parquet")
            output.parent.mkdir(parents=True, exist_ok=True)
            tracker.rows_out(len(df))
            tracker.rows_out(len(df), step="generate_features")
            df.to_parquet(output, index=False)

        if stage == "features" or stage is None:
            with tracker.step("upcoming_features"):
                upcoming = matrix.features.head(16).copy()
                if matrix.metadata is not None:
                    upcoming = pd.concat(
                        [
                            matrix.metadata.head(16).reset_index(drop=True),
                            upcoming.reset_index(drop=True),
                        ],
                        axis=1,
                    )
                tracker.rows_out(len(upcoming))
                tracker.rows_out(len(upcoming), step="upcoming_features")
                Path("data/processed").mkdir(parents=True, exist_ok=True)
                upcoming.to_parquet("data/processed/upcoming_features.parquet", index=False)


def main() -> None:
    parser = ArgumentParser(description="Build datasets")
    parser.add_argument("--stage", choices=["features"], default=None)
    args = parser.parse_args()
    build(stage=args.stage)


if __name__ == "__main__":
    main()
