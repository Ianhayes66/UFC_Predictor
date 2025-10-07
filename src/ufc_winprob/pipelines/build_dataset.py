"""Dataset construction pipeline."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from ..features.feature_builder import synthetic_dataset


def build(stage: str | None = None) -> None:
    matrix = synthetic_dataset(n=128)
    df = matrix.features.copy()
    if matrix.metadata is not None:
        df = pd.concat([matrix.metadata.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    df["target"] = matrix.target
    output = Path("data/processed/training_features.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    if stage == "features" or stage is None:
        upcoming = matrix.features.head(16).copy()
        if matrix.metadata is not None:
            upcoming = pd.concat([matrix.metadata.head(16).reset_index(drop=True), upcoming.reset_index(drop=True)], axis=1)
        upcoming.to_parquet("data/processed/upcoming_features.parquet", index=False)


def main() -> None:
    parser = ArgumentParser(description="Build datasets")
    parser.add_argument("--stage", choices=["features"], default=None)
    args = parser.parse_args()
    build(stage=args.stage)


if __name__ == "__main__":
    main()
