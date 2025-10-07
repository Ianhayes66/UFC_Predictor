"""Generate synthetic sample data for offline development."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..features.feature_builder import synthetic_dataset


def main() -> None:
    matrix = synthetic_dataset()
    frame = matrix.features.copy()
    frame["target"] = matrix.target
    path = Path("data/external/sample_training.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


if __name__ == "__main__":
    main()
