from __future__ import annotations

import pandas as pd

from ufc_winprob.features.feature_builder import synthetic_dataset


def test_synthetic_dataset_dimensions() -> None:
    matrix = synthetic_dataset(n=32)
    assert isinstance(matrix.features, pd.DataFrame)
    assert len(matrix.features) == len(matrix.target)
    assert "fighter_striking" in matrix.features.columns
