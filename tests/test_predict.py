from __future__ import annotations

import pandas as pd

from ufc_winprob.models.predict import predict


def test_predict_outputs_probabilities(trained_model) -> None:
    df = predict()
    assert isinstance(df, pd.DataFrame)
    assert df["probability"].between(0, 1).all()
