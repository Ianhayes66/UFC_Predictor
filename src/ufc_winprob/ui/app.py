"""Streamlit dashboard for UFC win probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ..models.backtest import BACKTEST_PATH
from ..models.predict import PREDICTIONS_PATH
from ..models.selection import rank_recommendations
from ..pipelines.update_upcoming import run as update_upcoming
from .plots import calibration_plot

st.set_page_config(page_title="UFC Win Probabilities", layout="wide")


def load_predictions() -> pd.DataFrame:
    if PREDICTIONS_PATH.exists():
        return pd.read_parquet(PREDICTIONS_PATH)
    return update_upcoming()["predictions"]


def main() -> None:
    st.title("UFC Win Probability Platform")
    if st.button("Refresh Data"):
        update_upcoming()

    if PREDICTIONS_PATH.exists():
        predictions = pd.read_parquet(PREDICTIONS_PATH)
    else:
        update_upcoming()
        predictions = pd.read_parquet(PREDICTIONS_PATH)

    st.subheader("Upcoming Probabilities")
    st.dataframe(predictions[[col for col in predictions.columns if "fighter" in col or "prob" in col]])

    st.subheader("EV Leaderboard")
    leaderboard_path = Path("data/processed/ev_leaderboard.csv")
    if leaderboard_path.exists():
        leaderboard = pd.read_csv(leaderboard_path)
        st.dataframe(leaderboard)
    else:
        st.info("Run make predict to generate leaderboard.")

    st.subheader("Calibration")
    buffer = calibration_plot(predictions["probability"], np.random.binomial(1, predictions["probability"]))
    st.image(buffer)

    st.subheader("Backtest Summary")
    if BACKTEST_PATH.exists():
        st.json(json.loads(Path(BACKTEST_PATH).read_text()))
    else:
        st.info("Run make backtest to produce summary.")


if __name__ == "__main__":
    main()
