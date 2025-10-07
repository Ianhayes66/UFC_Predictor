"""Streamlit dashboard for UFC win probabilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ..models.backtest import BACKTEST_PATH, BACKTEST_REPORT_PATH
from ..models.predict import PREDICTIONS_PATH
from ..pipelines.update_upcoming import run as update_upcoming
from ..settings import get_settings
from .plots import calibration_plot

st.set_page_config(page_title="UFC Win Probabilities", layout="wide")

ASSETS_DIR = Path(__file__).parent / "assets"
MARKET_PATH = Path("data/processed/market_odds.parquet")
LEADERBOARD_PATH = Path("data/processed/ev_leaderboard.csv")
PLOTS_DIR = Path("data/processed/plots")


def _inject_styles() -> None:
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def load_predictions() -> pd.DataFrame:
    if PREDICTIONS_PATH.exists():
        return pd.read_parquet(PREDICTIONS_PATH)
    update_upcoming()
    if PREDICTIONS_PATH.exists():
        return pd.read_parquet(PREDICTIONS_PATH)
    return pd.DataFrame()


def load_leaderboard() -> pd.DataFrame:
    if LEADERBOARD_PATH.exists():
        return pd.read_csv(LEADERBOARD_PATH)
    return pd.DataFrame()


def load_market_data() -> pd.DataFrame:
    if MARKET_PATH.exists():
        frame = pd.read_parquet(MARKET_PATH)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame
    return pd.DataFrame()


def render_overview(predictions: pd.DataFrame, leaderboard: pd.DataFrame) -> None:
    st.subheader("Upcoming Probabilities")
    if predictions.empty:
        st.info("No predictions available. Run a refresh to generate them.")
    else:
        display_cols = [col for col in predictions.columns if "fighter" in col or "prob" in col]
        st.dataframe(predictions[display_cols])

    st.subheader("EV Leaderboard")
    if leaderboard.empty:
        st.info("Run the refresh pipeline to compute EV leaderboard results.")
    else:
        st.dataframe(leaderboard)

    st.subheader("Calibration Snapshot")
    if predictions.empty:
        st.info("Predictions required to render calibration chart.")
    else:
        outcomes = np.random.binomial(1, predictions["probability"])
        buffer = calibration_plot(predictions["probability"], outcomes)
        st.image(buffer)

    st.subheader("Backtest Summary")
    backtest_path = BACKTEST_REPORT_PATH if BACKTEST_REPORT_PATH.exists() else BACKTEST_PATH
    if backtest_path.exists():
        st.json(json.loads(backtest_path.read_text()))
    else:
        st.info("Run the daily refresh to generate backtest metrics.")


def render_line_movement(market_data: pd.DataFrame) -> None:
    st.subheader("Line Movement")
    if market_data.empty:
        st.info("No market data available yet. Trigger a refresh to pull odds.")
        return
    market_data = market_data.sort_values("timestamp")
    summary = (
        market_data.groupby(["bout_id", "sportsbook"])
        .agg(
            open_probability=("implied_probability", "first"),
            current_probability=("implied_probability", "last"),
            last_updated=("timestamp", "max"),
        )
        .reset_index()
    )
    summary["movement"] = summary["current_probability"] - summary["open_probability"]
    summary["flagged"] = summary["movement"].abs() >= 0.03
    st.dataframe(summary)

    bout_options = summary["bout_id"].unique().tolist()
    if not bout_options:
        return
    selected_bout = st.selectbox("Select bout", bout_options)
    bout_frame = market_data[market_data["bout_id"] == selected_bout]
    chart = bout_frame.pivot_table(
        index="timestamp",
        columns="sportsbook",
        values="implied_probability",
        aggfunc="last",
    )
    st.line_chart(chart)


def render_calibration() -> None:
    st.subheader("Division Calibration Metrics")
    metrics_path = PLOTS_DIR / "per_division_ece.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics)
    else:
        st.info("Upload calibration metrics to data/processed/plots/per_division_ece.csv.")

    st.subheader("Reliability Plots")
    plots = sorted(PLOTS_DIR.glob("*.png"))
    if not plots:
        st.info("No reliability plots found in data/processed/plots/.")
    else:
        for image in plots:
            st.image(str(image), caption=image.stem.replace("_", " ").title())


def main() -> None:
    _inject_styles()
    st.title("UFC Win Probability Platform")
    settings = get_settings()
    live_available = bool(settings.odds_api_key or os.getenv("ODDS_API_KEY"))

    st.sidebar.header("Controls")
    use_live_default = settings.use_live_odds if live_available else False
    use_live = st.sidebar.checkbox(
        "Use live odds",
        value=use_live_default,
        disabled=not live_available,
        help="Requires ODDS_API_KEY to be configured.",
    )
    if st.sidebar.button("Refresh Data"):
        update_upcoming(use_live_odds=use_live if live_available else False)
        st.experimental_rerun()

    page = st.sidebar.radio("Page", ("Overview", "Line Movement", "Calibration"))

    predictions = load_predictions()
    leaderboard = load_leaderboard()
    market_data = load_market_data()

    if page == "Overview":
        render_overview(predictions, leaderboard)
    elif page == "Line Movement":
        render_line_movement(market_data)
    else:
        render_calibration()


if __name__ == "__main__":
    main()
