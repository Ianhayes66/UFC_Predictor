"""Streamlit application for UFC win probability insights."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import streamlit as st

from ufc_winprob.models.backtest import BACKTEST_PATH, BACKTEST_REPORT_PATH
from ufc_winprob.models.predict import PREDICTIONS_PATH
from ufc_winprob.pipelines.update_upcoming import run as update_upcoming
from ufc_winprob.settings import get_settings

st.set_page_config(page_title="UFC Win Probabilities", layout="wide")

ASSETS_DIR = Path(__file__).parent / "assets"
MARKET_PATH = Path("data/processed/market_odds.parquet")
LEADERBOARD_PATH = Path("reports/ev_leaderboard.csv")
PROCESSED_LEADERBOARD_PATH = Path("data/processed/ev_leaderboard.csv")
CALIBRATION_METRICS_PATH = Path("data/processed/metrics_by_division.csv")
CALIBRATION_PLOTS_DIR = Path("data/processed/plots")


def _inject_styles() -> None:
    """Inject optional CSS overrides into the Streamlit app."""
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def _load_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_predictions() -> pd.DataFrame:
    """Load cached predictions if present."""
    if PREDICTIONS_PATH.exists():
        return pd.read_parquet(PREDICTIONS_PATH)
    return pd.DataFrame()


def load_leaderboard() -> pd.DataFrame:
    """Load the EV leaderboard from either reports or processed data."""
    if LEADERBOARD_PATH.exists():
        return pd.read_csv(LEADERBOARD_PATH)
    if PROCESSED_LEADERBOARD_PATH.exists():
        return pd.read_csv(PROCESSED_LEADERBOARD_PATH)
    return pd.DataFrame()


def load_market_data() -> pd.DataFrame:
    """Load the latest market snapshot."""
    frame = _load_parquet(MARKET_PATH)
    if not frame.empty and "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def load_calibration_metrics() -> pd.DataFrame:
    """Load per-division calibration metrics if generated."""
    if CALIBRATION_METRICS_PATH.exists():
        return pd.read_csv(CALIBRATION_METRICS_PATH)
    return pd.DataFrame()


def _display_backtest_summary() -> None:
    """Render the backtest report if it has been produced."""
    st.subheader("Backtest Summary")
    report_path = BACKTEST_REPORT_PATH if BACKTEST_REPORT_PATH.exists() else BACKTEST_PATH
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        st.json(payload)
    else:
        st.info("Backtest report not available. Run `make refresh` to generate it.")


def render_overview(predictions: pd.DataFrame, leaderboard: pd.DataFrame) -> None:
    """Render the overview dashboard combining probabilities and reports."""
    st.subheader("Upcoming Probabilities")
    if predictions.empty:
        st.info("No predictions available. Use the refresh controls to generate data.")
    else:
        display_cols = [
            col
            for col in predictions.columns
            if any(keyword in col for keyword in ("bout", "fighter", "probability"))
        ]
        st.dataframe(predictions[display_cols], use_container_width=True)

    st.subheader("EV Leaderboard")
    if leaderboard.empty:
        st.info("No EV leaderboard found. Run `make refresh` to publish reports.")
    else:
        st.dataframe(leaderboard, use_container_width=True)

    _display_backtest_summary()


def _extract_events(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "event_id" not in working.columns:
        working["event_id"] = working["bout_id"].str.split("-", n=1).str[0]
    return working


def _pivot_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index="timestamp",
        columns="sportsbook",
        values="implied_probability",
        aggfunc="last",
    ).sort_index()
    return pivot


def render_line_movement(market_data: pd.DataFrame) -> None:
    """Visualize line movement for the selected event and bout."""
    st.subheader("Line Movement")
    if market_data.empty:
        st.info("No market data is available yet. Trigger a refresh to pull odds.")
        return

    market_data = _extract_events(market_data)
    event_ids = market_data["event_id"].dropna().unique().tolist()
    if not event_ids:
        st.info("No events found in the current odds snapshot.")
        return

    selected_event = st.selectbox("Select event", event_ids)
    event_frame = market_data[market_data["event_id"] == selected_event]
    bout_ids = event_frame["bout_id"].unique().tolist()
    if not bout_ids:
        st.info("No bouts found for the selected event.")
        return

    selected_bout = st.selectbox("Select bout", bout_ids)
    bout_frame = event_frame[event_frame["bout_id"] == selected_bout].sort_values("timestamp")
    summary = (
        bout_frame.groupby("sportsbook")
        .agg(
            open_probability=("implied_probability", "first"),
            current_probability=("implied_probability", "last"),
            last_updated=("timestamp", "max"),
        )
        .reset_index()
    )
    summary["delta"] = summary["current_probability"] - summary["open_probability"]
    summary = summary.sort_values("current_probability", ascending=False)

    st.markdown("### Sportsbook Snapshot")
    st.dataframe(
        summary[["sportsbook", "open_probability", "current_probability", "delta", "last_updated"]],
        use_container_width=True,
    )

    st.markdown("### Probability History")
    history = _pivot_probabilities(bout_frame)
    if history.empty or history.shape[0] < 2:
        st.info("Historical line movement is unavailable for this bout yet.")
    else:
        st.line_chart(history, use_container_width=True)


def _chunked(iterable: Iterable[Path], size: int) -> list[list[Path]]:
    """Split an iterable of paths into rows for display."""
    chunk: list[Path] = []
    grid: list[list[Path]] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            grid.append(chunk)
            chunk = []
    if chunk:
        grid.append(chunk)
    return grid


def render_calibration(metrics: pd.DataFrame) -> None:
    """Render calibration metrics table and associated plots."""
    st.subheader("Division Calibration Metrics")
    if metrics.empty:
        st.info("Calibration metrics not found. Run `make train` to generate them.")
    else:
        metrics_display = metrics.sort_values("ece")
        st.dataframe(metrics_display, use_container_width=True)

    st.subheader("Reliability Plots")
    plot_paths = sorted(CALIBRATION_PLOTS_DIR.glob("calibration_*.png"))
    if not plot_paths:
        st.info("Calibration plots missing. Run `make train` to create reliability visuals.")
        return

    for row in _chunked(plot_paths, size=2):
        columns = st.columns(len(row))
        for col, image_path in zip(columns, row, strict=False):
            col.image(str(image_path), caption=image_path.stem.replace("_", " ").title())


def _refresh_data(use_live_odds: bool) -> None:
    """Trigger the upcoming pipeline with the requested odds source."""
    update_upcoming(use_live_odds=use_live_odds)


def main() -> None:
    """Entrypoint for the Streamlit dashboard."""
    _inject_styles()
    st.title("UFC Win Probability Platform")

    settings = get_settings()
    odds_key = os.getenv("ODDS_API_KEY") or settings.odds_api_key
    live_available = bool(odds_key)

    st.sidebar.header("Controls")
    if "use_live_odds" not in st.session_state:
        st.session_state["use_live_odds"] = bool(live_available and settings.use_live_odds)

    toggle_value = st.sidebar.toggle(
        "Use live odds",
        value=st.session_state["use_live_odds"],
        disabled=not live_available,
        help="Requires ODDS_API_KEY to be configured in the environment.",
    )
    if toggle_value != st.session_state["use_live_odds"]:
        st.session_state["use_live_odds"] = toggle_value
        _refresh_data(use_live_odds=toggle_value)
        st.experimental_rerun()

    if st.sidebar.button("Refresh Data"):
        _refresh_data(use_live_odds=st.session_state["use_live_odds"])
        st.experimental_rerun()

    page = st.sidebar.radio("Page", ("Overview", "Line Movement", "Calibration"))

    predictions = load_predictions()
    leaderboard = load_leaderboard()
    market_data = load_market_data()
    calibration_metrics = load_calibration_metrics()

    if page == "Overview":
        render_overview(predictions, leaderboard)
    elif page == "Line Movement":
        render_line_movement(market_data)
    else:
        render_calibration(calibration_metrics)


if __name__ == "__main__":
    main()
