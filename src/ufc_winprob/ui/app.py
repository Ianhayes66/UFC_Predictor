"""Streamlit application for UFC win probability insights."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

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


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_predictions() -> pd.DataFrame:
    frame = _safe_read_parquet(PREDICTIONS_PATH)
    if not frame.empty and "fight_date" in frame:
        frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce")
    return frame


def load_leaderboard() -> pd.DataFrame:
    frame = _safe_read_csv(LEADERBOARD_PATH)
    if frame.empty:
        frame = _safe_read_csv(PROCESSED_LEADERBOARD_PATH)
    return frame


def load_market_data() -> pd.DataFrame:
    frame = _safe_read_parquet(MARKET_PATH)
    if not frame.empty and "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame


def load_calibration_metrics() -> pd.DataFrame:
    frame = _safe_read_csv(CALIBRATION_METRICS_PATH)
    if not frame.empty and "ece" in frame:
        frame = frame.sort_values("ece")
    return frame


def _display_backtest_summary() -> None:
    st.markdown("### Backtest Summary")
    report_path = BACKTEST_REPORT_PATH if BACKTEST_REPORT_PATH.exists() else BACKTEST_PATH
    if not report_path.exists():
        st.info("Backtest report not available yet. Run `make refresh` after training.")
        return
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        st.warning("Backtest report could not be parsed.")
        return
    st.json(payload)


def _display_predictions(predictions: pd.DataFrame) -> None:
    st.markdown("### Upcoming Predictions")
    if predictions.empty:
        st.info("No predictions found. Use the refresh controls to generate upcoming fights.")
        return
    display_cols = [
        col
        for col in predictions.columns
        if any(keyword in col for keyword in ("fight", "bout", "fighter", "prob"))
    ]
    if not display_cols:
        display_cols = predictions.columns.tolist()
    st.dataframe(predictions[display_cols], use_container_width=True)


def _display_leaderboard(leaderboard: pd.DataFrame) -> None:
    st.markdown("### EV Leaderboard")
    if leaderboard.empty:
        st.info("EV leaderboard not available. Generate reports with `make refresh`.")
        return
    st.dataframe(leaderboard, use_container_width=True)


def render_overview(predictions: pd.DataFrame, leaderboard: pd.DataFrame) -> None:
    _display_predictions(predictions)
    _display_leaderboard(leaderboard)
    _display_backtest_summary()


def _extract_event_id(bout_id: str) -> str:
    if not isinstance(bout_id, str):
        return ""
    return bout_id.split("-", 1)[0]


def _pivot_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    try:
        pivot = (
            frame.pivot_table(
                index="timestamp",
                columns="sportsbook",
                values="implied_probability",
                aggfunc="last",
            )
            .sort_index()
        )
        return pivot
    except Exception:
        return pd.DataFrame()


def _bout_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    summary = (
        frame.groupby("sportsbook")
        .agg(
            open_probability=("implied_probability", "first"),
            current_probability=("implied_probability", "last"),
            last_updated=("timestamp", "max"),
        )
        .reset_index()
    )
    summary["delta"] = summary["current_probability"] - summary["open_probability"]
    summary = summary.sort_values("current_probability", ascending=False)
    return summary


def render_line_movement(market_data: pd.DataFrame) -> None:
    st.markdown("### Line Movement")
    if market_data.empty:
        st.info("No market snapshots yet. Refresh with live odds enabled once configured.")
        return

    working = market_data.copy()
    if "event_id" not in working.columns:
        working["event_id"] = working["bout_id"].map(_extract_event_id)

    events = sorted({eid for eid in working["event_id"].dropna().unique() if eid})
    if not events:
        st.info("Events are missing from the current odds data.")
        return

    selected_event = st.selectbox("Event", events)
    event_frame = working[working["event_id"] == selected_event]

    bouts = event_frame["bout_id"].dropna().unique().tolist()
    if not bouts:
        st.info("No bouts found for the selected event.")
        return

    selected_bout = st.selectbox("Bout", bouts)
    bout_frame = event_frame[event_frame["bout_id"] == selected_bout].sort_values("timestamp")

    summary = _bout_summary(bout_frame)
    st.markdown("#### Sportsbook Snapshot")
    if summary.empty:
        st.info("Not enough sportsbook data for this bout yet.")
    else:
        display_cols = [
            "sportsbook",
            "open_probability",
            "current_probability",
            "delta",
            "last_updated",
        ]
        st.dataframe(summary[display_cols], use_container_width=True)

    st.markdown("#### Probability History")
    history = _pivot_probabilities(bout_frame)
    if history.empty or len(history.index.dropna()) < 2:
        st.info("Historical pricing requires multiple snapshots; check back after updates.")
    else:
        st.line_chart(history, use_container_width=True)


def _chunked(iterable: Iterable[Path], size: int) -> list[list[Path]]:
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
    st.markdown("### Calibration by Division")
    if metrics.empty:
        st.info(
            "Calibration metrics missing. Run the training pipeline to generate metrics_by_division.csv."
        )
    else:
        display_cols = [col for col in metrics.columns if col in {"division", "ece", "n_bouts"}]
        st.dataframe(metrics[display_cols], use_container_width=True)

    st.markdown("### Reliability Plots")
    plot_paths = [path for path in sorted(CALIBRATION_PLOTS_DIR.glob("*.png")) if path.exists()]
    if not plot_paths:
        st.info(
            "No calibration plots found under data/processed/plots/. Generate them via the training pipeline."
        )
        return

    for row in _chunked(plot_paths, size=2):
        columns = st.columns(len(row))
        for column, image_path in zip(columns, row, strict=False):
            column.image(str(image_path), caption=image_path.stem.replace("_", " ").title())


def _refresh_data(use_live_odds: bool) -> None:
    try:
        update_upcoming(use_live_odds=use_live_odds)
    except Exception as exc:  # pragma: no cover - defensive guard
        st.sidebar.error(f"Refresh failed: {exc}")


def _render_sidebar(live_available: bool, default_live: bool) -> None:
    st.sidebar.header("Controls")
    if "use_live_odds" not in st.session_state:
        st.session_state["use_live_odds"] = default_live if live_available else False

    if live_available:
        toggle_value = st.sidebar.toggle(
            "Use live odds",
            value=st.session_state["use_live_odds"],
            help="Toggle to fetch the latest prices from the live odds provider.",
        )
        if toggle_value != st.session_state["use_live_odds"]:
            st.session_state["use_live_odds"] = toggle_value
            _refresh_data(use_live_odds=toggle_value)
            st.experimental_rerun()
    else:
        st.sidebar.info("Configure ODDS_API_KEY to enable live odds updates.")

    if st.sidebar.button("Refresh Data"):
        _refresh_data(use_live_odds=st.session_state.get("use_live_odds", False))
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Select a page to explore predictions, market movement, and calibration diagnostics."
    )


def main() -> None:
    _inject_styles()
    st.title("UFC Win Probability Platform")

    settings = get_settings()
    odds_key = os.getenv("ODDS_API_KEY") or getattr(settings, "odds_api_key", "")
    live_available = bool(odds_key)
    default_live = bool(live_available and getattr(settings, "use_live_odds", False))

    _render_sidebar(live_available=live_available, default_live=default_live)
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
