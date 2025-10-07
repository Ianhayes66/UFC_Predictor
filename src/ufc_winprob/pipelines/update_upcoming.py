"""Pipeline for updating upcoming fights and odds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data_quality import validate_interim_to_processed, validate_raw_to_interim
from ..ingestion.odds_api_client import OddsAPIClient
from ..ingestion.schedule_upcoming import load_upcoming_cards
from ..models.predict import PREDICTIONS_PATH, predict
from ..models.selection import rank_recommendations
from ..observability import pipeline_run
from ..settings import get_settings

MARKET_PATH = Path("data/processed/market_odds.parquet")
LEADERBOARD_PATH = Path("data/processed/ev_leaderboard.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_market_frame(client: OddsAPIClient, cards: pd.DataFrame) -> pd.DataFrame:
    snapshots: list[dict[str, object]] = []
    for row in cards.itertuples(index=False):
        bout_id = f"{row.event_id}-main"
        for snapshot in client.fetch_odds(bout_id):
            payload = snapshot.model_dump()
            payload["bout_id"] = bout_id
            snapshots.append(payload)
    if not snapshots:
        return pd.DataFrame(columns=["bout_id", "sportsbook", "american_odds"])
    frame = pd.DataFrame(snapshots)
    # ensure timezone aware timestamps survive serialization
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["implied_rank"] = frame["implied_probability"].rank(method="dense")
    frame["normalized_rank"] = frame["normalized_probability"].rank(method="dense")
    frame["shin_rank"] = frame["shin_probability"].rank(method="dense")
    return frame


def run(use_live_odds: bool | None = None) -> dict[str, Path]:
    settings = get_settings()
    with pipeline_run("update_upcoming") as tracker:
        cards_path = Path("data/processed/upcoming_cards.parquet")
        market_frame = pd.DataFrame()
        with tracker.step("ingest_cards"):
            cards = load_upcoming_cards()
            tracker.rows_in(len(cards))
            tracker.rows_in(len(cards), step="ingest_cards")
            validate_raw_to_interim(cards)
            cards_path.parent.mkdir(parents=True, exist_ok=True)
            cards.to_parquet(cards_path, index=False)
            tracker.rows_out(len(cards))
            tracker.rows_out(len(cards), step="ingest_cards")

        with tracker.step("market_odds"):
            client = OddsAPIClient(
                sportsbooks=("MockBook", "SharpBook"),
                use_live_api=use_live_odds if use_live_odds is not None else settings.use_live_odds,
            )
            market_frame = _build_market_frame(client, cards)
            client.close()
            if not market_frame.empty:
                validate_interim_to_processed(market_frame)
                MARKET_PATH.parent.mkdir(parents=True, exist_ok=True)
                market_frame.to_parquet(MARKET_PATH, index=False)
                tracker.rows_out(len(market_frame))
                tracker.rows_out(len(market_frame), step="market_odds")

        with tracker.step("predict"):
            predictions = predict()
            tracker.rows_out(len(predictions))
            tracker.rows_out(len(predictions), step="predict")
        price_map = (
            market_frame.groupby("bout_id")["american_odds"].median()
            if not market_frame.empty
            else pd.Series(dtype=float)
        )
        enriched = predictions.copy()
        if "bout_id" in enriched.columns:
            enriched["american_odds"] = enriched["bout_id"].map(price_map).fillna(120.0)
        else:
            enriched["american_odds"] = 120.0
        with tracker.step("leaderboard"):
            leaderboard = rank_recommendations(enriched)
            leaderboard["stale"] = False
            leaderboard["sportsbook"] = "MockBook"
            leaderboard.to_csv(LEADERBOARD_PATH, index=False)
            leaderboard.to_csv(REPORTS_DIR / "ev_leaderboard.csv", index=False)
            tracker.rows_out(len(leaderboard))
            tracker.rows_out(len(leaderboard), step="leaderboard")

        return {
            "cards": cards_path,
            "predictions": PREDICTIONS_PATH,
            "leaderboard": LEADERBOARD_PATH,
            "market": MARKET_PATH,
        }


if __name__ == "__main__":
    run()
