"""Typer CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from .models import backtest, predict, training
from .pipelines import build_dataset, daily_refresh, update_upcoming

app = typer.Typer(help="Command-line interface for UFC Win Probability platform")


@app.command()
def data(stage: str | None = None) -> None:
    """Build datasets."""

    build_dataset.build(stage=stage)


@app.command()
def features() -> None:
    """Alias for building features."""

    build_dataset.build(stage="features")


@app.command()
def train() -> None:
    """Train models."""

    training.train()


@app.command()
def predict_cmd() -> None:
    """Generate predictions."""

    predict.predict()


@app.command()
def backtest_cmd() -> None:
    """Run backtest."""

    backtest.backtest()


@app.command()
def refresh(use_live_odds: bool = typer.Option(False, help="Pull live odds when true.")) -> None:
    """Run daily refresh pipeline."""

    daily_refresh.daily_refresh(use_live_odds=use_live_odds)


@app.command()
def upcoming(use_live_odds: bool = typer.Option(False, help="Poll live odds sources.")) -> None:
    """Update upcoming fights and EV leaderboard."""

    update_upcoming.run(use_live_odds=use_live_odds)


@app.command()
def reports(limit: int = typer.Option(5, help="Number of leaderboard rows to display.")) -> None:
    """Preview generated reports from the refresh workflow."""

    leaderboard_path = Path("reports/ev_leaderboard.csv")
    backtest_path = Path("reports/backtest_summary.json")

    if leaderboard_path.exists():
        frame = pd.read_csv(leaderboard_path).head(limit)
        typer.secho("EV Leaderboard:", fg=typer.colors.CYAN)
        typer.echo(frame.to_string(index=False))
    else:
        typer.secho(
            "Leaderboard report not found. Run `make refresh` first.", fg=typer.colors.YELLOW
        )

    if backtest_path.exists():
        payload = json.loads(backtest_path.read_text(encoding="utf-8"))
        typer.secho("\nBacktest Summary:", fg=typer.colors.CYAN)
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.secho("Backtest report not found. Run `make refresh` first.", fg=typer.colors.YELLOW)


if __name__ == "__main__":
    app()
