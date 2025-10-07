# UFC Win Probability Platform

```
ufc-winprob/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── pyproject.toml
├── uv.lock
├── Makefile
├── docker-compose.yml
├── Dockerfile
├── .github/workflows/ci.yml
├── .pre-commit-config.yaml
├── config/
│   ├── base.yaml
│   ├── dev.yaml
│   └── prod.yaml
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   ├── external/
│   │   └── .gitkeep
│   └── models/
│       └── .gitkeep
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_checks.ipynb
│   └── 03_model_cards.ipynb
├── src/
│   └── ufc_winprob/
│       ├── __init__.py
│       ├── settings.py
│       ├── logging.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── time_utils.py
│       │   ├── odds_utils.py
│       │   └── metrics.py
│       ├── evaluation.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── schemas.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── ufcstats_scraper.py
│       │   ├── tapology_scraper.py
│       │   ├── odds_api_client.py
│       │   └── schedule_upcoming.py
│       ├── cleaning/
│       │   ├── __init__.py
│       │   ├── normalize_fighters.py
│       │   ├── normalize_bouts.py
│       │   └── feature_coercions.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── style_taxonomy.py
│       │   ├── component_elo.py
│       │   ├── age_curve.py
│       │   ├── elo_updater.py
│       │   └── feature_builder.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── training.py
│       │   ├── predict.py
│       │   ├── calibration.py
│       │   ├── backtest.py
│       │   ├── selection.py
│       │   └── persistence.py
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── build_dataset.py
│       │   ├── update_upcoming.py
│       │   └── daily_refresh.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── schemas.py
│       │   ├── main.py
│       │   └── routers/
│       │       ├── __init__.py
│       │       ├── fights.py
│       │       ├── probabilities.py
│       │       ├── markets.py
│       │       └── recommendations.py
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── plots.py
│       │   └── assets/
│       │       └── styles.css
│       ├── jobs/
│       │   ├── __init__.py
│       │   └── scheduler.py
│       └── scripts/
│           ├── bootstrap.sh
│           ├── download_sample_data.py
│           └── run_all.sh
├── tests/
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_component_elo.py
│   ├── test_age_curves.py
│   ├── test_feature_builder.py
│   ├── test_training.py
│   ├── test_predict.py
│   ├── test_backtest.py
│   ├── test_selection.py
│   └── test_odds_math.py
└── docs/
    ├── SYSTEM_OVERVIEW.md
    ├── DATA_SOURCES.md
    ├── ELO_DESIGN.md
    ├── ODDS_MATH.md
    ├── MODELING_NOTES.md
    ├── API_SPEC.md
    ├── UI_GUIDE.md
    ├── OPERATIONS.md
    ├── MIGRATION_NOTES.md
    ├── SCRAPER_HEALTH.md
    ├── MODEL_CARD.md
    ├── RISK_POLICY.md
    ├── DATA_GOVERNANCE.md
    ├── GLOSSARY.md
    └── CONTRIBUTING.md
```

## Overview

The UFC Win Probability Platform is a production-ready system for ingesting public UFC data, tracking component-wise Elo ratings, blending them with supervised models, and producing calibrated betting opportunities. It is engineered for automation, reproducibility, and observability from day one.

## Quickstart

1. **Install tooling**
   ```bash
   make setup
   ```
2. **Download sample fixtures and bootstrap the environment**
   ```bash
   make data
   ```
3. **Generate features and train the meta-model**
   ```bash
   make features
   make train
   ```
4. **Produce calibrated probabilities and EV reports**
   ```bash
   make predict
   make backtest
   ```
5. **Explore the dashboard**
   ```bash
   make ui
   ```

## API Usage

### `/predict`

The predict endpoint returns a calibrated win probability for an ad-hoc matchup. Provide bout identifiers plus any available fighter attributes to enrich the feature vector.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "bout_id": "ufc-300-main",
        "fighter": "Alex Example",
        "opponent": "Jamie Sample",
        "american_odds": -140,
        "fighter_age": 31,
        "opponent_age": 29,
        "fighter_height": 73,
        "opponent_height": 71,
        "fighter_reach": 75,
        "opponent_reach": 72
      }'
```

Example response:

```json
{
  "bout_id": "ufc-300-main",
  "fighter": "Alex Example",
  "probability": 0.62,
  "probability_low": 0.57,
  "probability_high": 0.67,
  "market_probability": 0.58
}
```

If no odds are supplied the `market_probability` field is `null` and the model infers a neutral prior.

## Architecture

```
                    ┌────────────────────┐
                    │ External Providers │
                    └─────────┬──────────┘
                              │
              ┌───────────────▼────────────────┐
              │ Ingestion & Normalization Layer │
              └───────────────┬────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Feature Pipeline  │
                    └─────────┬─────────┘
                              │
                 ┌────────────▼────────────┐
                 │ Component Elo + Models │
                 └────────────┬────────────┘
                              │
           ┌──────────────────▼───────────────────┐
           │ Calibration · Backtest · Selection   │
           └──────────────────┬───────────────────┘
                              │
           ┌──────────────────▼───────────────────┐
           │ API · Streamlit UI · Scheduled Jobs  │
           └──────────────────────────────────────┘
```

## Key Features

- Component-wise, style-aware Elo ratings blended with division-aware supervised models.
- Deterministic pipelines with caching, retries, and graceful degradation.
- Automated scheduled jobs and comprehensive CI/CD with linting, tests, and coverage.
- FastAPI for programmatic access and Streamlit UI for analysts.
- Observability via structured logging, Prometheus metrics, and reporting artifacts.

## Dashboard

Launch the Streamlit UI with `make ui`. The sidebar includes a **Use live odds** toggle that is enabled when `ODDS_API_KEY` is configured; toggling it triggers a refresh of upcoming fights using the live provider. The navigation exposes **Overview**, **Line Movement**, and **Calibration** pages:

- **Overview** displays the latest probabilities, EV leaderboard, calibration snapshot, and backtest summary.
- **Line Movement** lets you select an event and bout to compare opening versus current implied probabilities by sportsbook, plus a time-series plot of movement.
- **Calibration** surfaces per-division reliability tables and saved calibration plots, and guides you to run `make train` when assets are missing.

## Reports & Metrics

- Run `make refresh` to execute the full pipeline. It produces `reports/ev_leaderboard.csv` and `reports/backtest_summary.json` alongside processed artifacts.
- The FastAPI service exposes Prometheus metrics at `GET /metrics`, including request counts, latency histograms, and pipeline instrumentation counters.

## Command Cheatsheet

| Command | Description |
| --- | --- |
| `make setup` | Install dependencies, configure pre-commit hooks, and prepare the environment. |
| `make data` | Download sample data and run lightweight ingestion checks. |
| `make features` | Build feature matrices using component Elo and contextual attributes. |
| `make train` | Train models with time-aware splits, calibration, and persist artifacts. |
| `make predict` | Produce probabilities, market comparisons, and EV leaderboards. |
| `make backtest` | Run strategy simulations, compute ROI, drawdown, and reports. |
| `make api` | Launch the FastAPI service. |
| `make ui` | Launch the Streamlit dashboard. |
| `make refresh` | Trigger the daily refresh pipeline, updating odds and writing `reports/ev_leaderboard.csv` and `reports/backtest_summary.json`. |
| `ufc reports` | Preview the published EV leaderboard and backtest reports in the terminal. |
| `make demo` | Run an end-to-end demo on synthetic data. |
| `make check` | Run linting (ruff, black check, mypy) and pytest with coverage. |

## Data Ethics & Governance

The project only stores public, regulation-permitted information. Scrapers respect robots.txt, implement rate limiting, and cache snapshots to avoid repeated load. Secrets are managed via environment variables and `.env` files (never committed). See [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md) and [docs/RISK_POLICY.md](docs/RISK_POLICY.md) for details.

## Contributing

Please review [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for coding standards, review expectations, and the preferred workflow. All contributions must pass CI and include documentation/tests.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
