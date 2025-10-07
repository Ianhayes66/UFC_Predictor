# API Specification

Base URL: `http://localhost:8000`

## `GET /health`
- **Response**: `{ "status": "ok", "timestamp": "..." }`

## `GET /upcoming`
- Returns upcoming bouts with fighters, weight class, and start time.

## `GET /probabilities`
- **Response**: List of probabilities with prediction intervals and optional market probability.

## `GET /markets`
- Provides latest odds snapshots with normalized probabilities and overround.

## `GET /recommendations`
- Ranked EV opportunities including implied Kelly fractions.

All responses include caching headers and are gzip-compressed when served via FastAPI middleware.
