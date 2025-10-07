# Scraper Health & Safety Notes

The Tapology and UFCStats scrapers are designed to be conservative citizens of the web.

## Politeness Defaults

- **User-Agent** – Configurable via `providers.user_agent` in `config/base.yaml` (defaults to `ufc-winprob-bot/1.0`).
- **Robots.txt** – Both scrapers fetch `robots.txt` on startup and abort requests when disallowed while logging the reason.
- **Rate Limiting** – Requests are throttled using the shared `providers.rate_limit_seconds` interval with a jittered backoff (`providers.rate_limit_jitter_seconds`) to avoid burstiness.
- **Retries** – Transient HTTP errors are retried with exponential backoff (`tenacity`) before falling back to cached fixtures.

## Fallback Behaviour

- Cached HTML/JSON fixtures under `data/raw/` and `data/interim/` are reused when live calls fail or are disallowed.
- If both live data and fixtures are unavailable, an empty list is returned so downstream pipelines can degrade gracefully.

## Operational Tips

- Increase the rate limit interval and jitter when running large historical backfills to reduce load on providers.
- Populate `.env` with `ODDS_API_KEY` only when live odds polling is required; otherwise the system remains in mock mode.
- Monitor `/metrics` for `ufc_pipeline_errors_total` and `ufc_api_exceptions_total` to spot ingestion slowdowns.
