# Operations Runbook

## Daily Refresh
- Cron: `0 8 * * *` (UTC) via APScheduler.
- Steps: build dataset → train → predict → update upcoming.
- Artifacts stored under `data/processed/` and `data/models/` with timestamps.

## Monitoring
- Structured logs (loguru) with contextual keys.
- Prometheus client counters emitted for ingestion counts (extend in production).
- `SCRAPER_HEALTH.md` updated when scrapers fallback to cached data.

## Incident Response
1. Check logs in `data/logs` (if file sink enabled).
2. Re-run `make refresh` locally to reproduce.
3. File GitHub issue with details, attach metrics snapshots.

## Secrets
- API keys managed via `.env` or orchestration secret store.
- Never commit real credentials; `.env.example` documents required keys.
