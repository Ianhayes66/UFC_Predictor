# Data Sources

| Source | Type | Access Method | Notes |
| --- | --- | --- | --- |
| UFCStats | Historical fights, fighters, events | Polite scraping with cached HTML snapshots | Respect robots.txt, 5s delay, cached in `data/raw/ufcstats_cache`. |
| Tapology | Upcoming events (optional) | Configurable scraper | Disabled by default, requires manual enablement. |
| Odds APIs | Market odds | REST API via configurable client | Aggregates by median, supports Shin normalization. |

All scrapers implement retries (tenacity), randomized user-agents, and checksum verification. If a scrape fails, the system reuses the last successful snapshot and records an alert in `SCRAPER_HEALTH.md`.
