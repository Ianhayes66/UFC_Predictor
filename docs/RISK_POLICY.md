# Risk Policy

1. **Data Compliance** – Only public and regulatory-permitted data is stored. Scrapers honor robots.txt and back off on HTTP 429.
2. **Operational Risk** – Pipelines include retries, caching, and fallback snapshots. Failures surface via structured logging and CI.
3. **Model Risk** – Synthetic training may diverge from reality; monitor calibration drift monthly. Do not use unvalidated outputs for production wagering.
4. **Security** – Secrets managed via environment variables. No credentials are stored in the repo. Enable network ACLs for production deployments.
5. **Responsible Betting** – EV recommendations assume fractional Kelly staking with caps. Users must comply with jurisdictional regulations and bankroll management best practices.
