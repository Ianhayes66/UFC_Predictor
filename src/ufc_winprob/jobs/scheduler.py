"""APScheduler based job runner."""

from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

from ..pipelines.daily_refresh import daily_refresh


def start_scheduler() -> None:
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(lambda: daily_refresh(mock=True), "cron", hour=8, minute=0)
    logger.info("Starting scheduler with daily refresh job at 08:00 UTC")
    scheduler.start()


def main() -> None:
    start_scheduler()


if __name__ == "__main__":
    main()
