"""APScheduler based job runner."""

from __future__ import annotations

import os
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from ..pipelines.daily_refresh import daily_refresh
from ..settings import get_settings


def start_scheduler() -> None:
    settings = get_settings()
    timezone_name = os.environ.get("TZ") or settings.tz
    scheduler = BlockingScheduler(timezone=timezone_name)
    cron = settings.jobs.daily_refresh_cron
    trigger = CronTrigger.from_crontab(cron, timezone=scheduler.timezone)
    scheduler.add_job(
        lambda: daily_refresh(use_live_odds=settings.use_live_odds),
        trigger=trigger,
        id="daily_refresh",
        replace_existing=True,
    )
    next_run = trigger.get_next_fire_time(None, datetime.now(scheduler.timezone))
    logger.info(
        "Scheduler initialised (timezone=%s, cron=%s, next_run=%s)",
        timezone_name,
        cron,
        next_run,
    )
    for job in scheduler.get_jobs():
        logger.info("Scheduled job %s -> %s", job.id, job.next_run_time)
    scheduler.start()


def main() -> None:
    start_scheduler()


if __name__ == "__main__":
    main()
