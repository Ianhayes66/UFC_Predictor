"""APScheduler based job runner."""

from __future__ import annotations

import os
from collections.abc import Callable

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from ufc_winprob.pipelines.daily_refresh import daily_refresh
from ufc_winprob.settings import get_settings


def _build_daily_refresh_job(use_live_odds: bool) -> Callable[[], None]:
    """Create the callable used by APScheduler for the daily refresh job."""

    def _job() -> None:
        try:
            daily_refresh(use_live_odds=use_live_odds)
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("daily_refresh job failed: %s", exc)
            raise

    return _job


def _log_job_event(scheduler: BlockingScheduler, event: JobExecutionEvent) -> None:
    """Log a concise summary for each job execution event."""
    job = scheduler.get_job(event.job_id)
    next_run = job.next_run_time.isoformat() if job and job.next_run_time else "unscheduled"
    if event.exception:
        logger.error(
            "Job {} failed at {} (next_run={})",
            event.job_id,
            event.scheduled_run_time,
            next_run,
        )
    else:
        logger.info(
            "Job {} succeeded at {} (next_run={})",
            event.job_id,
            event.scheduled_run_time,
            next_run,
        )


def start_scheduler() -> None:
    """Initialise and start the APScheduler blocking scheduler."""
    settings = get_settings()
    timezone_name = os.environ.get("TZ") or settings.tz
    scheduler = BlockingScheduler(timezone=timezone_name)

    cron_expression = settings.jobs.daily_refresh_cron
    trigger = CronTrigger.from_crontab(cron_expression, timezone=scheduler.timezone)

    job = scheduler.add_job(
        _build_daily_refresh_job(settings.use_live_odds),
        trigger=trigger,
        id="daily_refresh",
        replace_existing=True,
    )

    if job.next_run_time:
        logger.info(
            "daily_refresh scheduled (cron={}, timezone={}, next_run={})",
            cron_expression,
            scheduler.timezone,
            job.next_run_time.isoformat(),
        )
    else:  # pragma: no cover - defensive logging path
        logger.warning(
            "daily_refresh scheduled without a computed next run (cron={}, timezone={})",
            cron_expression,
            scheduler.timezone,
        )

    scheduler.add_listener(
        lambda event: _log_job_event(scheduler, event),
        EVENT_JOB_EXECUTED | EVENT_JOB_ERROR,
    )

    scheduler.start()


def main() -> None:
    """Entry point used by console scripts."""
    start_scheduler()


if __name__ == "__main__":
    main()
