from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from types import ModuleType
from typing import ClassVar
from zoneinfo import ZoneInfo

import pytest
from apscheduler.triggers.cron import CronTrigger


@dataclass
class DummyJob:
    scheduler: DummyScheduler
    job_id: str
    trigger: CronTrigger
    func: Callable[[], None]
    next_run_time: datetime = field(init=False)

    def __post_init__(self) -> None:
        """Compute the next run time when the job is created."""
        self.next_run_time = self.trigger.get_next_fire_time(
            None,
            datetime.now(self.scheduler.timezone),
        )


class DummyScheduler:
    instances: ClassVar[list[DummyScheduler]] = []

    def __init__(self, timezone: str | ZoneInfo) -> None:
        """Initialise the dummy scheduler with the provided timezone."""
        self.timezone = ZoneInfo(timezone) if isinstance(timezone, str) else timezone
        self.jobs: dict[str, DummyJob] = {}
        self.listeners: list[tuple[Callable[[object], None], int]] = []
        self.started = False
        DummyScheduler.instances.append(self)

    def add_job(
        self,
        func: Callable[[], None],
        trigger: CronTrigger,
        *,
        replace_existing: bool = False,
        **kwargs: object,
    ) -> DummyJob:
        job_id = str(kwargs["id"])
        if replace_existing and job_id in self.jobs:
            del self.jobs[job_id]
        job = DummyJob(self, job_id, trigger, func)
        self.jobs[job_id] = job
        return job

    def add_listener(self, callback: Callable[[object], None], mask: int) -> None:
        self.listeners.append((callback, mask))

    def get_job(self, job_id: str) -> DummyJob | None:
        return self.jobs.get(job_id)

    def start(self) -> None:
        self.started = True


class FakeJobsConfig:
    daily_refresh_cron = "15 10 * * *"


class FakeSettings:
    jobs = FakeJobsConfig()
    tz = "America/Los_Angeles"
    use_live_odds = True


@pytest.fixture(autouse=True)
def clear_dummy_instances() -> None:
    DummyScheduler.instances.clear()


def test_scheduler_uses_cron_and_timezone(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stub_daily_refresh = ModuleType("ufc_winprob.pipelines.daily_refresh")
    stub_daily_refresh.daily_refresh = lambda **_: None
    monkeypatch.setitem(sys.modules, "ufc_winprob.pipelines.daily_refresh", stub_daily_refresh)
    sys.modules.pop("ufc_winprob.jobs.scheduler", None)
    scheduler_module = importlib.import_module("ufc_winprob.jobs.scheduler")

    monkeypatch.setenv("TZ", "Europe/Paris")
    monkeypatch.setattr(scheduler_module, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(scheduler_module, "BlockingScheduler", DummyScheduler)
    scheduler_module.logger.add(caplog.handler, level="INFO")

    caplog.set_level("INFO")
    scheduler_module.start_scheduler()

    scheduler_instance = DummyScheduler.instances[-1]
    job = scheduler_instance.get_job("daily_refresh")
    if job is None:
        pytest.fail("daily_refresh job was not registered")

    tz = scheduler_instance.timezone
    reference = datetime(2024, 1, 1, 9, 0, tzinfo=tz)
    next_run = job.trigger.get_next_fire_time(None, reference)
    expected = datetime(2024, 1, 1, 10, 15, tzinfo=tz)
    if next_run != expected:
        pytest.fail(f"Unexpected next run time: {next_run!r}")

    next_run_str = job.next_run_time.isoformat()
    if next_run_str not in caplog.text:
        pytest.fail("Next run time was not logged on scheduler startup")

    if not scheduler_instance.started:
        pytest.fail("Scheduler.start was not invoked")
