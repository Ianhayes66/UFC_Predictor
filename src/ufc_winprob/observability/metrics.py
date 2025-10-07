"""Prometheus metrics instrumentation utilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

from prometheus_client import Counter, Histogram

PIPELINE_ROWS = Counter(
    "ufc_pipeline_rows_total",
    "Number of rows observed per pipeline stage.",
    labelnames=("pipeline", "direction"),
)
PIPELINE_DURATION = Histogram(
    "ufc_pipeline_duration_seconds",
    "Pipeline execution duration in seconds.",
    labelnames=("pipeline",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, float("inf")),
)
PIPELINE_ERRORS = Counter(
    "ufc_pipeline_errors_total",
    "Count of pipeline executions that raised an exception.",
    labelnames=("pipeline",),
)

API_REQUESTS = Counter(
    "ufc_api_requests_total",
    "Total API requests processed by route, method, and status code.",
    labelnames=("route", "method", "status"),
)
API_LATENCY = Histogram(
    "ufc_api_request_latency_seconds",
    "Latency of API requests in seconds.",
    labelnames=("route", "method"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, float("inf")),
)
API_EXCEPTIONS = Counter(
    "ufc_api_exceptions_total",
    "Exceptions raised while serving API routes.",
    labelnames=("route", "method"),
)


@dataclass(slots=True)
class _PipelineTracker:
    """Helper allowing pipelines to record metrics within a context manager."""

    pipeline: str

    def rows_in(self, count: int) -> None:
        """Record the number of input rows consumed by the pipeline."""

        if count >= 0:
            PIPELINE_ROWS.labels(pipeline=self.pipeline, direction="in").inc(count)

    def rows_out(self, count: int) -> None:
        """Record the number of output rows produced by the pipeline."""

        if count >= 0:
            PIPELINE_ROWS.labels(pipeline=self.pipeline, direction="out").inc(count)


@contextmanager
def pipeline_run(name: str):
    """Context manager that measures execution time and errors for pipelines."""

    start = perf_counter()
    tracker = _PipelineTracker(pipeline=name)
    try:
        yield tracker
    except Exception:
        PIPELINE_ERRORS.labels(pipeline=name).inc()
        PIPELINE_DURATION.labels(pipeline=name).observe(perf_counter() - start)
        raise
    else:
        PIPELINE_DURATION.labels(pipeline=name).observe(perf_counter() - start)


__all__ = [
    "PIPELINE_ROWS",
    "PIPELINE_DURATION",
    "PIPELINE_ERRORS",
    "API_REQUESTS",
    "API_LATENCY",
    "API_EXCEPTIONS",
    "pipeline_run",
]
