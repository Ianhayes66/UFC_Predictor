"""Prometheus metrics instrumentation utilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

from prometheus_client import Counter, Histogram

PIPELINE_ROWS_IN = Counter(
    "ufc_pipeline_rows_in_total",
    "Number of rows consumed by each pipeline step.",
    labelnames=("pipeline", "step"),
)
PIPELINE_ROWS_OUT = Counter(
    "ufc_pipeline_rows_out_total",
    "Number of rows produced by each pipeline step.",
    labelnames=("pipeline", "step"),
)
PIPELINE_STEP_DURATION = Histogram(
    "ufc_pipeline_step_duration_seconds",
    "Execution time of pipeline steps in seconds.",
    labelnames=("pipeline", "step"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, float("inf")),
)
PIPELINE_ERRORS = Counter(
    "ufc_pipeline_errors_total",
    "Number of pipeline step failures due to exceptions.",
    labelnames=("pipeline", "step"),
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

    def rows_in(self, count: int, step: str = "overall") -> None:
        """Record the number of input rows consumed by the pipeline."""

        if count >= 0:
            PIPELINE_ROWS_IN.labels(pipeline=self.pipeline, step=step).inc(count)

    def rows_out(self, count: int, step: str = "overall") -> None:
        """Record the number of output rows produced by the pipeline."""

        if count >= 0:
            PIPELINE_ROWS_OUT.labels(pipeline=self.pipeline, step=step).inc(count)

    @contextmanager
    def step(self, name: str):
        """Context manager used to capture timing and failures for a step."""

        start = perf_counter()
        try:
            yield
        except Exception:
            PIPELINE_ERRORS.labels(pipeline=self.pipeline, step=name).inc()
            PIPELINE_STEP_DURATION.labels(pipeline=self.pipeline, step=name).observe(
                perf_counter() - start
            )
            raise
        else:
            PIPELINE_STEP_DURATION.labels(pipeline=self.pipeline, step=name).observe(
                perf_counter() - start
            )


@contextmanager
def pipeline_run(name: str):
    """Context manager that measures execution time and errors for pipelines."""

    tracker = _PipelineTracker(pipeline=name)
    with tracker.step("__pipeline__"):
        yield tracker


__all__ = [
    "PIPELINE_ROWS_IN",
    "PIPELINE_ROWS_OUT",
    "PIPELINE_STEP_DURATION",
    "PIPELINE_ERRORS",
    "API_REQUESTS",
    "API_LATENCY",
    "API_EXCEPTIONS",
    "pipeline_run",
]
