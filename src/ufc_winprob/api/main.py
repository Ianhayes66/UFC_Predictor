"""FastAPI application entrypoint with observability instrumentation."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from time import perf_counter

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from ufc_winprob.api.routers import fights, markets, predict, probabilities, recommendations
from ufc_winprob.logging import configure_logging
from ufc_winprob.settings import get_settings

try:
    from ufc_winprob.observability import API_EXCEPTIONS, API_LATENCY, API_REQUESTS
except ImportError:  # pragma: no cover - fallback for stripped deployments
    from prometheus_client import Counter, Histogram

    API_REQUESTS = Counter(
        "ufc_api_requests_total",
        "Total API requests processed by route, method, and status code.",
        labelnames=("route", "method", "status"),
    )
    API_LATENCY = Histogram(
        "ufc_api_request_latency_seconds",
        "Latency of API requests in seconds.",
        labelnames=("route", "method"),
    )
    API_EXCEPTIONS = Counter(
        "ufc_api_exceptions_total",
        "Exceptions raised while serving API routes.",
        labelnames=("route", "method"),
    )


RequestHandler = Callable[[Request], Awaitable[Response]]


class PrometheusRequestMiddleware(BaseHTTPMiddleware):
    """Collect Prometheus metrics for each API request."""

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:  # type: ignore[override]
        """Record metrics for a request lifecycle and propagate the response."""
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        method = request.method
        start = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            API_EXCEPTIONS.labels(route=route_path, method=method).inc()
            API_REQUESTS.labels(route=route_path, method=method, status="500").inc()
            API_LATENCY.labels(route=route_path, method=method).observe(perf_counter() - start)
            raise
        status_code = str(response.status_code)
        API_REQUESTS.labels(route=route_path, method=method, status=status_code).inc()
        API_LATENCY.labels(route=route_path, method=method).observe(perf_counter() - start)
        return response


def create_app() -> FastAPI:
    """Instantiate the FastAPI application."""
    settings = get_settings()
    configure_logging()

    app = FastAPI(title=settings.project_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://localhost:3000",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(PrometheusRequestMiddleware)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "time": datetime.now(UTC).isoformat()}

    @app.get("/metrics", tags=["system"])
    def metrics() -> Response:
        payload = generate_latest()
        return Response(payload, media_type=CONTENT_TYPE_LATEST)

    app.include_router(fights.router)
    app.include_router(markets.router)
    app.include_router(probabilities.router)
    app.include_router(recommendations.router)
    app.include_router(predict.router)

    return app


def main() -> None:
    """Run the development server."""
    import uvicorn

    app = create_app()
    settings = get_settings()

    port_str = os.getenv("PORT") or os.getenv("APP_PORT") or str(getattr(settings, "api_port", ""))
    try:
        port = int(port_str)
    except (TypeError, ValueError):
        port = 8000

    uvicorn.run(app, host="0.0.0.0", port=port)  # noqa: S104


if __name__ == "__main__":
    main()
