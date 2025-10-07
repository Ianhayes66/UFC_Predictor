"""FastAPI application entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from ..logging import configure_logging
from ..settings import get_settings
from .routers import fights, markets, predict, probabilities, recommendations
from .schemas import HealthResponse
from ..observability import API_EXCEPTIONS, API_LATENCY, API_REQUESTS


class PrometheusRequestMiddleware(BaseHTTPMiddleware):
    """Collect Prometheus metrics for each API request."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        method = request.method
        start = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration = perf_counter() - start
            API_EXCEPTIONS.labels(route=route_path, method=method).inc()
            API_REQUESTS.labels(route=route_path, method=method, status="500").inc()
            API_LATENCY.labels(route=route_path, method=method).observe(duration)
            raise
        duration = perf_counter() - start
        API_REQUESTS.labels(route=route_path, method=method, status=str(response.status_code)).inc()
        API_LATENCY.labels(route=route_path, method=method).observe(duration)
        return response


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()
    app = FastAPI(title=settings.project_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(PrometheusRequestMiddleware)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc))

    @app.get("/metrics", tags=["system"])
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(fights.router)
    app.include_router(probabilities.router)
    app.include_router(predict.router)
    app.include_router(markets.router)
    app.include_router(recommendations.router)

    return app


def main() -> None:
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
