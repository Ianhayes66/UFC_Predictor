"""FastAPI application entrypoint with observability instrumentation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from time import perf_counter

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from ufc_winprob.api.routers import fights, markets, predict, probabilities, recommendations
from ufc_winprob.api.schemas import HealthResponse
from ufc_winprob.logging import configure_logging
from ufc_winprob.observability import API_EXCEPTIONS, API_LATENCY, API_REQUESTS
from ufc_winprob.settings import get_settings

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
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(PrometheusRequestMiddleware)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", timestamp=datetime.now(UTC))

    @app.get("/metrics", tags=["system"])
    def metrics() -> Response:
        payload = generate_latest()
        return Response(payload, media_type=CONTENT_TYPE_LATEST)

    app.include_router(fights.router)
    app.include_router(probabilities.router)
    app.include_router(predict.router)
    app.include_router(markets.router)
    app.include_router(recommendations.router)

    return app


def main() -> None:
    """Run the development server."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104


if __name__ == "__main__":
    main()
