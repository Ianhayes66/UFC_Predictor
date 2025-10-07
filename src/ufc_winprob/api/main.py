"""FastAPI application entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..logging import configure_logging, logger
from ..settings import get_settings
from .routers import fights, markets, probabilities, recommendations
from .schemas import HealthResponse


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

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc))

    app.include_router(fights.router)
    app.include_router(probabilities.router)
    app.include_router(markets.router)
    app.include_router(recommendations.router)

    return app


def main() -> None:
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
