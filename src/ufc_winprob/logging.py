"""Structured logging utilities using loguru."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from .settings import get_settings

_LOG_CONFIGURED = False


def configure_logging(log_path: Optional[Path] = None) -> None:
    """Configure loguru logging to stdout and rotating file."""

    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return

    settings = get_settings()
    log_level = settings.logging.level
    serialize = settings.logging.json
    default_log_path = settings.logging.directory / "app.log"
    log_file = log_path or default_log_path

    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level,
        format=(
            "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {message}"
        ),
        colorize=False,
        serialize=serialize,
    )
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            rotation="7 days",
            retention="30 days",
            compression="zip",
            enqueue=True,
            serialize=serialize,
        )
    _LOG_CONFIGURED = True


__all__ = ["configure_logging", "logger"]
