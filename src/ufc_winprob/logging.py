"""Structured logging utilities using loguru."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from .settings import get_settings

_LOG_CONFIGURED = False


def configure_logging(log_path: Optional[Path] = None) -> None:
    """Configure loguru logging with structured JSON output.

    Args:
        log_path: Optional path for file logging. When omitted only stdout is used.
    """

    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return

    settings = get_settings()
    log_level = settings.logging.level

    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level,
        format=(
            "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {extra[event_id]!s} | {extra[bout_id]!s} | "
            "{message}"
        ),
        colorize=False,
        serialize=False,
    )
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, level=log_level, rotation="7 days", compression="zip", enqueue=True)
    _LOG_CONFIGURED = True


__all__ = ["configure_logging", "logger"]
