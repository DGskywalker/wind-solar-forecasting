"""Structured logging utilities using structlog."""

import logging
import sys
import uuid
from typing import Any

import structlog


def setup_logger(log_level: str = "INFO", json_format: bool = False) -> structlog.BoundLogger:
    """Configures structured logger with context variables and timestamps.

    Args:
        log_level: Logging verbosity level (DEBUG, INFO, WARNING, ERROR).
        json_format: Whether to output raw JSON logs instead of formatted console text.

    Returns:
        Configured structlog BoundLogger instance.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    formatter_processor: Any
    if json_format:
        formatter_processor = structlog.processors.JSONRenderer()
    else:
        formatter_processor = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [formatter_processor],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger()
    logger = logger.bind(correlation_id=str(uuid.uuid4())[:8])
    return logger


logger = setup_logger()
