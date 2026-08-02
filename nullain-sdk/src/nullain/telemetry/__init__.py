"""Nullain Agent SDK — Telemetry and Structured Logging."""

import logging
import sys
from typing import Any, cast

import structlog
from structlog.stdlib import BoundLogger

_configured = False


def configure_telemetry(log_level: str = "INFO", json_format: bool = False) -> None:
    """Configure structlog for Nullain SDK.

    Args:
        log_level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        json_format: If True, output JSON formatted logs, else key-value console logs.
    """
    global _configured
    if _configured:
        return

    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),  # type: ignore[reportUnknownMemberType]
        structlog.processors.StackInfoRenderer(),  # type: ignore[reportUnknownMemberType]
        structlog.processors.UnicodeDecoder(),  # type: ignore[reportUnknownMemberType]
    ]

    renderer: Any
    if json_format:
        renderer = structlog.processors.JSONRenderer()  # type: ignore[reportUnknownMemberType]
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())  # type: ignore[reportUnknownMemberType]

    structlog.configure(  # type: ignore[reportUnknownMemberType]
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,  # type: ignore[reportUnknownMemberType]
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),  # type: ignore[reportUnknownMemberType]
        wrapper_class=structlog.stdlib.BoundLogger,  # type: ignore[reportUnknownMemberType]
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(  # type: ignore[reportUnknownMemberType]
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,  # type: ignore[reportUnknownMemberType]
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(cast(logging.Formatter, formatter))

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    _configured = True


def get_logger(name: str = "nullain") -> BoundLogger:
    """Get a bound structlog logger."""
    if not _configured:
        configure_telemetry()
    return cast(BoundLogger, structlog.get_logger(name))  # type: ignore[reportUnknownMemberType]


def bind_context(**kwargs: Any) -> None:
    """Bind contextual key-value pairs to the current async execution context."""
    structlog.contextvars.bind_contextvars(**kwargs)  # type: ignore[reportUnknownMemberType]


def clear_context() -> None:
    """Clear bound context variables."""
    structlog.contextvars.clear_contextvars()  # type: ignore[reportUnknownMemberType]


__all__ = [
    "bind_context",
    "clear_context",
    "configure_telemetry",
    "get_logger",
]
