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
        structlog.processors.TimeStamper(fmt="iso"),  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs
        structlog.processors.StackInfoRenderer(),  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs
        structlog.processors.UnicodeDecoder(),  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs
    ]

    renderer: Any
    if json_format:
        renderer = structlog.processors.JSONRenderer()  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs

    structlog.configure(  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs
        wrapper_class=structlog.stdlib.BoundLogger,  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(cast(logging.Formatter, formatter))

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # httpx/httpcore log one INFO line per outbound HTTP request via the
    # stdlib `logging` module (not structlog), which propagates straight to
    # the root handler above — printing "HTTP Request: POST ... 200 OK" into
    # the middle of the interactive TUI's tool-call stream, entirely outside
    # Rich's control. Wanted for debugging with a raised log level, not for
    # normal interactive use, so it's raised to WARNING here regardless of
    # the configured `level` (a real HTTP failure still logs at ERROR).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str = "nullain") -> BoundLogger:
    """Get a bound structlog logger."""
    if not _configured:
        configure_telemetry()
    return cast(BoundLogger, structlog.get_logger(name))  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs


def bind_context(**kwargs: Any) -> None:
    """Bind contextual key-value pairs to the current async execution context."""
    structlog.contextvars.bind_contextvars(**kwargs)  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs


def clear_context() -> None:
    """Clear bound context variables."""
    structlog.contextvars.clear_contextvars()  # type: ignore[reportUnknownMemberType]  # structlog public API lacks complete type stubs


from nullain.telemetry.tracing import (  # noqa: E402  (deferred: tracing imports get_logger from this package)
    CostTracker,
    configure_tracing,
    get_cost_tracker,
    get_tracer,
    span,
)

__all__ = [
    "CostTracker",
    "bind_context",
    "clear_context",
    "configure_telemetry",
    "configure_tracing",
    "get_cost_tracker",
    "get_logger",
    "get_tracer",
    "span",
]
