"""Unit tests for telemetry/logging configuration."""

import logging

import nullain.telemetry as telemetry


def test_configure_telemetry_silences_httpx_info_logs() -> None:
    """Regression (M20): httpx/httpcore log one INFO line per outbound HTTP
    request via the stdlib `logging` module, which propagates to the root
    handler configure_telemetry() installs — printing "HTTP Request: POST
    ... 200 OK" straight into the interactive TUI's tool-call stream,
    entirely outside Rich's control (reported live: this line appeared
    mid-chat between tool results). Both loggers must be raised above INFO
    so normal interactive use stays clean.
    """
    telemetry._configured = False  # type: ignore[reportPrivateUsage]
    logging.getLogger("httpx").setLevel(logging.NOTSET)
    logging.getLogger("httpcore").setLevel(logging.NOTSET)

    telemetry.configure_telemetry()

    assert logging.getLogger("httpx").getEffectiveLevel() > logging.INFO
    assert logging.getLogger("httpcore").getEffectiveLevel() > logging.INFO
