"""Unit tests for nullain.telemetry.tracing.configure_tracing's exporter
selection — the fix that matters: an unrecognized exporter value used to
silently install no span processor at all (spans built and discarded, no
error, no warning); it now raises. And "otlp" was pure dead code (no
branch handled it) until this change."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_tracing_provider_singleton() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """`configure_tracing` is idempotent via a module-level `_provider`
    singleton — reset it before and after each test so tests don't leak
    state into each other (the second call in a process would otherwise
    always be a no-op, hiding real bugs in later tests)."""
    import nullain.telemetry.tracing as tracing_mod  # type: ignore[reportPrivateUsage]

    tracing_mod._provider = None  # type: ignore[reportPrivateUsage]
    yield
    tracing_mod._provider = None  # type: ignore[reportPrivateUsage]


def test_console_exporter_still_works() -> None:
    from nullain.telemetry.tracing import configure_tracing, get_tracer

    configure_tracing(exporter="console")
    tracer = get_tracer()
    with tracer.start_as_current_span("test") as span:
        assert span is not None


def test_unrecognized_exporter_raises_instead_of_silently_no_oping() -> None:
    """The actual bug being fixed: before this change, an unrecognized
    exporter value installed a TracerProvider with NO span processor —
    spans were built and discarded silently. Now it must fail loudly at
    configure_tracing() time instead."""
    from nullain.telemetry.tracing import configure_tracing

    with pytest.raises(ValueError, match="Unrecognized exporter"):
        configure_tracing(exporter="not-a-real-exporter")


def test_configure_tracing_is_idempotent() -> None:
    from nullain.telemetry.tracing import configure_tracing, get_tracer

    configure_tracing(exporter="console")
    first_tracer = get_tracer()
    configure_tracing(exporter="otlp")  # second call must be a no-op, not raise/switch
    second_tracer = get_tracer()
    assert first_tracer is second_tracer


@pytest.fixture
def fake_otlp_exporter_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Installs a fake OTLP exporter module so configure_tracing's lazy
    import resolves without the real `opentelemetry-exporter-otlp-proto-http`
    extra installed — same pattern as test_postgres_event_store.py's fake
    asyncpg module for an optional dependency."""
    fake_module = ModuleType("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    fake_exporter_cls = MagicMock()
    fake_module.OTLPSpanExporter = fake_exporter_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", fake_module
    )
    return fake_module


def test_otlp_exporter_missing_extra_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)
    from nullain.telemetry.tracing import configure_tracing

    with pytest.raises(ImportError, match=r"pip install nullain-sdk\[otlp\]"):
        configure_tracing(exporter="otlp")


def test_otlp_exporter_constructed_with_configured_endpoint_and_headers(
    fake_otlp_exporter_module: ModuleType,
) -> None:
    from nullain.telemetry.tracing import configure_tracing

    configure_tracing(
        exporter="otlp",
        otlp_endpoint="http://collector.internal:4318/v1/traces",
        otlp_headers={"Authorization": "Bearer token"},
        otlp_timeout=5.0,
    )

    exporter_cls: Any = fake_otlp_exporter_module.OTLPSpanExporter
    exporter_cls.assert_called_once_with(
        endpoint="http://collector.internal:4318/v1/traces",
        headers={"Authorization": "Bearer token"},
        timeout=5.0,
    )


def test_otlp_exporter_defaults_let_underlying_sdk_read_env_vars(
    fake_otlp_exporter_module: ModuleType,
) -> None:
    """No endpoint/headers/timeout passed -> None/None/None forwarded to
    the exporter constructor, which is how the underlying OTel SDK's own
    OTEL_EXPORTER_OTLP_* env var fallback gets a chance to apply. This
    module must not reimplement that precedence logic itself."""
    from nullain.telemetry.tracing import configure_tracing

    configure_tracing(exporter="otlp")

    exporter_cls: Any = fake_otlp_exporter_module.OTLPSpanExporter
    exporter_cls.assert_called_once_with(endpoint=None, headers=None, timeout=None)
