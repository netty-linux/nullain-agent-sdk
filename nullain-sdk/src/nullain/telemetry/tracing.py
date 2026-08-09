"""Nullain Agent SDK — OpenTelemetry tracing facade and cost tracking.

Exposes a thin wrapper over the OpenTelemetry API so the AgentLoop, provider,
and tool dispatcher can emit structured spans:

    agent.run (root)
      ├─ llm_request  (model, prompt_tokens, completion_tokens, ttft_ms, cost_usd)
      └─ tool         (tool_name, is_error, blocked)

When ``configure_tracing`` has not been called, the OpenTelemetry API returns
a no-op tracer, so emitting spans is cheap and safe in tests / unattended runs.

Two exporters: ``"console"`` (the original, prints spans to stdout — useful
for dev, no external dependency) and ``"otlp"`` (ships spans to any real
OTLP-compatible backend — Jaeger, Tempo, Honeycomb, etc.). Before this,
``configure_tracing(exporter=...)`` silently installed NO span processor for
any value other than ``"console"`` — spans were built and discarded with no
error, no warning, nothing ever reaching a real backend. That was the actual
bug; ``"otlp"`` support is the fix, not just a new feature. An unrecognized
exporter value now raises instead of silently no-op'ing.

Cost tracking is layered on top via a ``CostTracker`` singleton: each LLM
request is priced against a configurable per-model table (USD per 1K tokens,
input/output) and accumulated per (model, tier, agent). The computed cost is
recorded both as a span attribute and on the tracker for aggregation.
"""

import importlib
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Span
from opentelemetry.trace.status import StatusCode
from structlog.stdlib import BoundLogger

from nullain.telemetry import get_logger

logger: BoundLogger = get_logger("nullain.telemetry.tracing")

_provider: TracerProvider | None = None

#: Exporter names `configure_tracing` accepts. Kept as an explicit set (not
#: inferred from a dict of constructors) so the "unrecognized value" error
#: message can list them without importing the otlp module eagerly.
_VALID_EXPORTERS = frozenset({"console", "otlp"})

# Approximate USD per 1K tokens (input, output). Unknown models default to 0.0
# (cost not tracked). These are rough list prices for rough accounting only;
# override via ``CostTracker.set_price`` for production accuracy.
_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-oss:20b": (0.0, 0.0),
    "gpt-oss:120b": (0.0, 0.0),
    "qwen3-coder:480b-cloud": (0.0, 0.0),
    "deepseek-v4-pro": (0.0, 0.0),
}


def _build_otlp_exporter(
    endpoint: str | None, headers: dict[str, str] | None, timeout: float | None
) -> Any:
    """Import the OTLP HTTP exporter lazily — same pattern as the SDK's
    other optional-dependency adapters (`FastEmbedProvider`, `QdrantStore`,
    `PostgresEventStore`): `opentelemetry-exporter-otlp-proto-http` is an
    extra, not a hard dependency, so importing it eagerly at module load
    would force it on every installation regardless of whether OTLP export
    is ever used. HTTP/protobuf over gRPC: no native `grpcio` dependency to
    compile/ship, and any OTLP-compatible collector accepts it on the
    standard port 4318."""
    try:
        otlp_mod = importlib.import_module("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    except ImportError as exc:
        raise ImportError(
            "configure_tracing(exporter='otlp') requires the 'otlp' extra: "
            "pip install nullain-sdk[otlp]"
        ) from exc
    exporter_cls: Any = otlp_mod.OTLPSpanExporter
    # endpoint=None/headers=None let the exporter fall back to the standard
    # OTEL_EXPORTER_OTLP_* environment variables itself — not reimplemented
    # here, since the underlying SDK already owns that precedence logic.
    return exporter_cls(endpoint=endpoint, headers=headers, timeout=timeout)


def configure_tracing(
    service_name: str = "nullain",
    exporter: str = "console",
    *,
    otlp_endpoint: str | None = None,
    otlp_headers: dict[str, str] | None = None,
    otlp_timeout: float | None = None,
) -> None:
    """Install a global TracerProvider.

    Args:
        service_name: OTel resource service.name attribute.
        exporter: ``"console"`` prints spans to stdout (useful for dev, no
            external dependency). ``"otlp"`` ships spans to a real
            OTLP-compatible backend (Jaeger, Tempo, Honeycomb, ...) via
            HTTP/protobuf, batched (not one export call per span). Any
            other value raises ``ValueError`` — an earlier version of this
            function silently installed no span processor at all for an
            unrecognized value, which meant spans were built and discarded
            with no error and no warning; that was a bug, not intended
            behavior, and this is the fix.
        otlp_endpoint: OTLP HTTP endpoint (e.g.
            ``"http://localhost:4318/v1/traces"``). When None, the
            exporter falls back to the standard
            ``OTEL_EXPORTER_OTLP_ENDPOINT``/``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``
            environment variables. Ignored when ``exporter != "otlp"``.
        otlp_headers: Extra headers sent with every export request (e.g.
            an API key some collectors require). Ignored when
            ``exporter != "otlp"``.
        otlp_timeout: Export request timeout in seconds. Ignored when
            ``exporter != "otlp"``.

    Raises:
        ValueError: ``exporter`` is not one of the recognized values.
        ImportError: ``exporter="otlp"`` but the ``otlp`` extra isn't
            installed (``pip install nullain-sdk[otlp]``).

    Idempotent — a second call is a no-op regardless of the arguments
    passed, matching the original behavior.
    """
    if exporter not in _VALID_EXPORTERS:
        raise ValueError(
            f"Unrecognized exporter {exporter!r}; expected one of {sorted(_VALID_EXPORTERS)}."
        )
    global _provider
    if _provider is not None:
        return
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    if exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        otlp_exporter = _build_otlp_exporter(otlp_endpoint, otlp_headers, otlp_timeout)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    trace.set_tracer_provider(provider)
    _provider = provider


def get_tracer() -> trace.Tracer:
    """Return the configured tracer (no-op if ``configure_tracing`` not called)."""
    return trace.get_tracer("nullain.agent")


@contextmanager
def span(name: str, **attributes: Any) -> Generator[Span, None, None]:
    """Context manager that opens a span, records attributes, and on exit
    sets status (ERROR on exception, recording the exception)."""
    with get_tracer().start_as_current_span(name) as s:
        span_obj: Span = s  # type: ignore[assignment]
        if attributes:
            span_obj.set_attributes(attributes)
        try:
            yield span_obj
        except Exception as err:
            span_obj.record_exception(err)
            span_obj.set_status(StatusCode.ERROR, str(err))
            raise
        else:
            span_obj.set_status(StatusCode.OK)


class CostTracker:
    """Accumulates LLM cost per (model, tier, agent) from token usage.

    Prices are USD per 1K tokens as (input, output). Models without a price
    entry are tracked at 0.0 cost (tokens still counted).
    """

    def __init__(self) -> None:
        self._prices: dict[str, tuple[float, float]] = dict(_DEFAULT_PRICES)
        # key: (model, tier, agent) -> {"input": int, "output": int, "cost": float}
        self._totals: dict[tuple[str, str, str], dict[str, float]] = {}

    def set_price(self, model: str, input_per_1k: float, output_per_1k: float) -> None:
        """Override or register the price for a model."""
        self._prices[model] = (input_per_1k, output_per_1k)

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        tier: str = "",
        agent: str = "",
    ) -> float:
        """Record a request's tokens and return its computed cost in USD."""
        in_price, out_price = self._prices.get(model, (0.0, 0.0))
        cost = (prompt_tokens / 1000.0) * in_price + (completion_tokens / 1000.0) * out_price
        key = (model, tier, agent)
        bucket = self._totals.setdefault(key, {"input": 0.0, "output": 0.0, "cost": 0.0})
        bucket["input"] += prompt_tokens
        bucket["output"] += completion_tokens
        bucket["cost"] += cost
        return cost

    def totals(self) -> dict[tuple[str, str, str], dict[str, float]]:
        """Return a copy of accumulated totals keyed by (model, tier, agent)."""
        return {k: dict(v) for k, v in self._totals.items()}

    def reset(self) -> None:
        """Clear accumulated totals (prices retained)."""
        self._totals.clear()


_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Return the process-global CostTracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


__all__ = [
    "CostTracker",
    "configure_tracing",
    "get_cost_tracker",
    "get_tracer",
    "span",
]
