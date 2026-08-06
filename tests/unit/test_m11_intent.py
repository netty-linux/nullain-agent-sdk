"""M11 — IntentParser with real classifier (4.3).

Offline tests: a fake LLM provider returns scripted classifier output, so the
heuristic-first / classifier-fallback / cache behavior of ``parse_async`` is
exercised with no network. The classifier failure path is covered by a provider
that raises. ``_parse_classifier_output`` is exercised directly, so
``reportPrivateUsage`` is disabled for this file (same as ``test_cli.py``).
"""

# pyright: reportPrivateUsage=false

from collections.abc import AsyncGenerator

import pytest
from nullain.llm import CompletionChunk, CompletionRequest
from nullain.router import Complexity, IntentParser, IntentResult


class FakeProvider:
    """LLM provider that returns a scripted classifier response."""

    def __init__(self, delta_text: str = "", raise_on_generate: bool = False) -> None:
        self._delta_text = delta_text
        self._raise_on_generate = raise_on_generate
        self.calls: list[CompletionRequest] = []

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        self.calls.append(request)
        if self._raise_on_generate:
            raise RuntimeError("classifier unavailable")
        return CompletionChunk(delta_text=self._delta_text)

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        yield CompletionChunk(delta_text=self._delta_text)

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Heuristic-first: confident prompts never call the classifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_async_confident_heuristic_skips_classifier() -> None:
    provider = FakeProvider(delta_text="intent: complex_architecture\ncomplexity: HIGH")
    parser = IntentParser()

    res = await parser.parse_async("refactor the event bus", provider, "clf-model")

    assert res.complexity == Complexity.HIGH
    assert res.intent_type == "complex_architecture"
    assert provider.calls == []  # classifier never invoked


# ---------------------------------------------------------------------------
# Classifier fallback: uncertain prompt routes to the classifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_async_uncertain_prompt_uses_classifier() -> None:
    provider = FakeProvider(delta_text="intent: general_task\ncomplexity: HIGH")
    parser = IntentParser()

    res = await parser.parse_async("build a distributed tracing system", provider, "clf-model")

    assert res.complexity == Complexity.HIGH
    assert res.suggested_tier == "deep"
    assert len(provider.calls) == 1
    assert provider.calls[0].model == "clf-model"
    assert provider.calls[0].temperature == 0.0


@pytest.mark.asyncio
async def test_parse_async_classifier_low_complexity() -> None:
    provider = FakeProvider(delta_text="intent: simple_edit\ncomplexity: LOW")
    parser = IntentParser()

    res = await parser.parse_async("tweak the css on the landing page", provider, "clf-model")

    assert res.complexity == Complexity.LOW
    assert res.suggested_tier == "fast"


# ---------------------------------------------------------------------------
# Failure fallback: classifier outage / no provider -> heuristic default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_async_classifier_failure_falls_back_to_heuristic() -> None:
    provider = FakeProvider(raise_on_generate=True)
    parser = IntentParser()

    res = await parser.parse_async("build a distributed tracing system", provider, "clf-model")

    # Falls back to the heuristic MEDIUM default.
    assert res.complexity == Complexity.MEDIUM
    assert res.intent_type == "general_task"
    assert res.suggested_tier == "balanced"


@pytest.mark.asyncio
async def test_parse_async_no_provider_stays_heuristic_only() -> None:
    parser = IntentParser()

    res = await parser.parse_async("build a distributed tracing system", None, None)

    assert res.complexity == Complexity.MEDIUM
    assert res.intent_type == "general_task"


@pytest.mark.asyncio
async def test_parse_async_unparseable_classifier_output_falls_back() -> None:
    provider = FakeProvider(delta_text="I am not sure what this is.")
    parser = IntentParser()

    res = await parser.parse_async("build a distributed tracing system", provider, "clf-model")

    assert res.complexity == Complexity.MEDIUM
    assert res.intent_type == "general_task"


# ---------------------------------------------------------------------------
# Caching: same prompt is served from cache without re-calling the classifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_async_caches_by_prompt_hash() -> None:
    provider = FakeProvider(delta_text="intent: general_task\ncomplexity: HIGH")
    parser = IntentParser()

    first = await parser.parse_async("build a distributed tracing system", provider, "clf-model")
    second = await parser.parse_async("build a distributed tracing system", provider, "clf-model")

    assert first == second
    assert len(provider.calls) == 1  # classifier called only once


@pytest.mark.asyncio
async def test_parse_async_different_prompts_not_cached_together() -> None:
    provider = FakeProvider(delta_text="intent: general_task\ncomplexity: HIGH")
    parser = IntentParser()

    await parser.parse_async("build a distributed tracing system", provider, "clf-model")
    await parser.parse_async("design a new payment gateway", provider, "clf-model")

    assert len(provider.calls) == 2


# ---------------------------------------------------------------------------
# parse (sync) remains heuristic-only and backward compatible
# ---------------------------------------------------------------------------


def test_parse_sync_remains_heuristic_only() -> None:
    parser = IntentParser()
    res = parser.parse("build a distributed tracing system")
    assert res.complexity == Complexity.MEDIUM
    assert res.intent_type == "general_task"


def test_parse_sync_confident_keyword() -> None:
    parser = IntentParser()
    res = parser.parse("format this file")
    assert res.complexity == Complexity.LOW
    assert res.intent_type == "simple_edit"


# ---------------------------------------------------------------------------
# Classifier output parsing
# ---------------------------------------------------------------------------


def test_parse_classifier_output_recognizes_all_complexities() -> None:
    parser = IntentParser()
    assert parser._parse_classifier_output("intent: simple_edit\ncomplexity: LOW") == IntentResult(
        intent_type="simple_edit", complexity=Complexity.LOW, suggested_tier="fast"
    )
    assert parser._parse_classifier_output(
        "intent: general_task\ncomplexity: MEDIUM"
    ) == IntentResult(
        intent_type="general_task", complexity=Complexity.MEDIUM, suggested_tier="balanced"
    )
    assert parser._parse_classifier_output(
        "intent: complex_architecture\ncomplexity: HIGH"
    ) == IntentResult(
        intent_type="complex_architecture", complexity=Complexity.HIGH, suggested_tier="deep"
    )


def test_parse_classifier_output_rejects_garbage() -> None:
    parser = IntentParser()
    assert parser._parse_classifier_output("") is None
    assert parser._parse_classifier_output("hello world") is None
    assert parser._parse_classifier_output("complexity: HIGH") is None  # missing intent
    assert parser._parse_classifier_output("intent: simple_edit") is None  # missing complexity
