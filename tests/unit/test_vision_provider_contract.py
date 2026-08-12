"""Contract tests for `nullain.ports.vision.VisionProvider`.

Mirrors `test_search_provider_contract.py`'s shape (a parametrized
`adapter_factory` fixture whose test bodies run unchanged against any
`VisionProvider` implementation). `ModelRouterVisionProvider` (PLAN.md
Fase 2, rescoped: a plain multimodal-chat adapter routed through the core's
own `ModelRouter`/`LLMProvider`, not the separate `nullain-vision` package
Fase 0 originally assumed) is the first adapter validated here, via a fake
`LLMProvider` so this suite stays offline (AGENTS.md rule 8) — no real
network call, no API key needed.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable

import pytest
from nullain.config.settings import RouterConfig, TierConfig
from nullain.errors import VisionError
from nullain.llm.types import CompletionChunk, CompletionRequest, ImagePart
from nullain.ports.vision import ModelRouterVisionProvider, VisionProvider
from nullain.router.router import ModelRouter


class _FakeLLMProvider:
    """Offline `LLMProvider` stub: echoes back a fixed reply, or raises."""

    def __init__(self, reply: str = "a fake vision reply", *, raises: bool = False) -> None:
        self.reply = reply
        self.raises = raises
        self.last_request: CompletionRequest | None = None

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        self.last_request = request
        if self.raises:
            raise RuntimeError("model rejected multimodal content")
        return CompletionChunk(delta_text=self.reply)

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        yield CompletionChunk(delta_text=self.reply)

    async def health_check(self) -> bool:
        return True


def _model_router_vision_provider_factory() -> VisionProvider:
    router = ModelRouter(config=RouterConfig(tiers={"vision": TierConfig(models=["fake-vlm"])}))
    return ModelRouterVisionProvider(router, _FakeLLMProvider())


#: One entry per `VisionProvider` adapter this suite must validate. Add a
#: new `pytest.param(factory, id="...")` here when a new adapter is ready
#: for contract testing.
_ADAPTER_FACTORIES: list[Callable[[], VisionProvider]] = [
    pytest.param(_model_router_vision_provider_factory, id="model_router_vlm"),  # type: ignore[list-item]
]


@pytest.fixture(params=_ADAPTER_FACTORIES)
def provider(request: pytest.FixtureRequest) -> VisionProvider:
    factory: Callable[[], VisionProvider] = request.param
    return factory()


def test_vision_provider_protocol_declares_expected_methods() -> None:
    """Pins the port's method surface so a signature change here is a
    deliberate, reviewed edit — not an accidental one caught only once an
    adapter (in another repo) fails to satisfy the Protocol."""
    for method_name in ("describe_image", "ocr", "analyze_screenshot"):
        assert hasattr(VisionProvider, method_name)


async def test_satisfies_vision_provider_protocol(provider: VisionProvider) -> None:
    assert isinstance(provider, VisionProvider)


async def test_describe_image_returns_text(provider: VisionProvider) -> None:
    result = await provider.describe_image(b"\x89PNG...", mime_type="image/png")
    assert isinstance(result, str)


async def test_ocr_returns_text(provider: VisionProvider) -> None:
    result = await provider.ocr(b"\x89PNG...", mime_type="image/png")
    assert isinstance(result, str)


async def test_analyze_screenshot_returns_text(provider: VisionProvider) -> None:
    result = await provider.analyze_screenshot(b"\x89PNG...", mime_type="image/png")
    assert isinstance(result, str)


async def test_model_router_vision_provider_wraps_provider_failure_as_vision_error() -> None:
    """Adapter-specific, not part of the shared contract: pins
    `ModelRouterVisionProvider`'s translation of an underlying `LLMProvider`
    failure (e.g. a model rejecting multimodal content) to `VisionError`,
    so callers never see a raw provider exception leak through the port."""
    router = ModelRouter(config=RouterConfig(tiers={"vision": TierConfig(models=["fake-vlm"])}))
    adapter = ModelRouterVisionProvider(router, _FakeLLMProvider(raises=True))
    with pytest.raises(VisionError):
        await adapter.describe_image(b"\x89PNG...", mime_type="image/png")


async def test_model_router_vision_provider_sends_image_as_content_part() -> None:
    """Adapter-specific: confirms the image travels as an `ImagePart`
    content block on the request `ModelRouterVisionProvider` builds, not
    just that some string comes back."""
    router = ModelRouter(config=RouterConfig(tiers={"vision": TierConfig(models=["fake-vlm"])}))
    llm = _FakeLLMProvider()
    adapter = ModelRouterVisionProvider(router, llm)
    await adapter.ocr(b"\x89PNG...", mime_type="image/png")
    assert llm.last_request is not None
    assert llm.last_request.model == "fake-vlm"
    content = llm.last_request.messages[0].content
    assert isinstance(content, list)
    image_part = content[1]
    assert isinstance(image_part, ImagePart)
    assert image_part.data == b"\x89PNG..."
    assert image_part.mime_type == "image/png"


async def test_model_router_vision_provider_appends_hint_to_prompt_when_given() -> None:
    """`hint` (optional, keyword-only) is folded into the text prompt sent
    alongside the image — the Protocol's only lever for a caller to steer
    what an adapter prioritizes describing/transcribing."""
    router = ModelRouter(config=RouterConfig(tiers={"vision": TierConfig(models=["fake-vlm"])}))
    llm = _FakeLLMProvider()
    adapter = ModelRouterVisionProvider(router, llm)
    await adapter.describe_image(
        b"\x89PNG...", mime_type="image/png", hint="Focus on the error message."
    )
    assert llm.last_request is not None
    content = llm.last_request.messages[0].content
    assert isinstance(content, list)
    text_part = content[0]
    assert "Focus on the error message." in text_part.text  # type: ignore[union-attr]


async def test_model_router_vision_provider_omits_hint_block_when_not_given() -> None:
    """No `hint` (the default) leaves the prompt exactly as it was before
    this parameter existed — a regression here would silently change every
    existing caller's prompt."""
    router = ModelRouter(config=RouterConfig(tiers={"vision": TierConfig(models=["fake-vlm"])}))
    llm = _FakeLLMProvider()
    adapter = ModelRouterVisionProvider(router, llm)
    await adapter.describe_image(b"\x89PNG...", mime_type="image/png")
    assert llm.last_request is not None
    content = llm.last_request.messages[0].content
    assert isinstance(content, list)
    text_part = content[0]
    assert "User context" not in text_part.text  # type: ignore[union-attr]
