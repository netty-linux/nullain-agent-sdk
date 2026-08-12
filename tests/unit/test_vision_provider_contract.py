"""Contract tests for `nullain.ports.vision.VisionProvider`.

Mirrors `test_search_provider_contract.py`'s shape (a parametrized
`adapter_factory` fixture whose test bodies run unchanged against any
`VisionProvider` implementation), but no adapter exists yet in this repo —
per PLAN.md Fase 0 scope, `VisionProvider` implementations live in the
separate `nullain-vision` package (Fase 2), installed as an optional extra.

`_ADAPTER_FACTORIES` is intentionally empty: this suite defines the
contract's *shape* (the Protocol conformance and call-signature checks
below), so it is ready to validate the first real adapter the moment one is
added — add a `pytest.param(factory, id="...")` to `_ADAPTER_FACTORIES`
then, with no other changes needed here. Until then, the parametrized tests
collect zero cases (pytest reports them as no-ops, not failures) and the
protocol-shape test runs unconditionally, so this file stays green in
`make check` without depending on any vision implementation existing.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from nullain.ports.vision import VisionProvider

#: One entry per `VisionProvider` adapter this suite must validate. Empty
#: today (PLAN.md Fase 0: no vision adapter ships here) — add a
#: `pytest.param(factory, id="...")` once `nullain-vision` (Fase 2) has a
#: concrete implementation to test against.
_ADAPTER_FACTORIES: list[Callable[[], VisionProvider]] = []


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
