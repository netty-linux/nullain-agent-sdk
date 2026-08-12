"""Contract tests for `nullain.ports.search.SearchProvider`.

Every `SearchProvider` adapter must pass this suite — parametrized over an
`adapter_factory` fixture so the same behavioral assertions run against any
implementation. Two adapters are covered: `WebSearchProvider`
(`nullain_tools.web_search`, always runs) and `RustSearchAdapter`
(`nullain.ports.search`, PLAN.md Fase 1 — skipped automatically when the
`nullain-search` wheel isn't installed, since it's an optional extra).

This is the first concrete implementation of the "contract test" pattern
`tests/unit/test_events_port.py`'s docstring describes but doesn't actually
apply (it exercises one adapter behaviorally and the other via a single
`isinstance` check, not one shared parametrized suite) — later contract
suites (e.g. `test_vision_provider_contract.py`) should follow this file's
shape: a factory fixture, `SearchProvider`-typed test bodies, offline via
respx (AGENTS.md rule 8: unit tests 100% offline).

Each test seeds the provider via `index()` with content matching what it
asserts on, rather than relying on a specific backend's search results —
`WebSearchProvider.index` is a documented no-op, so this seed is a no-op
for it and the DuckDuckGo mock below is what it actually searches; for
`RustSearchAdapter`, the seed is what makes the query findable, since
there's no web backend involved. Both paths converge on the same
assertions without either adapter needing backend-specific test bodies.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

import httpx
import pytest
import respx
from nullain.errors import SearchError
from nullain.ports.search import RustSearchAdapter, SearchProvider
from nullain_tools.web_search import WebSearchProvider

_DDG_URL = "https://html.duckduckgo.com/html/"

_SAMPLE_DDG_HTML = """
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frealpython.com%2Fasync%2Dio%2Dpython%2F&amp;rut=abc">
      Python&#x27;s asyncio: A Hands-On Walkthrough
    </a>
  </h2>
  <a class="result__snippet"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frealpython.com%2Fasync%2Dio%2Dpython%2F&amp;rut=abc">
    Learn <b>asyncio</b> with this tutorial.
  </a>
</div>
"""

_HAS_NULLAIN_SEARCH = importlib.util.find_spec("nullain_search") is not None


def _web_search_provider_factory() -> SearchProvider:
    return WebSearchProvider()


def _rust_search_adapter_factory() -> SearchProvider:
    return RustSearchAdapter(web_fallback=WebSearchProvider())


#: One entry per `SearchProvider` adapter this suite must validate. Add a
#: new `pytest.param(factory, id="...")` here when a new adapter is ready
#: for contract testing.
_ADAPTER_FACTORIES: list[Callable[[], SearchProvider]] = [
    pytest.param(_web_search_provider_factory, id="web_search"),  # type: ignore[list-item]
    pytest.param(  # type: ignore[list-item]
        _rust_search_adapter_factory,
        id="rust_search",
        marks=pytest.mark.skipif(
            not _HAS_NULLAIN_SEARCH, reason="nullain-search wheel not installed"
        ),
    ),
]


@pytest.fixture(params=_ADAPTER_FACTORIES)
def provider(request: pytest.FixtureRequest) -> SearchProvider:
    factory: Callable[[], SearchProvider] = request.param
    return factory()


def test_satisfies_search_provider_protocol(provider: SearchProvider) -> None:
    assert isinstance(provider, SearchProvider)


@respx.mock
async def test_query_returns_formatted_results(provider: SearchProvider) -> None:
    respx.get(_DDG_URL).mock(return_value=httpx.Response(200, text=_SAMPLE_DDG_HTML))
    await provider.index(
        "Python's asyncio: A Hands-On Walkthrough", source="https://realpython.com/async-io-python/"
    )
    result = await provider.query("asyncio tutorial", limit=5)
    assert isinstance(result, str)
    assert "realpython.com" in result


@respx.mock
async def test_query_reports_no_results_without_raising(provider: SearchProvider) -> None:
    respx.get(_DDG_URL).mock(return_value=httpx.Response(200, text="<html></html>"))
    result = await provider.query("a query with absolutely no matches anywhere")
    assert isinstance(result, str)
    assert "No results found" in result


async def test_query_rejects_empty_query(provider: SearchProvider) -> None:
    with pytest.raises(SearchError):
        await provider.query("   ")


@respx.mock
async def test_fetch_returns_url_content(provider: SearchProvider) -> None:
    respx.get("https://example.com/page").mock(
        return_value=httpx.Response(
            200, text="<p>Hello world</p>", headers={"content-type": "text/html"}
        )
    )
    result = await provider.fetch("https://example.com/page")
    assert isinstance(result, str)
    assert "Hello world" in result


@respx.mock
async def test_fetch_raises_on_http_error(provider: SearchProvider) -> None:
    respx.get("https://example.com/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(SearchError):
        await provider.fetch("https://example.com/missing")


async def test_index_is_a_documented_no_op_or_real_ingestion(provider: SearchProvider) -> None:
    # Must not raise either way — a web-only adapter treats this as a
    # documented no-op, a local-index adapter performs real ingestion.
    await provider.index("some content", source="https://example.com/doc")
