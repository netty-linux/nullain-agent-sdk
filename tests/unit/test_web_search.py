"""Unit tests for web_search's real HTTP path (respx-mocked, no network)
and its DuckDuckGo HTML parsing regexes."""

import httpx
import pytest
import respx
from nullain_tools.web_search import (
    _extract_target_url,  # type: ignore[reportPrivateUsage]
    create_web_search_tool,
)

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
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html&amp;rut=def">
      asyncio — Python 3 documentation
    </a>
  </h2>
  <a class="result__snippet"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html&amp;rut=def">
    Official <b>asyncio</b> docs.
  </a>
</div>
"""


def test_extract_target_url_decodes_ddg_redirector() -> None:
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fq%3D1&rut=abc"
    assert _extract_target_url(href) == "https://example.com/path?q=1"


def test_extract_target_url_returns_none_without_uddg_param() -> None:
    assert _extract_target_url("//duckduckgo.com/l/?rut=abc") is None


@pytest.mark.asyncio
@respx.mock
async def test_web_search_parses_results_with_real_urls_and_snippets() -> None:
    respx.get("https://html.duckduckgo.com/html/").respond(200, text=_SAMPLE_DDG_HTML)
    tool = create_web_search_tool()
    out = await tool.func(query="python asyncio tutorial")

    assert "https://realpython.com/async-io-python/" in out
    assert "https://docs.python.org/3/library/asyncio.html" in out
    assert "asyncio" in out.lower()
    # The DDG redirector URL must never leak into the result — only the
    # decoded target, since the agent passes this straight to web_fetch.
    assert "duckduckgo.com/l/" not in out


@pytest.mark.asyncio
@respx.mock
async def test_web_search_respects_limit() -> None:
    respx.get("https://html.duckduckgo.com/html/").respond(200, text=_SAMPLE_DDG_HTML)
    tool = create_web_search_tool()
    out = await tool.func(query="python asyncio tutorial", limit=1)

    assert "realpython.com" in out
    assert "docs.python.org" not in out


@pytest.mark.asyncio
@respx.mock
async def test_web_search_reports_no_results() -> None:
    respx.get("https://html.duckduckgo.com/html/").respond(
        200, text="<html><body>no results div here</body></html>"
    )
    tool = create_web_search_tool()
    out = await tool.func(query="asdkjqwoeiuzxcv")
    assert "No results found" in out


@pytest.mark.asyncio
async def test_web_search_rejects_empty_query() -> None:
    tool = create_web_search_tool()
    out = await tool.func(query="   ")
    assert out.is_error


@pytest.mark.asyncio
@respx.mock
async def test_web_search_reports_http_error() -> None:
    respx.get("https://html.duckduckgo.com/html/").respond(503, text="unavailable")
    tool = create_web_search_tool()
    out = await tool.func(query="anything")
    assert out.is_error
    assert "503" in out.output


@pytest.mark.asyncio
@respx.mock
async def test_web_search_reports_request_error() -> None:
    respx.get("https://html.duckduckgo.com/html/").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    tool = create_web_search_tool()
    out = await tool.func(query="anything")
    assert out.is_error
    assert "search request failed" in out.output


@pytest.mark.asyncio
@respx.mock
async def test_web_search_sends_default_bot_identifying_user_agent() -> None:
    route = respx.get("https://html.duckduckgo.com/html/").respond(200, text=_SAMPLE_DDG_HTML)
    tool = create_web_search_tool()
    await tool.func(query="anything")
    assert route.calls.last.request.headers["user-agent"] == "Nullain-Agent-SDK/0.1 (+web_search)"


@pytest.mark.asyncio
@respx.mock
async def test_web_search_honors_custom_headers_override() -> None:
    route = respx.get("https://html.duckduckgo.com/html/").respond(200, text=_SAMPLE_DDG_HTML)
    tool = create_web_search_tool(headers={"User-Agent": "MyCompanyBot/1.0"})
    await tool.func(query="anything")
    assert route.calls.last.request.headers["user-agent"] == "MyCompanyBot/1.0"


@pytest.mark.asyncio
@respx.mock
async def test_web_search_uses_searxng_when_configured_and_skips_ddg() -> None:
    searxng_route = respx.get("http://searxng.local/search").respond(
        200,
        json={
            "results": [
                {
                    "title": "SearXNG result",
                    "url": "https://example.com/searxng-hit",
                    "content": "a snippet from searxng",
                }
            ]
        },
    )
    ddg_route = respx.get("https://html.duckduckgo.com/html/").respond(200, text=_SAMPLE_DDG_HTML)

    tool = create_web_search_tool(searxng_base_url="http://searxng.local")
    out = await tool.func(query="anything")

    assert searxng_route.called
    assert not ddg_route.called
    assert "https://example.com/searxng-hit" in out
    assert "a snippet from searxng" in out


@pytest.mark.asyncio
@respx.mock
async def test_web_search_falls_back_to_ddg_when_searxng_unreachable() -> None:
    respx.get("http://searxng.local/search").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    ddg_route = respx.get("https://html.duckduckgo.com/html/").respond(200, text=_SAMPLE_DDG_HTML)

    tool = create_web_search_tool(searxng_base_url="http://searxng.local")
    out = await tool.func(query="python asyncio tutorial")

    assert ddg_route.called
    assert "https://realpython.com/async-io-python/" in out


@pytest.mark.asyncio
@respx.mock
async def test_web_search_falls_back_to_ddg_on_searxng_http_error() -> None:
    respx.get("http://searxng.local/search").respond(500, text="internal error")
    ddg_route = respx.get("https://html.duckduckgo.com/html/").respond(200, text=_SAMPLE_DDG_HTML)

    tool = create_web_search_tool(searxng_base_url="http://searxng.local")
    out = await tool.func(query="python asyncio tutorial")

    assert ddg_route.called
    assert "https://realpython.com/async-io-python/" in out


@pytest.mark.asyncio
@respx.mock
async def test_web_search_falls_back_to_ddg_on_searxng_malformed_json() -> None:
    respx.get("http://searxng.local/search").respond(200, text="not json at all")
    ddg_route = respx.get("https://html.duckduckgo.com/html/").respond(200, text=_SAMPLE_DDG_HTML)

    tool = create_web_search_tool(searxng_base_url="http://searxng.local")
    out = await tool.func(query="python asyncio tutorial")

    assert ddg_route.called
    assert "https://realpython.com/async-io-python/" in out


@pytest.mark.asyncio
@respx.mock
async def test_web_search_falls_back_to_ddg_on_searxng_empty_results() -> None:
    respx.get("http://searxng.local/search").respond(200, json={"results": []})
    ddg_route = respx.get("https://html.duckduckgo.com/html/").respond(200, text=_SAMPLE_DDG_HTML)

    tool = create_web_search_tool(searxng_base_url="http://searxng.local")
    out = await tool.func(query="python asyncio tutorial")

    assert ddg_route.called
    assert "https://realpython.com/async-io-python/" in out


@pytest.mark.asyncio
@respx.mock
async def test_web_search_without_searxng_base_url_never_calls_searxng() -> None:
    """Default behavior (searxng_base_url=None) must be identical to
    before this parameter existed — DuckDuckGo only, no attempt to reach
    any SearXNG endpoint at all."""
    ddg_route = respx.get("https://html.duckduckgo.com/html/").respond(200, text=_SAMPLE_DDG_HTML)
    tool = create_web_search_tool()
    await tool.func(query="anything")
    assert ddg_route.called
