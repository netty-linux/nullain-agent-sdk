"""Nullain Tools — web_search: SearXNG (self-hosted meta-search) with a
DuckDuckGo HTML fallback.

Exists so an agent can find real URLs before calling `web_fetch` instead
of guessing a plausible-looking URL from its own (often stale or wrong)
training knowledge — the single biggest source of `web_fetch` 404s in
practice.

Two backends, tried in order:

1. **SearXNG** (opt-in via `searxng_base_url`) — a self-hosted meta-search
   instance queried through its JSON API (`{base_url}/search?format=json`).
   Aggregates multiple upstream engines and isn't itself subject to any
   single engine's bot detection the way scraping one directly is. Any
   failure (unreachable, timeout, malformed response, disabled JSON
   format) falls through to DuckDuckGo rather than erroring the tool call
   — a misconfigured or temporarily-down self-hosted instance must never
   make search unavailable when a working zero-infra fallback exists.
2. **DuckDuckGo HTML** (always available, the prior sole implementation) —
   scrapes the public, robots-permitted HTML endpoint
   (`html.duckduckgo.com/html/`), the same one a browser without
   JavaScript gets. No API key, no server to operate.

Same honest-bot-identifier discipline as `web_fetch` (see its module
docstring) for both — no spoofed browser headers.

`WebSearchProvider` wraps this logic behind the SDK's `SearchProvider` port
(`nullain.ports.search`) as the always-available default adapter: `query`
is exactly the search behavior above, `fetch` retrieves one URL's content
(mirroring `web_fetch`'s plain-httpx path, without web_fetch's Crawl4AI/
Wayback fallback layers — those are that tool's concern, not this port's),
and `index` is a documented no-op since a web-search adapter has no local
store to ingest into. `create_web_search_tool` is unchanged behavior:
it now builds a `WebSearchProvider` internally and its `web_search`
closure delegates to `provider.query(...)`.
"""

import re
from html import unescape
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from nullain.authority import Capability
from nullain.errors import SearchError
from nullain.tools import RegisteredTool, tool
from nullain.tools.result import ToolResult

from nullain_tools.web import html_to_text

_REQUEST_TIMEOUT = 30.0
_SEARXNG_TIMEOUT = 15.0
_USER_AGENT = "Nullain-Agent-SDK/0.1 (+web_search)"
_MAX_RESULTS = 10

# DuckDuckGo's HTML results wrap each result's title in `result__a` and its
# snippet in `result__snippet`, both linking through a `/l/?uddg=<encoded
# target URL>` redirector — never the target URL directly. Matches title and
# snippet as separate passes since they're two independent <a> tags per
# result, not a single nested structure regex could capture in one pass.
_RESULT_LINK = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
_RESULT_SNIPPET = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
_TAG_STRIP = re.compile(r"<[^>]+>")


def _clean_text(html_fragment: str) -> str:
    return unescape(_TAG_STRIP.sub("", html_fragment)).strip()


def _extract_target_url(redirector_href: str) -> str | None:
    """Pull the real target URL out of DuckDuckGo's `/l/?uddg=...`
    redirector link — the HTML endpoint never links directly to results."""
    href = redirector_href if redirector_href.startswith("http") else f"https:{redirector_href}"
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    target = query.get("uddg")
    if not target:
        return None
    return unquote(target[0])


def _format_result(index: int, title: str, url: str, snippet: str) -> str:
    entry = f"{index}. {title}\n   {url}"
    if snippet:
        entry += f"\n   {snippet}"
    return entry


async def _searxng_search(
    client: httpx.AsyncClient, base_url: str, query: str, limit: int
) -> list[str] | None:
    """Query a self-hosted SearXNG instance's JSON API. Returns formatted
    result strings, or None on ANY failure (unreachable, timeout, non-200,
    malformed JSON, JSON format disabled on the instance) — every failure
    mode here is a fall-through signal for the DuckDuckGo backend, never
    a raised exception, so a misconfigured or down SearXNG instance
    degrades to the zero-infra fallback instead of breaking web_search."""
    try:
        resp = await client.get(
            f"{base_url.rstrip('/')}/search",
            params={"q": query, "format": "json"},
            timeout=_SEARXNG_TIMEOUT,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
    except (httpx.HTTPError, ValueError):
        return None

    raw_results: Any = data.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        return None
    result_list = cast("list[Any]", raw_results)

    formatted: list[str] = []
    for item in result_list[:limit]:
        if not isinstance(item, dict):
            continue
        item_dict = cast("dict[Any, Any]", item)
        raw_url = item_dict.get("url")
        if not raw_url or not isinstance(raw_url, str):
            continue
        raw_title = item_dict.get("title")
        title = raw_title if isinstance(raw_title, str) and raw_title else raw_url
        raw_snippet = item_dict.get("content")
        snippet = raw_snippet if isinstance(raw_snippet, str) else ""
        formatted.append(_format_result(len(formatted) + 1, title, raw_url, snippet))

    return formatted or None


def _duckduckgo_search(body: str, limit: int) -> list[str]:
    """Parse DuckDuckGo's HTML results page into formatted result strings."""
    links = _RESULT_LINK.findall(body)
    snippets = _RESULT_SNIPPET.findall(body)

    results: list[str] = []
    for i, (href, title_html) in enumerate(links[:limit]):
        target = _extract_target_url(href)
        if not target:
            continue
        title = _clean_text(title_html)
        snippet = _clean_text(snippets[i]) if i < len(snippets) else ""
        results.append(_format_result(len(results) + 1, title, target, snippet))
    return results


class WebSearchProvider:
    """Default `SearchProvider` (`nullain.ports.search`) adapter: SearXNG
    (opt-in) with a DuckDuckGo HTML fallback for `query`, plain-httpx GET
    for `fetch`. Always available — no API key, no server required for the
    DuckDuckGo path. Has no local index, so `index` is a no-op.

    This is the same two-backend logic `create_web_search_tool` has always
    used; wrapping it as a `SearchProvider` adapter here lets the tool
    delegate to it instead of duplicating the request logic, and lets
    future callers (or a contract-test suite) exercise the same behavior
    directly, without going through the `@tool`-wrapped closure.
    """

    def __init__(
        self, headers: dict[str, str] | None = None, searxng_base_url: str | None = None
    ) -> None:
        self._headers = {
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            **(headers or {}),
        }
        self._searxng_base_url = searxng_base_url

    async def index(self, content: str, *, source: str) -> None:
        """No-op: a web-search adapter has no local store to ingest into."""
        return None

    async def query(self, text: str, *, limit: int = 5) -> str:
        """Search the web via SearXNG (if configured) then DuckDuckGo.

        Raises:
            SearchError: on an empty query or a request failure.
        """
        text = text.strip()
        if not text:
            raise SearchError("query must not be empty.")
        limit = max(1, min(limit, _MAX_RESULTS))

        async with httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT, follow_redirects=True, headers=self._headers
        ) as client:
            if self._searxng_base_url:
                searxng_results = await _searxng_search(client, self._searxng_base_url, text, limit)
                if searxng_results is not None:
                    return "\n\n".join(searxng_results)

            try:
                response = await client.get("https://html.duckduckgo.com/html/", params={"q": text})
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                raise SearchError(f"search failed with HTTP {err.response.status_code}.") from err
            except httpx.RequestError as err:
                raise SearchError(f"search request failed: {err}.") from err

            results = _duckduckgo_search(response.text, limit)

        if not results:
            return f"No results found for '{text}'."
        return "\n\n".join(results)

    async def fetch(self, source: str) -> str:
        """Fetch `source`'s content via plain HTTP GET (HTML is stripped to
        text). Mirrors `web_fetch`'s plain-httpx path; does not layer on
        `web_fetch`'s Crawl4AI or Wayback Machine fallbacks — those belong
        to that tool, not to this port's minimal contract.

        Raises:
            SearchError: on a request failure.
        """
        async with httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT, follow_redirects=True, headers=self._headers
        ) as client:
            try:
                response = await client.get(source)
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                raise SearchError(
                    f"fetch failed with HTTP {err.response.status_code} for '{source}'."
                ) from err
            except httpx.RequestError as err:
                raise SearchError(f"fetch request failed for '{source}': {err}.") from err

        content_type = response.headers.get("content-type", "")
        body = response.text
        if "html" in content_type.lower():
            body = html_to_text(body)
        return body or "(empty page)"


def create_web_search_tool(
    headers: dict[str, str] | None = None, searxng_base_url: str | None = None
) -> RegisteredTool:
    """Build the ``web_search`` tool.

    Args:
        headers: HTTP headers sent on every request, overriding the
            default honest bot-identifying ``User-Agent``. Mirrors
            ``create_web_fetch_tool``'s ``headers`` parameter.
        searxng_base_url: Base URL of a self-hosted SearXNG instance
            (e.g. ``http://searxng:8080`` on an internal Docker network).
            When set, SearXNG is tried first and DuckDuckGo is the
            fallback on any failure; when None (the default), DuckDuckGo
            is used directly — identical behavior to before this
            parameter existed.
    """
    provider = WebSearchProvider(headers=headers, searxng_base_url=searxng_base_url)

    @tool(
        name="web_search",
        description=(
            "Search the web and return a list of real result URLs with titles and "
            "snippets. ALWAYS use this before web_fetch when you don't already have "
            "a known-correct URL — never guess or construct a URL from memory, "
            "since training knowledge about specific pages/slugs is frequently "
            "stale or wrong and produces 404s. Pass the URLs this tool returns "
            "directly to web_fetch."
        ),
        read_only=True,
        requires=frozenset({Capability.READ, Capability.NETWORK}),
    )
    async def web_search(query: str, limit: int = 5) -> str | ToolResult:
        try:
            return await provider.query(query, limit=limit)
        except SearchError as err:
            return ToolResult(output=f"Error: {err}", is_error=True, error_type="ToolError")

    return web_search


__all__ = ["WebSearchProvider", "create_web_search_tool"]
