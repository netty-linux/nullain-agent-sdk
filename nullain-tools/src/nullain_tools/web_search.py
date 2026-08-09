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
"""

import re
from html import unescape
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from nullain.authority import Capability
from nullain.tools import RegisteredTool, tool
from nullain.tools.result import ToolResult

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
    request_headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        **(headers or {}),
    }

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
        query = query.strip()
        if not query:
            return ToolResult(
                output="Error: query must not be empty.", is_error=True, error_type="ToolError"
            )
        limit = max(1, min(limit, _MAX_RESULTS))

        async with httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT, follow_redirects=True, headers=request_headers
        ) as client:
            if searxng_base_url:
                searxng_results = await _searxng_search(client, searxng_base_url, query, limit)
                if searxng_results is not None:
                    return "\n\n".join(searxng_results)

            try:
                response = await client.get(
                    "https://html.duckduckgo.com/html/", params={"q": query}
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                return ToolResult(
                    output=f"Error: search failed with HTTP {err.response.status_code}.",
                    is_error=True,
                    error_type="ToolError",
                )
            except httpx.RequestError as err:
                return ToolResult(
                    output=f"Error: search request failed: {err}.",
                    is_error=True,
                    error_type="ToolError",
                )

            results = _duckduckgo_search(response.text, limit)

        if not results:
            return f"No results found for '{query}'."
        return "\n\n".join(results)

    return web_search


__all__ = ["create_web_search_tool"]
