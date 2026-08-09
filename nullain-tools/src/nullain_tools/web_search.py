"""Nullain Tools — web_search: DuckDuckGo HTML search results.

Exists so an agent can find real URLs before calling `web_fetch` instead
of guessing a plausible-looking URL from its own (often stale or wrong)
training knowledge — the single biggest source of `web_fetch` 404s in
practice. No API key: scrapes DuckDuckGo's public, robots-permitted HTML
endpoint (`html.duckduckgo.com/html/`), the same one a browser without
JavaScript gets. Same honest-bot-identifier discipline as `web_fetch`
(see its module docstring) — no spoofed browser headers.
"""

import re
from html import unescape
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from nullain.authority import Capability
from nullain.tools import RegisteredTool, tool
from nullain.tools.result import ToolResult

_REQUEST_TIMEOUT = 30.0
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


def create_web_search_tool(headers: dict[str, str] | None = None) -> RegisteredTool:
    """Build the ``web_search`` tool.

    Args:
        headers: HTTP headers sent on every request, overriding the
            default honest bot-identifying ``User-Agent``. Mirrors
            ``create_web_fetch_tool``'s ``headers`` parameter.
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

        try:
            async with httpx.AsyncClient(
                timeout=_REQUEST_TIMEOUT, follow_redirects=True, headers=request_headers
            ) as client:
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

        body = response.text
        links = _RESULT_LINK.findall(body)
        snippets = _RESULT_SNIPPET.findall(body)

        results: list[str] = []
        for i, (href, title_html) in enumerate(links[:limit]):
            target = _extract_target_url(href)
            if not target:
                continue
            title = _clean_text(title_html)
            snippet = _clean_text(snippets[i]) if i < len(snippets) else ""
            entry = f"{len(results) + 1}. {title}\n   {target}"
            if snippet:
                entry += f"\n   {snippet}"
            results.append(entry)

        if not results:
            return f"No results found for '{query}'."
        return "\n\n".join(results)

    return web_search


__all__ = ["create_web_search_tool"]
