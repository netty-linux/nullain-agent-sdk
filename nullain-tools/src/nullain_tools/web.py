"""Nullain Tools — web_fetch: retrieve a URL as raw markdown text.

Two fetch strategies, tried in order:

1. **Crawl4AI/Playwright** (opt-in via `use_crawl4ai=True`) — a real
   headless browser renders the page, executes JavaScript, and returns
   already-clean Markdown. This solves the class of pages plain `httpx`
   cannot: content that only appears after JS runs, and — as a side
   effect of looking like a real browser rather than a bare HTTP client —
   some (not all) sites that block bare `httpx` requests. `playwright`/
   `crawl4ai` are optional dependencies, resolved lazily so the base
   install stays light.
2. **Plain httpx** (always available, the prior sole implementation) —
   direct GET + a minimal regex tag-stripping pass to plain text. No
   extraction model, no browser.

On a bot-block response (401/403/429) from EITHER strategy, falls back to
the Wayback Machine (archive.org) instead of just telling the agent to
give up: fetches the most recent snapshot, and if it's stale (>7 days)
triggers a fresh capture first. This is a deterministic fallback chain,
not something the agent has to remember to do itself via a separate tool
call — and it exists because Crawl4AI's browser rendering does NOT solve
bot-detection blocks the way it solves JS-heavy pages; a real browser can
still be fingerprinted and blocked, so both fallback layers matter for
different failure modes.
"""

import contextlib
import importlib
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from nullain.authority import Capability
from nullain.tools import RegisteredTool, tool
from nullain.tools.result import ToolResult

_MAX_RESPONSE_CHARS = 50_000
_REQUEST_TIMEOUT = 30.0
_WAYBACK_TIMEOUT = 20.0
_CRAWL4AI_TIMEOUT_MS = 30_000
_USER_AGENT = "Nullain-Agent-SDK/0.1 (+web_fetch)"

#: Status codes a site commonly returns for bot detection, paywalls, or
#: rate limiting — distinct from a genuine "this URL is broken" error, so
#: the agent can tell "try a different source" apart from "fix the URL".
_BOT_BLOCK_STATUS_CODES = frozenset({401, 403, 429})

#: A snapshot older than this triggers a fresh Wayback Machine capture
#: before falling back to it, so a blocked page doesn't silently serve
#: week-old (or older) content for something time-sensitive.
_STALE_SNAPSHOT_AGE = timedelta(days=7)

# Block-level tags that should introduce a line break when stripped.
_BLOCK_TAGS = re.compile(r"</?(p|div|br|li|h[1-6]|tr|section|article|header|footer)[^>]*>", re.I)
_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.DOTALL)
_TAG_STRIP = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"[ \t]+")


def _parse_wayback_timestamp(timestamp: str) -> datetime | None:
    try:
        return datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


async def _wayback_snapshot(client: httpx.AsyncClient, url: str) -> tuple[str, str] | None:
    """Return `(snapshot_url, timestamp)` of the most recent Wayback Machine
    capture of `url`, or None if archive.org has never captured it (or the
    availability check itself failed — this is a best-effort fallback, not
    a hard dependency)."""
    try:
        resp = await client.get(
            "http://archive.org/wayback/available",
            params={"url": url},
            timeout=_WAYBACK_TIMEOUT,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
    except (httpx.HTTPError, ValueError):
        return None

    closest = data.get("archived_snapshots", {}).get("closest")
    if not closest or not closest.get("available"):
        return None
    return closest["url"], closest["timestamp"]


async def _wayback_trigger_capture(client: httpx.AsyncClient, url: str) -> None:
    """Ask archive.org to capture a fresh snapshot of `url`. Best-effort:
    the /save/ endpoint rate-limits aggressively and capture is async on
    their end anyway (no guarantee a snapshot exists immediately after this
    returns) — failures here are silently absorbed by the caller falling
    back to whatever snapshot (possibly stale, possibly none) it already
    has, rather than failing the whole fetch over a save-endpoint hiccup."""
    with contextlib.suppress(httpx.HTTPError):
        await client.get(f"https://web.archive.org/save/{url}", timeout=_WAYBACK_TIMEOUT)


async def _fetch_via_wayback(
    client: httpx.AsyncClient, url: str, *, original_status: int
) -> str | ToolResult:
    """Fallback path when the live URL returned a bot-block status.
    Fetches the closest Wayback Machine snapshot; if it's older than
    `_STALE_SNAPSHOT_AGE`, triggers a fresh capture first and re-checks.
    Always states the snapshot's date so the caller knows how current the
    content is — never silently serves old data as if it were live."""
    snapshot = await _wayback_snapshot(client, url)

    if snapshot is not None:
        _, timestamp = snapshot
        captured_at = _parse_wayback_timestamp(timestamp)
        is_stale = captured_at is not None and (
            datetime.now(UTC) - captured_at > _STALE_SNAPSHOT_AGE
        )
        if is_stale:
            await _wayback_trigger_capture(client, url)
            refreshed = await _wayback_snapshot(client, url)
            if refreshed is not None:
                snapshot = refreshed

    if snapshot is None:
        return ToolResult(
            output=(
                f"Error: HTTP {original_status} for '{url}', and no Wayback Machine "
                "snapshot is available either — use a different source instead."
            ),
            is_error=True,
            error_type="ToolError",
        )

    snapshot_url, timestamp = snapshot
    captured_at = _parse_wayback_timestamp(timestamp)
    date_label = captured_at.date().isoformat() if captured_at else timestamp

    try:
        resp = await client.get(snapshot_url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return ToolResult(
            output=(
                f"Error: HTTP {original_status} for '{url}'; found a Wayback Machine "
                f"snapshot from {date_label} but failed to fetch it: {exc}."
            ),
            is_error=True,
            error_type="ToolError",
        )

    content_type = resp.headers.get("content-type", "")
    body = resp.text
    if "html" in content_type.lower():
        body = html_to_text(body)
    if len(body) > _MAX_RESPONSE_CHARS:
        body = body[:_MAX_RESPONSE_CHARS] + "\n...[truncated]"

    prefix = (
        f"[Live fetch of '{url}' returned HTTP {original_status}; this is a Wayback "
        f"Machine snapshot from {date_label}, not the current live page — cite the "
        f"date when using this content.]\n\n"
    )
    return prefix + (body or "(empty page)")


class _Crawl4AIResult:
    """Outcome of a Crawl4AI fetch attempt — three distinct cases the
    caller must handle differently, not collapsible into `str | None`:
    genuine content, a bot-block status (route to Wayback), or a
    strategy-level failure (route to plain httpx)."""

    __slots__ = ("failed", "markdown", "status_code")

    def __init__(
        self, *, markdown: str | None = None, status_code: int | None = None, failed: bool = False
    ) -> None:
        self.markdown = markdown
        self.status_code = status_code
        self.failed = failed


def _import_crawl4ai() -> Any:
    try:
        return importlib.import_module("crawl4ai")
    except ImportError as exc:
        raise ImportError(
            "Crawl4AI-based web_fetch requires the 'crawl' extra "
            "(includes Playwright): pip install nullain-sdk[crawl] "
            "&& playwright install chromium"
        ) from exc


async def _fetch_via_crawl4ai(url: str) -> _Crawl4AIResult:
    """Render `url` in a real headless browser and return clean Markdown.
    Never raises — not for a missing 'crawl' extra, and not for a normal
    fetch failure (timeout, browser crash, DNS error, non-2xx/-blocked
    status). All of those come back as `failed=True` so the caller falls
    through to the plain-httpx strategy, which the Wayback fallback then
    layers on top of either way. The caller (`web_fetch`) only ever opts
    into this path explicitly (`use_crawl4ai=True`); a missing extra must
    degrade gracefully there, not crash a request the caller otherwise
    expected plain httpx to still be able to serve."""
    try:
        crawl4ai_mod = _import_crawl4ai()
        crawler_cls: Any = crawl4ai_mod.AsyncWebCrawler
        run_config_cls: Any = crawl4ai_mod.CrawlerRunConfig

        async with crawler_cls() as crawler:
            result = await crawler.arun(
                url=url, config=run_config_cls(page_timeout=_CRAWL4AI_TIMEOUT_MS)
            )
    except Exception:
        return _Crawl4AIResult(failed=True)

    # arun() returns a CrawlResultContainer for a single URL (no
    # deep-crawl config here), which behaves like a length-1 sequence of
    # CrawlResult — indexing is the documented way to get the one result.
    single_result: Any = result[0] if hasattr(result, "__getitem__") else result

    status_code = getattr(single_result, "status_code", None)
    if not getattr(single_result, "success", False):
        if isinstance(status_code, int) and status_code in _BOT_BLOCK_STATUS_CODES:
            return _Crawl4AIResult(status_code=status_code)
        return _Crawl4AIResult(failed=True)

    markdown = str(getattr(single_result, "markdown", "") or "")
    if not markdown.strip():
        return _Crawl4AIResult(failed=True)
    return _Crawl4AIResult(markdown=markdown)


def html_to_text(html: str) -> str:
    """Minimal HTML → plain text conversion (no extraction model)."""
    text = _SCRIPT_STYLE.sub("", html)
    text = _BLOCK_TAGS.sub("\n", text)
    text = _TAG_STRIP.sub("", text)
    text = (
        text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    )
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def create_web_fetch_tool(
    headers: dict[str, str] | None = None, use_crawl4ai: bool = False
) -> RegisteredTool:
    """Build the ``web_fetch`` tool.

    Args:
        headers: HTTP headers sent on every request, overriding the
            default ``User-Agent``/``Accept``/``Accept-Language``. When
            None, the honest bot-identifying defaults are used (see
            ``nullain.config.WebFetchConfig``'s docstring for why those
            are the default rather than a spoofed browser User-Agent).
        use_crawl4ai: When True, tries a real headless browser (Crawl4AI/
            Playwright) first for every fetch, falling back to plain
            httpx on any Crawl4AI-level failure (not installed, browser
            crash, timeout). Requires the ``crawl`` extra and a Chromium
            install (``playwright install chromium``) at runtime — an
            operator opt-in given the image-size/memory cost, not a
            silent default. When False (the default), behavior is
            identical to before this parameter existed.
    """
    request_headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        **(headers or {}),
    }

    @tool(
        name="web_fetch",
        description=(
            "Fetch a URL and return its content as plain text (HTML is stripped "
            "to text; no summarization). Useful for reading documentation or APIs. "
            "If the site blocks automated requests (401/403/429), this automatically "
            "falls back to the Wayback Machine's most recent snapshot — no need to "
            "retry or give up on your own; the result will state the snapshot's date "
            "when that fallback was used."
        ),
        read_only=True,
        requires=frozenset({Capability.READ, Capability.NETWORK}),
    )
    async def web_fetch(url: str) -> str | ToolResult:
        if not url.startswith(("http://", "https://")):
            return ToolResult(
                output=f"Error: URL must start with http:// or https:// (got '{url}').",
                is_error=True,
                error_type="ToolError",
            )

        async with httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
            headers=request_headers,
        ) as client:
            if use_crawl4ai:
                crawl_result = await _fetch_via_crawl4ai(url)
                if crawl_result.markdown is not None:
                    body = crawl_result.markdown
                    if len(body) > _MAX_RESPONSE_CHARS:
                        body = body[:_MAX_RESPONSE_CHARS] + "\n...[truncated]"
                    return body
                if crawl_result.status_code in _BOT_BLOCK_STATUS_CODES:
                    status_code = crawl_result.status_code
                    assert status_code is not None
                    return await _fetch_via_wayback(client, url, original_status=status_code)
                # crawl_result.failed (not installed, timeout, crash, non-
                # bot-block error): fall through to plain httpx below,
                # same as if use_crawl4ai had never been set.

            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                status = err.response.status_code
                if status in _BOT_BLOCK_STATUS_CODES:
                    return await _fetch_via_wayback(client, url, original_status=status)
                return ToolResult(
                    output=f"Error: HTTP {status} for '{url}'.",
                    is_error=True,
                    error_type="ToolError",
                )
            except httpx.RequestError as err:
                return ToolResult(
                    output=f"Error: Request failed for '{url}': {err}.",
                    is_error=True,
                    error_type="ToolError",
                )

            content_type = response.headers.get("content-type", "")
            body = response.text
            if "html" in content_type.lower():
                body = html_to_text(body)

            if len(body) > _MAX_RESPONSE_CHARS:
                body = body[:_MAX_RESPONSE_CHARS] + "\n...[truncated]"
            return body or "Error: Empty response body."

    return web_fetch


__all__ = ["create_web_fetch_tool", "html_to_text"]
