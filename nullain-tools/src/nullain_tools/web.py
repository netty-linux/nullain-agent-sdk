"""Nullain Tools — web_fetch: retrieve a URL as raw markdown text.

No extraction model is involved: HTML is converted to plain text with a
minimal tag-stripping pass. This gives the agent page content to reason over
without an LLM summarization round-trip, matching the parity-feature scope.

On a bot-block response (401/403/429), falls back to the Wayback Machine
(archive.org) instead of just telling the agent to give up: fetches the
most recent snapshot, and if it's stale (>7 days) triggers a fresh capture
first. This is a deterministic fallback chain, not something the agent has
to remember to do itself via a separate tool call.
"""

import contextlib
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


def create_web_fetch_tool(headers: dict[str, str] | None = None) -> RegisteredTool:
    """Build the ``web_fetch`` tool.

    Args:
        headers: HTTP headers sent on every request, overriding the
            default ``User-Agent``/``Accept``/``Accept-Language``. When
            None, the honest bot-identifying defaults are used (see
            ``nullain.config.WebFetchConfig``'s docstring for why those
            are the default rather than a spoofed browser User-Agent).
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
