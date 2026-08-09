"""Unit tests for web_fetch's actual HTTP path (respx-mocked, no network).

test_p2_features.py already covers the offline paths (non-http:// URL
rejection, html_to_text tag-stripping in isolation); this file covers
web_fetch's real fetch logic — success, HTTP errors, request errors,
content-type-driven HTML conversion, response truncation, and the
Wayback Machine fallback on bot-block statuses — none of which were
previously exercised at all.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import respx
from nullain_tools.web import (
    _MAX_RESPONSE_CHARS,  # type: ignore[reportPrivateUsage]
    create_web_fetch_tool,
)

_RECENT_TIMESTAMP = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y%m%d%H%M%S")
_STALE_TIMESTAMP = (datetime.now(UTC) - timedelta(days=30)).strftime("%Y%m%d%H%M%S")


def _wayback_available_response(*, timestamp: str, snapshot_url: str) -> dict[str, Any]:
    return {
        "url": "example.com/blocked",
        "archived_snapshots": {
            "closest": {
                "status": "200",
                "available": True,
                "url": snapshot_url,
                "timestamp": timestamp,
            }
        },
    }


def _wayback_unavailable_response() -> dict[str, Any]:
    return {"url": "example.com/blocked", "archived_snapshots": {}}


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_returns_plain_text_body_unchanged() -> None:
    respx.get("https://example.com/data.txt").respond(
        200, text="raw plain text", headers={"content-type": "text/plain"}
    )
    tool = create_web_fetch_tool()
    out = await tool.func(url="https://example.com/data.txt")
    assert out == "raw plain text"


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_converts_html_content_type_to_text() -> None:
    respx.get("https://example.com/page").respond(
        200,
        text="<html><body><p>Hello</p></body></html>",
        headers={"content-type": "text/html; charset=utf-8"},
    )
    tool = create_web_fetch_tool()
    out = await tool.func(url="https://example.com/page")
    assert out == "Hello"
    assert "<p>" not in out


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_reports_http_error_status() -> None:
    respx.get("https://example.com/missing").respond(404, text="not found")
    tool = create_web_fetch_tool()
    out = await tool.func(url="https://example.com/missing")
    assert out.is_error
    assert "404" in out.output


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_reports_request_error() -> None:
    respx.get("https://unreachable.example.com/").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    tool = create_web_fetch_tool()
    out = await tool.func(url="https://unreachable.example.com/")
    assert out.is_error
    assert "Request failed" in out.output


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_truncates_oversized_response() -> None:
    big_body = "x" * (_MAX_RESPONSE_CHARS + 1000)
    respx.get("https://example.com/big").respond(
        200, text=big_body, headers={"content-type": "text/plain"}
    )
    tool = create_web_fetch_tool()
    out = await tool.func(url="https://example.com/big")
    assert len(out) < len(big_body)
    assert out.endswith("...[truncated]")


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_reports_empty_body() -> None:
    respx.get("https://example.com/empty").respond(
        200, text="", headers={"content-type": "text/plain"}
    )
    tool = create_web_fetch_tool()
    out = await tool.func(url="https://example.com/empty")
    assert out == "Error: Empty response body."


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_sends_default_bot_identifying_user_agent() -> None:
    route = respx.get("https://example.com/ua").respond(
        200, text="ok", headers={"content-type": "text/plain"}
    )
    tool = create_web_fetch_tool()
    await tool.func(url="https://example.com/ua")
    assert route.calls.last.request.headers["user-agent"] == "Nullain-Agent-SDK/0.1 (+web_fetch)"


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_honors_custom_headers_override() -> None:
    route = respx.get("https://example.com/ua").respond(
        200, text="ok", headers={"content-type": "text/plain"}
    )
    tool = create_web_fetch_tool(headers={"User-Agent": "MyCompanyBot/1.0"})
    await tool.func(url="https://example.com/ua")
    assert route.calls.last.request.headers["user-agent"] == "MyCompanyBot/1.0"


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_404_falls_through_without_wayback() -> None:
    """404 (and other non-401/403/429 statuses) is a genuine "not found" —
    triggering the Wayback fallback would mislead the model into treating a
    plain broken URL/typo as a bot block, so it must only fire for the
    specific codes sites actually use for anti-scraping/rate-limiting."""
    respx.get("https://example.com/missing").respond(404, text="not found")
    tool = create_web_fetch_tool()
    out = await tool.func(url="https://example.com/missing")
    assert "404" in out.output
    assert "Wayback" not in out.output


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_401_falls_back_to_recent_wayback_snapshot() -> None:
    respx.get("https://example.com/paywalled").respond(401, text="unauthorized")
    respx.get("http://archive.org/wayback/available").respond(
        200,
        json=_wayback_available_response(
            timestamp=_RECENT_TIMESTAMP,
            snapshot_url="http://web.archive.org/web/20260101000000/https://example.com/paywalled",
        ),
    )
    respx.get("http://web.archive.org/web/20260101000000/https://example.com/paywalled").respond(
        200, text="archived content", headers={"content-type": "text/plain"}
    )

    tool = create_web_fetch_tool()
    out = await tool.func(url="https://example.com/paywalled")
    assert isinstance(out, str)
    assert "archived content" in out
    assert "401" in out
    assert "not the current live page" in out


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_403_and_429_also_trigger_wayback_fallback() -> None:
    respx.get("https://example.com/forbidden").respond(403, text="forbidden")
    respx.get("https://example.com/ratelimited").respond(429, text="too many requests")
    respx.get("http://archive.org/wayback/available").respond(
        200,
        json=_wayback_available_response(
            timestamp=_RECENT_TIMESTAMP, snapshot_url="http://web.archive.org/web/20260101000000/x"
        ),
    )
    respx.get("http://web.archive.org/web/20260101000000/x").respond(
        200, text="ok", headers={"content-type": "text/plain"}
    )

    tool = create_web_fetch_tool()
    out_403 = await tool.func(url="https://example.com/forbidden")
    assert "403" in out_403

    out_429 = await tool.func(url="https://example.com/ratelimited")
    assert "429" in out_429


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_stale_wayback_snapshot_triggers_fresh_capture() -> None:
    """A snapshot older than 7 days must trigger /save/ before being used —
    otherwise a time-sensitive blocked page could silently serve month-old
    data with no signal to the caller that a refresh was even attempted."""
    respx.get("https://example.com/blocked").respond(403, text="forbidden")
    save_route = respx.get("https://web.archive.org/save/https://example.com/blocked").respond(
        200, text="capture triggered"
    )
    availability_route = respx.get("http://archive.org/wayback/available").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_wayback_available_response(
                    timestamp=_STALE_TIMESTAMP,
                    snapshot_url="http://web.archive.org/web/20260101000000/stale",
                ),
            ),
            httpx.Response(
                200,
                json=_wayback_available_response(
                    timestamp=_RECENT_TIMESTAMP,
                    snapshot_url="http://web.archive.org/web/20260201000000/fresh",
                ),
            ),
        ]
    )
    respx.get("http://web.archive.org/web/20260201000000/fresh").respond(
        200, text="freshly captured content", headers={"content-type": "text/plain"}
    )

    tool = create_web_fetch_tool()
    out = await tool.func(url="https://example.com/blocked")

    assert save_route.called
    assert availability_route.call_count == 2
    assert "freshly captured content" in out


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_no_wayback_snapshot_reports_clear_error() -> None:
    respx.get("https://example.com/never-archived").respond(403, text="forbidden")
    respx.get("http://archive.org/wayback/available").respond(
        200, json=_wayback_unavailable_response()
    )

    tool = create_web_fetch_tool()
    out = await tool.func(url="https://example.com/never-archived")
    assert out.is_error
    assert "403" in out.output
    assert "no Wayback Machine snapshot is available" in out.output


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_wayback_availability_check_itself_failing_reports_clear_error() -> None:
    """The Wayback fallback is best-effort — if archive.org's own
    availability API is unreachable, the caller must still get a clear
    error, not an unrelated exception bubbling up."""
    respx.get("https://example.com/blocked").respond(403, text="forbidden")
    respx.get("http://archive.org/wayback/available").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    tool = create_web_fetch_tool()
    out = await tool.func(url="https://example.com/blocked")
    assert out.is_error
    assert "403" in out.output
