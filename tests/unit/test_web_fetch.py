"""Unit tests for web_fetch's actual HTTP path (respx-mocked, no network).

test_p2_features.py already covers the offline paths (non-http:// URL
rejection, html_to_text tag-stripping in isolation); this file covers
web_fetch's real fetch logic — success, HTTP errors, request errors,
content-type-driven HTML conversion, and response truncation — none of
which were previously exercised at all.
"""

import httpx
import pytest
import respx
from nullain_tools.web import (
    _MAX_RESPONSE_CHARS,  # type: ignore[reportPrivateUsage]
    create_web_fetch_tool,
)


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
