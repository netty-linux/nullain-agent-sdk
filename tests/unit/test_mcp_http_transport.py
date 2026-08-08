"""Unit tests for HttpTransport — MCP over streamable HTTP.

Exercises the transport directly (request/response framing, SSE parsing,
error propagation) with `respx`, and end-to-end through MCPClient +
register_mcp_tools with a mocked Composio-shaped endpoint, matching the
pattern already used for OpenAICompatibleProvider (test_openai_compat_provider.py).
"""

import json

import httpx
import pytest
import respx
from nullain.errors import MCPTransportError
from nullain.mcp import HttpTransport, MCPClient, register_mcp_tools
from nullain.mcp.protocol import JSONRPCNotification, JSONRPCRequest
from nullain.tools import ToolRegistry

URL = "https://mcp.example.com/mcp"


def _rpc_response(request_id: int, result: object) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result})


def _sse_framed(payload: str) -> str:
    return f"data: {payload}\n\n"


# ---------------------------------------------------------------------------
# Transport-level: request/response framing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_send_request_bare_json_body() -> None:
    respx.post(URL).respond(200, json={"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
    transport = HttpTransport(url=URL)
    raw = await transport.send_request(JSONRPCRequest(id=1, method="initialize", params={}))
    assert json.loads(raw) == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
    await transport.close()


@pytest.mark.asyncio
@respx.mock
async def test_send_request_sse_framed_body() -> None:
    """Composio's streamable-HTTP endpoint frames the JSON-RPC envelope as an
    SSE `data:` line rather than a bare JSON body — the transport must unwrap
    it the same way the app's own `_parse_sse_jsonrpc` does."""
    body = _sse_framed(_rpc_response(1, {"tools": []}))
    respx.post(URL).respond(200, text=body, headers={"content-type": "text/event-stream"})
    transport = HttpTransport(url=URL)
    raw = await transport.send_request(JSONRPCRequest(id=1, method="tools/list", params={}))
    assert json.loads(raw) == {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
    await transport.close()


@pytest.mark.asyncio
@respx.mock
async def test_send_request_sse_multiple_data_lines_takes_last() -> None:
    body = (
        _sse_framed("[DONE]")  # sentinel-only line, must be skipped
        + _sse_framed(_rpc_response(2, {"final": True}))
    )
    respx.post(URL).respond(200, text=body)
    transport = HttpTransport(url=URL)
    raw = await transport.send_request(JSONRPCRequest(id=2, method="tools/list", params={}))
    assert json.loads(raw) == {"jsonrpc": "2.0", "id": 2, "result": {"final": True}}
    await transport.close()


@pytest.mark.asyncio
@respx.mock
async def test_send_request_http_error_raises_mcp_transport_error() -> None:
    respx.post(URL).respond(401, json={"error": "Authorization required"})
    transport = HttpTransport(url=URL, headers={"x-consumer-api-key": "bad-key"})
    with pytest.raises(MCPTransportError, match="HTTP 401"):
        await transport.send_request(JSONRPCRequest(id=1, method="initialize", params={}))
    await transport.close()


@pytest.mark.asyncio
@respx.mock
async def test_send_request_network_error_raises_mcp_transport_error() -> None:
    respx.post(URL).mock(side_effect=httpx.ConnectError("connection refused"))
    transport = HttpTransport(url=URL)
    with pytest.raises(MCPTransportError, match="MCP HTTP request failed"):
        await transport.send_request(JSONRPCRequest(id=1, method="initialize", params={}))
    await transport.close()


@pytest.mark.asyncio
@respx.mock
async def test_send_notification_posts_and_discards_response() -> None:
    route = respx.post(URL).respond(200, json={"jsonrpc": "2.0"})
    transport = HttpTransport(url=URL)
    await transport.send_notification(
        JSONRPCNotification(method="notifications/initialized", params={})
    )
    assert route.called
    await transport.close()


@pytest.mark.asyncio
@respx.mock
async def test_extra_headers_forwarded_on_every_request() -> None:
    route = respx.post(URL).respond(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})
    transport = HttpTransport(url=URL, headers={"x-consumer-api-key": "secret-key"})
    await transport.send_request(JSONRPCRequest(id=1, method="initialize", params={}))
    assert route.calls.last.request.headers["x-consumer-api-key"] == "secret-key"
    await transport.close()


def test_empty_url_rejected() -> None:
    with pytest.raises(MCPTransportError, match="non-empty url"):
        HttpTransport(url="")


# ---------------------------------------------------------------------------
# End-to-end through MCPClient + register_mcp_tools (Composio-shaped server)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_register_mcp_tools_over_http_transport() -> None:
    """Full handshake -> tools/list -> registry population, mirroring how
    agent/bridge.py in nullain-agent registers Composio's tools (Fase 0,
    docs/FUSION_PLAN.md)."""

    def _dispatch(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = body["method"]
        if method == "notifications/initialized":
            # Fire-and-forget notification: no id, no JSON-RPC envelope reply
            # expected — send_notification discards the body regardless.
            return httpx.Response(200, json={"jsonrpc": "2.0"})
        rid = body["id"]
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {"name": "composio", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "GITHUB_STAR_REPO",
                        "description": "Star a GitHub repository",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"repo": {"type": "string"}},
                            "required": ["repo"],
                        },
                    }
                ]
            }
        else:
            raise AssertionError(f"unexpected method {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": rid, "result": result})

    respx.post(URL).mock(side_effect=_dispatch)

    transport = HttpTransport(url=URL, headers={"x-consumer-api-key": "k"})
    client = MCPClient(transport, name="composio")
    registry = ToolRegistry()

    registered = await register_mcp_tools(registry, client)

    assert registered == ["mcp__composio__GITHUB_STAR_REPO"]
    tool = registry.get_tool("mcp__composio__GITHUB_STAR_REPO")
    assert tool.spec.function.parameters["required"] == ["repo"]
    await transport.close()


@pytest.mark.asyncio
@respx.mock
async def test_register_mcp_tools_surfaces_auth_failure() -> None:
    """An invalid Composio consumer key (HTTP 401) must propagate as a clean
    MCPTransportError rather than crashing the caller with a raw httpx error
    — agent/bridge.py relies on this to degrade gracefully (Fase 0)."""
    respx.post(URL).respond(401, json={"error": "Authorization required"})
    transport = HttpTransport(url=URL, headers={"x-consumer-api-key": "bad"})
    client = MCPClient(transport, name="composio")
    registry = ToolRegistry()

    with pytest.raises(MCPTransportError, match="HTTP 401"):
        await register_mcp_tools(registry, client)
    await transport.close()
