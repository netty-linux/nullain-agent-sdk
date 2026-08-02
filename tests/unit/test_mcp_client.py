"""Unit tests for the MCP client (JSON-RPC over an in-memory fake transport).

The fake transport speaks the same newline-delimited JSON-RPC framing as
:class:`~nullain.mcp.transport.StdioTransport`, so these tests exercise the
client's handshake, tool-listing, tool-calling, and registry integration
logic fully offline. A separate test spawns a real Python subprocess acting as
a minimal MCP server to verify :class:`StdioTransport` end-to-end.
"""

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from nullain.errors import MCPProtocolError, MCPTransportError, ToolPermissionError
from nullain.llm.types import ToolCall
from nullain.mcp import MCPClient, StdioTransport, register_mcp_tools
from nullain.mcp.protocol import JSONRPCNotification, JSONRPCRequest
from nullain.tools import ToolRegistry
from nullain.tools.permissions import PermissionLevel


class FakeTransport:
    """In-memory MCP transport with scripted responses.

    ``responses`` maps a JSON-RPC method name to either a list of result
    payloads (dequeued in order) or an exception to raise. Requests and
    notifications sent by the client are recorded for assertions.
    """

    def __init__(self, responses: dict[str, list[Any]]) -> None:
        self._responses = {k: list(v) for k, v in responses.items()}
        self.requests: list[JSONRPCRequest] = []
        self.notifications: list[JSONRPCNotification] = []
        self.closed = False

    async def start(self) -> None:
        """No-op for the in-memory transport."""

    async def send_request(self, request: JSONRPCRequest) -> str:
        self.requests.append(request)
        queue = self._responses.get(request.method)
        if queue is None:
            raise MCPTransportError(f"no scripted response for {request.method}")
        if not queue:
            raise MCPTransportError(f"response queue exhausted for {request.method}")
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return json.dumps({"jsonrpc": "2.0", "id": request.id, "result": result})

    async def send_notification(self, notification: JSONRPCNotification) -> None:
        self.notifications.append(notification)

    async def close(self) -> None:
        self.closed = True


def _ok_init_result() -> dict[str, Any]:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "serverInfo": {"name": "fake-server", "version": "0.0.1"},
    }


def _tools_list_result() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": "search",
                "description": "Search the index",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "fetch",
                "description": "Fetch a record",
                "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
            },
        ]
    }


# ---------------------------------------------------------------------------
# Handshake and listing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_handshake_completes_and_notifies() -> None:
    transport = FakeTransport({"initialize": [_ok_init_result()]})
    client = MCPClient(transport=transport, name="fake")

    init = await client.initialize()

    assert client.is_initialized is True
    assert init.server_info.name == "fake-server"
    # initialize request then a notifications/initialized notification
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "initialize"
    assert transport.requests[0].params["protocolVersion"] == "2024-11-05"
    assert len(transport.notifications) == 1
    assert transport.notifications[0].method == "notifications/initialized"


@pytest.mark.asyncio
async def test_list_tools_returns_parsed_definitions() -> None:
    transport = FakeTransport(
        {"initialize": [_ok_init_result()], "tools/list": [_tools_list_result()]}
    )
    client = MCPClient(transport=transport, name="fake")
    await client.initialize()

    tools = await client.list_tools()

    assert [t.name for t in tools] == ["search", "fetch"]
    assert tools[0].input_schema["type"] == "object"
    assert "query" in tools[0].input_schema["properties"]
    # tools/list request was sent with empty params
    assert transport.requests[-1].method == "tools/list"


# ---------------------------------------------------------------------------
# Tool invocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_returns_text_content() -> None:
    transport = FakeTransport(
        {
            "initialize": [_ok_init_result()],
            "tools/list": [_tools_list_result()],
            "tools/call": [{"content": [{"type": "text", "text": "hello world"}]}],
        }
    )
    client = MCPClient(transport=transport, name="fake")
    await client.initialize()

    out = await client.call_tool("search", {"query": "x"})
    assert out == "hello world"
    last = transport.requests[-1]
    assert last.method == "tools/call"
    assert last.params == {"name": "search", "arguments": {"query": "x"}}


@pytest.mark.asyncio
async def test_call_tool_error_result_is_prefixed_as_failure() -> None:
    transport = FakeTransport(
        {
            "initialize": [_ok_init_result()],
            "tools/call": [{"content": [{"type": "text", "text": "not found"}], "isError": True}],
        }
    )
    client = MCPClient(transport=transport, name="fake")
    await client.initialize()

    out = await client.call_tool("search", {"query": "missing"})
    # Prefixed with "Error:" so AgentLoop's error detection treats it as failed.
    assert out.startswith("Error:")
    assert "not found" in out


@pytest.mark.asyncio
async def test_call_tool_non_text_content_rendered_as_placeholder() -> None:
    transport = FakeTransport(
        {
            "initialize": [_ok_init_result()],
            "tools/call": [
                {
                    "content": [
                        {"type": "text", "text": "caption"},
                        {"type": "image", "data": "base64..."},
                    ]
                }
            ],
        }
    )
    client = MCPClient(transport=transport, name="fake")
    await client.initialize()

    out = await client.call_tool("fetch", {"id": "1"})
    assert "caption" in out
    assert "unsupported MCP content type: image" in out


# ---------------------------------------------------------------------------
# Protocol error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jsonrpc_error_response_raises_mcp_protocol_error() -> None:
    transport = FakeTransport({"initialize": [_ok_init_result()]})
    client = MCPClient(transport=transport, name="fake")
    await client.initialize()

    # Override the transport so the next request (tools/call) returns a
    # JSON-RPC error envelope instead of a result.
    async def _err_response(request: JSONRPCRequest) -> str:
        transport.requests.append(request)
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request.id,
                "error": {"code": -32602, "message": "invalid params"},
            }
        )

    transport.send_request = _err_response  # type: ignore[method-assign]

    with pytest.raises(MCPProtocolError) as exc:
        await client.call_tool("search", {"query": "x"})
    assert "invalid params" in str(exc.value)


@pytest.mark.asyncio
async def test_non_json_response_raises_protocol_error() -> None:
    transport = FakeTransport({"initialize": [_ok_init_result()]})
    client = MCPClient(transport=transport, name="fake")
    await client.initialize()

    async def _bad_response(request: JSONRPCRequest) -> str:
        transport.requests.append(request)
        return "this is not json"

    transport.send_request = _bad_response  # type: ignore[method-assign]

    with pytest.raises(MCPProtocolError):
        await client.call_tool("search", {"query": "x"})


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_mcp_tools_creates_namespaced_wrappers() -> None:
    transport = FakeTransport(
        {"initialize": [_ok_init_result()], "tools/list": [_tools_list_result()]}
    )
    client = MCPClient(transport=transport, name="fake")
    registry = ToolRegistry()

    names = await register_mcp_tools(registry, client, auto_approve=False)

    assert names == ["mcp__fake__search", "mcp__fake__fetch"]
    spec_names = {t.function.name for t in registry.list_specs()}
    assert {"mcp__fake__search", "mcp__fake__fetch"} <= spec_names
    # Side-effecting by default -> not read-only -> sequential dispatch.
    assert not registry.is_read_only("mcp__fake__search")
    # auto_approve=False -> ASK permission level on the wrapper.
    search_tool = registry.get_tool("mcp__fake__search")
    assert search_tool.permission_level == PermissionLevel.ASK


@pytest.mark.asyncio
async def test_register_mcp_tools_auto_approve_sets_allow() -> None:
    transport = FakeTransport(
        {"initialize": [_ok_init_result()], "tools/list": [_tools_list_result()]}
    )
    client = MCPClient(transport=transport, name="fake")
    registry = ToolRegistry()

    await register_mcp_tools(registry, client, auto_approve=True)

    fetch_tool = registry.get_tool("mcp__fake__fetch")
    assert fetch_tool.permission_level == PermissionLevel.ALLOW


@pytest.mark.asyncio
async def test_registered_mcp_wrapper_proxies_call_to_client() -> None:
    transport = FakeTransport(
        {
            "initialize": [_ok_init_result()],
            "tools/list": [_tools_list_result()],
            "tools/call": [{"content": [{"type": "text", "text": "result-42"}]}],
        }
    )
    client = MCPClient(transport=transport, name="fake")
    registry = ToolRegistry()
    await register_mcp_tools(registry, client, auto_approve=True)

    # auto_approve=True so the registry executes without a permission callback.
    out = await registry.execute("mcp__fake__search", {"query": "x"})
    assert out == "result-42"
    # The tools/call request carried the ORIGINAL (non-namespaced) tool name.
    call_req = next(r for r in transport.requests if r.method == "tools/call")
    assert call_req.params["name"] == "search"
    assert call_req.params["arguments"] == {"query": "x"}


@pytest.mark.asyncio
async def test_mcp_tool_ask_without_callback_is_fail_closed() -> None:
    transport = FakeTransport(
        {"initialize": [_ok_init_result()], "tools/list": [_tools_list_result()]}
    )
    client = MCPClient(transport=transport, name="fake")
    registry = ToolRegistry()  # no permission_callback
    await register_mcp_tools(registry, client, auto_approve=False)

    with pytest.raises(ToolPermissionError):
        await registry.execute("mcp__fake__search", {"query": "x"})


# ---------------------------------------------------------------------------
# StdioTransport: argv construction (no shell) + real subprocess E2E
# ---------------------------------------------------------------------------


def test_stdio_transport_builds_argv_without_shell() -> None:
    transport = StdioTransport(command="npx", args=["-y", "@mcp/server"], env={"FOO": "bar"})
    argv = transport._argv_for_exec()  # type: ignore[reportPrivateUsage]
    assert argv == ["npx", "-y", "@mcp/server"]
    assert "FOO" in transport._env  # type: ignore[reportPrivateUsage]
    assert transport._env["PYTHONUNBUFFERED"] == "1"  # type: ignore[reportPrivateUsage]


def test_stdio_transport_rejects_empty_command() -> None:
    with pytest.raises(MCPTransportError):
        StdioTransport(command="")


@pytest.mark.asyncio
async def test_stdio_transport_real_subprocess_handshake(tmp_path: Path) -> None:
    """Spawn a real Python subprocess acting as a minimal MCP server.

    Verifies the actual StdioTransport I/O (create_subprocess_exec with an
    argv list, newline-delimited JSON exchange) works end-to-end. Fully
    offline — no network, only a local subprocess.
    """
    server_script = tmp_path / "fake_mcp_server.py"
    server_script.write_text(
        textwrap.dedent(
            """
            import json
            import sys

            def main() -> None:
                while True:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "id" not in msg:
                        continue  # notification, no response
                    method = msg.get("method")
                    rid = msg["id"]
                    if method == "initialize":
                        result = {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "serverInfo": {"name": "py-fake", "version": "0.1"},
                        }
                    elif method == "tools/list":
                        result = {"tools": [
                            {"name": "echo", "description": "echo back",
                             "inputSchema": {"type": "object",
                                             "properties": {"text": {"type": "string"}}}}
                        ]}
                    elif method == "tools/call":
                        args = msg.get("params", {}).get("arguments", {})
                        result = {"content": [{"type": "text",
                                               "text": args.get("text", "")}]}
                    else:
                        sys.stdout.write(json.dumps({
                            "jsonrpc": "2.0", "id": rid,
                            "error": {"code": -32601, "message": "unknown method"},
                        }) + "\\n")
                        sys.stdout.flush()
                        continue
                    sys.stdout.write(json.dumps({
                        "jsonrpc": "2.0", "id": rid, "result": result,
                    }) + "\\n")
                    sys.stdout.flush()

            if __name__ == "__main__":
                main()
            """
        )
    )

    transport = StdioTransport(command=sys.executable, args=[str(server_script)], timeout=15.0)
    client = MCPClient(transport=transport, name="py-fake")
    try:
        init = await client.initialize()
        assert init.server_info.name == "py-fake"
        tools = await client.list_tools()
        assert tools[0].name == "echo"
        out = await client.call_tool("echo", {"text": "roundtrip"})
        assert out == "roundtrip"
    finally:
        await client.close()
        assert transport._proc is None  # type: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# AgentLoop integration: an MCP tool is callable through the act loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_invocable_via_agent_loop(tmp_path: Path) -> None:
    """An MCP-registered tool participates in the Act loop like a built-in."""
    from collections.abc import AsyncGenerator

    from nullain.agent import AgentLoop
    from nullain.events import EventBus
    from nullain.llm import CompletionChunk, CompletionRequest, LLMProvider, TokenUsage

    transport = FakeTransport(
        {
            "initialize": [_ok_init_result()],
            "tools/list": [
                {
                    "tools": [
                        {
                            "name": "lookup",
                            "description": "look something up",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                            },
                        }
                    ]
                }
            ],
            "tools/call": [{"content": [{"type": "text", "text": "answer=42"}]}],
        }
    )
    client = MCPClient(transport=transport, name="kb")
    registry = ToolRegistry()
    await register_mcp_tools(registry, client, auto_approve=True)

    class _Provider(LLMProvider):
        def __init__(self) -> None:
            self._calls = 0

        async def generate(self, request: CompletionRequest) -> CompletionChunk:
            self._calls += 1
            if self._calls == 1:
                # Plan-phase spec for a MEDIUM task -> JSON spec.
                return CompletionChunk(
                    delta_text='{"objective": "lookup", "steps": ["lookup"], '
                    '"target_files": [], "acceptance_criteria": []}'
                )
            if self._calls == 2:
                return CompletionChunk(
                    tool_calls=[
                        ToolCall(id="c1", name="mcp__kb__lookup", arguments={"q": "meaning"})
                    ],
                    usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )
            return CompletionChunk(delta_text="The answer is 42.")

        async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
            yield await self.generate(request)

        async def health_check(self) -> bool:
            return True

    agent = AgentLoop(
        provider=_Provider(),
        tools=registry,
        event_bus=EventBus(),
        max_steps=5,
        workspace_root=tmp_path,
    )
    result = await agent.run_result("lookup the meaning")
    assert result.status == "success"
    assert "42" in (result.final_text or "")
    # Confirm the MCP tool was actually invoked via tools/call.
    assert any(r.method == "tools/call" for r in transport.requests)
