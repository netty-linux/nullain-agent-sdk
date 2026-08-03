"""Nullain Agent SDK — MCP Client and ToolRegistry Integration.

:class:`MCPClient` speaks the MCP JSON-RPC protocol over an injectable
:class:`~nullain.mcp.transport.MCPTransport`: it performs the initialize
handshake, lists remote tools, and forwards ``tools/call`` invocations. The
client never imports the concrete transport — that is injected, mirroring the
:class:`~nullain.llm.provider.LLMProvider` port pattern.

:func:`register_mcp_tools` bridges MCP tools into the SDK's
:class:`~nullain.tools.registry.ToolRegistry`: each remote tool becomes a
:class:`~nullain.tools.decorator.RegisteredTool` whose execution function is an
async closure proxying to the client. Because an MCP server's side-effects
cannot be inferred from its arguments, each wrapper carries a fixed
``permission_level`` — either ``ALLOW`` (trusted server, ``auto_approve``) or
``ASK`` (gated through the human approval loop). Names follow the
``mcp__<server>__<tool>`` convention to avoid collisions with built-in tools.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from nullain.authority import Capability
from nullain.errors import MCPProtocolError
from nullain.llm.types import FunctionSpec, ToolSpec
from nullain.mcp.protocol import (
    MCP_PROTOCOL_VERSION,
    METHOD_INITIALIZE,
    METHOD_NOTIFICATIONS_INITIALIZED,
    METHOD_TOOLS_CALL,
    METHOD_TOOLS_LIST,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPToolDefinition,
)
from nullain.mcp.transport import MCPTransport
from nullain.tools.decorator import RegisteredTool
from nullain.tools.permissions import PermissionLevel
from nullain.tools.registry import ToolRegistry


class MCPClient:
    """JSON-RPC 2.0 client for a single MCP server.

    The client is single-flight (one outstanding request at a time), which
    matches the synchronous stdio request/response model and the agent's
    sequential tool dispatch for side-effecting tools.
    """

    def __init__(
        self,
        transport: MCPTransport,
        name: str = "mcp",
        client_name: str = "nullain",
        client_version: str = "0.1.0",
    ) -> None:
        self._transport = transport
        self.name = name
        self._client_name = client_name
        self._client_version = client_version
        self._next_id = 1
        self._initialized = False

    def _next_request_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    @property
    def is_initialized(self) -> bool:
        """Whether the initialize handshake has completed successfully."""
        return self._initialized

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        """Send a JSON-RPC request and return its validated ``result`` field.

        Raises:
            MCPProtocolError: server returned an error or malformed response.
            MCPTransportError: transport-level failure (EOF, timeout, spawn).
        """
        request = JSONRPCRequest(id=self._next_request_id(), method=method, params=params)
        raw = await self._transport.send_request(request)
        return _parse_jsonrpc_result(raw, request.id)

    async def initialize(self) -> Any:
        """Perform the MCP initialize handshake.

        Sends ``initialize``, validates the result, then emits the
        ``notifications/initialized`` notification. Subsequent ``list_tools``
        / ``call_tool`` calls require this to have succeeded.
        """
        result = await self._request(
            METHOD_INITIALIZE,
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self._client_name, "version": self._client_version},
            },
        )
        # Validate the server's initialize result shape (untrusted output).
        from nullain.mcp.protocol import InitializeResult

        init = InitializeResult.model_validate(result)
        self._initialized = True
        await self._transport.send_notification(
            JSONRPCNotification(method=METHOD_NOTIFICATIONS_INITIALIZED, params={})
        )
        return init

    async def list_tools(self) -> list[MCPToolDefinition]:
        """Return the tool definitions advertised by the server."""
        result = await self._request(METHOD_TOOLS_LIST, {})
        from nullain.mcp.protocol import MCPListToolsResult

        parsed = MCPListToolsResult.model_validate(result or {})
        return list(parsed.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke a remote tool and return its text content.

        Non-text content items are rendered as placeholders. If the server
        signals ``isError``, the concatenated content is returned prefixed so
        the Act loop's error-detection treats it as a failed tool call.
        """
        result = await self._request(METHOD_TOOLS_CALL, {"name": name, "arguments": arguments})
        from nullain.mcp.protocol import MCPToolResult

        parsed = MCPToolResult.model_validate(result or {})
        rendered = "\n".join(item.render() for item in parsed.content).strip()
        if parsed.is_error:
            # Mirror the bundled tools' failure convention so self-correction
            # and loop-detection in AgentLoop recognize the failure.
            return f"Error: MCP tool '{name}' returned an error: {rendered}"
        return rendered

    async def close(self) -> None:
        """Tear down the underlying transport."""
        self._initialized = False
        await self._transport.close()


def _parse_jsonrpc_result(raw: str, request_id: int | str) -> Any:
    """Parse a JSON-RPC response line and return its ``result`` field.

    The envelope is validated through :class:`JSONRPCResponse` — MCP server
    output is untrusted, so it goes through Pydantic at the boundary exactly
    like LLM output (AGENTS.md rule 3). Raises :class:`MCPProtocolError` for
    error responses, id mismatches, or malformed JSON.

    Args:
        raw: The raw response line from the transport.
        request_id: The id of the request this response corresponds to.

    Returns:
        The validated ``result`` payload (may be ``None`` for an explicit
        ``"result": null``).
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as err:
        raise MCPProtocolError(
            f"MCP server returned non-JSON response: {raw[:200]}",
            details={"raw": raw[:500]},
        ) from err
    try:
        response = JSONRPCResponse.model_validate(obj)
    except ValidationError as err:
        raise MCPProtocolError(
            "MCP response is not a valid JSON-RPC response object",
            details={"raw": raw[:500]},
        ) from err
    if response.id != request_id:
        raise MCPProtocolError(
            "MCP response id does not match request id",
            details={"expected": request_id, "got": response.id},
        )
    if response.error is not None:
        raise MCPProtocolError(
            f"MCP server error (code {response.error.code}): {response.error.message}",
            details={"code": response.error.code, "message": response.error.message},
        )
    # ``result: Any | None = None`` cannot distinguish an explicit ``null`` from
    # a missing key, so check the raw object to reject envelopes carrying neither
    # ``result`` nor ``error`` (malformed per JSON-RPC 2.0).
    if "result" not in (obj if isinstance(obj, dict) else {}):
        raise MCPProtocolError(
            "MCP response has neither result nor error", details={"raw": raw[:500]}
        )
    return response.result


def _namespaced_tool_name(server: str, tool: MCPToolDefinition) -> str:
    """Build a collision-safe registered name: ``mcp__<server>__<tool>``."""
    safe_server = server.replace("-", "_")
    return f"mcp__{safe_server}__{tool.name}"


def _make_tool_spec(server: str, tool: MCPToolDefinition) -> ToolSpec:
    """Convert an MCP tool definition into the SDK's ToolSpec."""
    return ToolSpec(
        type="function",
        function=FunctionSpec(
            name=_namespaced_tool_name(server, tool),
            description=tool.description or f"MCP tool {tool.name} from server '{server}'",
            parameters=tool.input_schema or {},
        ),
    )


async def register_mcp_tools(
    registry: ToolRegistry,
    client: MCPClient,
    *,
    auto_approve: bool = False,
) -> list[str]:
    """Register an MCP server's tools into a ToolRegistry.

    Performs the initialize handshake (if not already done) and ``tools/list``,
    then registers one :class:`RegisteredTool` per remote tool. Each wrapper
    proxies execution to ``client.call_tool``. MCP tools are treated as
    side-effecting (``read_only=False``) so the Act loop dispatches them
    sequentially.

    Args:
        registry: Target tool registry.
        client: MCP client (initialized here if it has not been already).
        auto_approve: When True, MCP tool calls resolve to ``ALLOW``. When
            False (the default), they resolve to ``ASK`` and flow through the
            registry's permission callback — fail-closed if none is set.

    Returns:
        The list of registered tool names.
    """
    if not client.is_initialized:
        await client.initialize()
    tools = await client.list_tools()
    level = PermissionLevel.ALLOW if auto_approve else PermissionLevel.ASK
    registered_names: list[str] = []
    for tool in tools:
        namespaced = _namespaced_tool_name(client.name, tool)
        spec = _make_tool_spec(client.name, tool)
        remote_name = tool.name

        async def _proxy(_remote: str = remote_name, **kwargs: Any) -> str:
            return await client.call_tool(_remote, dict(kwargs))

        wrapper = RegisteredTool(
            name=namespaced,
            description=spec.function.description,
            spec=spec,
            func=_proxy,
            read_only=False,
            permission_level=level,
            # An MCP server's side-effects cannot be introspected, so each
            # wrapper is conservatively treated as WRITE-capable: a subagent
            # delegated read-only authority therefore cannot invoke it. The
            # fixed permission_level (ASK/ALLOW) remains the human-approval
            # gate; this capability tag is the authority-intersection gate.
            requires=frozenset({Capability.WRITE}),
        )
        registry.register(wrapper)
        registered_names.append(namespaced)
    return registered_names


__all__ = [
    "MCPClient",
    "register_mcp_tools",
]
