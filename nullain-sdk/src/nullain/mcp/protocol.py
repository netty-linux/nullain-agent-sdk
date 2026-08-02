"""Nullain Agent SDK — MCP (Model Context Protocol) JSON-RPC Data Models.

These are the wire-level types for speaking JSON-RPC 2.0 with an MCP server
(initialize handshake, ``tools/list``, ``tools/call``). All server responses are
validated through Pydantic — MCP server output is untrusted, exactly like LLM
output and workspace files (AGENTS.md rule 3).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: JSON-RPC protocol version string used by MCP.
JSONRPC_VERSION = "2.0"

#: MCP protocol version negotiated during the initialize handshake.
MCP_PROTOCOL_VERSION = "2024-11-05"


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response envelope.

    Exactly one of ``result`` / ``error`` is present on a real response. Both
    are optional on the model so a single type can carry either shape; the
    client raises :class:`~nullain.errors.MCPProtocolError` when neither (or
    both) is present, or when ``error`` is present.
    """

    model_config = ConfigDict(extra="ignore")

    jsonrpc: str = JSONRPC_VERSION
    id: int | str | None = None
    result: Any | None = None
    error: JSONRPCError | None = None


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request envelope."""

    jsonrpc: str = JSONRPC_VERSION
    id: int
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JSONRPCNotification(BaseModel):
    """JSON-RPC 2.0 notification (no id, no response expected)."""

    jsonrpc: str = JSONRPC_VERSION
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class MCPCapabilities(BaseModel):
    """Server capabilities advertised in the initialize result."""

    model_config = ConfigDict(extra="allow")


class MCPServerInfo(BaseModel):
    """Server name/version advertised in the initialize result."""

    name: str = ""
    version: str = ""


class InitializeResult(BaseModel):
    """Result payload of the MCP ``initialize`` request."""

    model_config = ConfigDict(populate_by_name=True)

    protocol_version: str = Field(default="", alias="protocolVersion")
    capabilities: MCPCapabilities = Field(default_factory=MCPCapabilities)
    server_info: MCPServerInfo = Field(default_factory=MCPServerInfo, alias="serverInfo")


class MCPToolDefinition(BaseModel):
    """A single tool advertised by an MCP server (``tools/list``)."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict, alias="inputSchema")


class MCPListToolsResult(BaseModel):
    """Result payload of ``tools/list``."""

    tools: list[MCPToolDefinition] = Field(default_factory=list[MCPToolDefinition])


class MCPContentItem(BaseModel):
    """One content item inside a ``tools/call`` result.

    MCP tools may return multiple typed content items (text, image, resource).
    Only ``text`` content is surfaced to the agent; non-text items are rendered
    as a placeholder so the model knows something was returned it cannot see.
    """

    type: str = "text"
    text: str | None = None

    def render(self) -> str:
        """Render this item as a string for inclusion in a tool result."""
        if self.type == "text":
            return self.text or ""
        return f"[unsupported MCP content type: {self.type}]"


class MCPToolResult(BaseModel):
    """Result payload of ``tools/call``."""

    model_config = ConfigDict(populate_by_name=True)

    content: list[MCPContentItem] = Field(default_factory=list[MCPContentItem])
    is_error: bool = Field(default=False, alias="isError")


#: Method names used by the client.
METHOD_INITIALIZE = "initialize"
METHOD_NOTIFICATIONS_INITIALIZED = "notifications/initialized"
METHOD_TOOLS_LIST = "tools/list"
METHOD_TOOLS_CALL = "tools/call"


__all__ = [
    "JSONRPC_VERSION",
    "MCP_PROTOCOL_VERSION",
    "METHOD_INITIALIZE",
    "METHOD_NOTIFICATIONS_INITIALIZED",
    "METHOD_TOOLS_CALL",
    "METHOD_TOOLS_LIST",
    "InitializeResult",
    "JSONRPCError",
    "JSONRPCNotification",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "MCPCapabilities",
    "MCPContentItem",
    "MCPListToolsResult",
    "MCPServerInfo",
    "MCPToolDefinition",
    "MCPToolResult",
]
