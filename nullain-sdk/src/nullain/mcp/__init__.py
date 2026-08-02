"""Nullain Agent SDK — Model Context Protocol (MCP) Client.

Connects to external MCP servers (stdio transport) and exposes their tools as
:class:`~nullain.tools.decorator.RegisteredTool` entries in a
:class:`~nullain.tools.registry.ToolRegistry`, so the agent can invoke them
alongside the built-in tools. The client depends on a
:class:`~nullain.mcp.transport.MCPTransport` port, keeping the JSON-RPC logic
decoupled from the subprocess adapter (hexagonal architecture).
"""

from nullain.mcp.client import MCPClient, register_mcp_tools
from nullain.mcp.protocol import MCPToolDefinition
from nullain.mcp.transport import MCPTransport, StdioTransport

__all__ = [
    "MCPClient",
    "MCPToolDefinition",
    "MCPTransport",
    "StdioTransport",
    "register_mcp_tools",
]
