"""Example 05 — Connect an MCP server.

Shows how to spawn an MCP server over stdio and register its tools into a
registry that the Agent executes against. The server is declared in
``nullain.toml`` under ``[mcp.servers.<name>]``; this example wires one
programmatically.

Run:  uv run python examples/05_mcp_server.py
"""

import asyncio

from nullain import Agent
from nullain.mcp import MCPClient, StdioTransport, register_mcp_tools
from nullain.tools import ToolRegistry


async def main() -> None:
    """Spawn an MCP server, register its tools, and run a prompt."""
    transport = StdioTransport(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
    )
    client = MCPClient(transport=transport, name="filesystem")
    try:
        await client.initialize()
    except Exception as err:
        print(f"MCP server unavailable: {err}")
        return

    registry = ToolRegistry()
    await register_mcp_tools(registry, client, auto_approve=False)
    agent = Agent(workspace_root=".", tools=registry)
    result = await agent.run("list the files in the workspace")
    print(f"status: {result.status}")
    print(f"output: {result.final_text}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
