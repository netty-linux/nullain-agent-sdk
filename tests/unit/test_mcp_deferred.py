"""Unit tests for P4.26 — deferred MCP tool schemas + tool search.

100% offline: the MCP client is driven by an in-memory :class:`FakeTransport`
(no subprocess), and the deferred-schema / search / hydration behavior is
exercised directly against the registry and the ``search_tools`` tool.

The scaling invariant under test: a deferred tool's full schema is NOT loaded
eagerly and is hidden from ``list_specs()`` until the agent searches for it, at
which point it is hydrated on demand — so the LLM only ever sees schemas for
tools it has chosen to use.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from nullain.errors import MCPProtocolError
from nullain.mcp import MCPClient
from nullain.mcp.protocol import JSONRPCNotification, JSONRPCRequest
from nullain.tools import ToolRegistry, ToolSearchResult
from nullain_tools.search import create_search_tools_tool


def _list_calls(client: MCPClient) -> list[JSONRPCRequest]:
    """Return the ``tools/list`` requests the client sent (for cache assertions)."""
    transport = client._transport  # type: ignore[reportPrivateUsage]
    assert isinstance(transport, FakeTransport)
    return [r for r in transport.requests if r.method == "tools/list"]


class FakeTransport:
    """In-memory MCP transport with scripted responses (mirrors test_plugins)."""

    def __init__(self, responses: dict[str, list[Any]]) -> None:
        self._responses: dict[str, list[Any]] = {k: list(v) for k, v in responses.items()}
        self.requests: list[JSONRPCRequest] = []
        self.notifications: list[JSONRPCNotification] = []
        self.closed = False

    async def start(self) -> None:
        """No-op."""

    async def send_request(self, request: JSONRPCRequest) -> str:
        self.requests.append(request)
        queue = self._responses.get(request.method)
        if not queue:
            raise AssertionError(f"no scripted response for {request.method}")
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
                "name": "get_commit",
                "description": "Get a commit by SHA",
                "inputSchema": {
                    "type": "object",
                    "properties": {"sha": {"type": "string"}},
                    "required": ["sha"],
                },
            },
            {
                "name": "list_repos",
                "description": "List repositories",
                "inputSchema": {
                    "type": "object",
                    "properties": {"org": {"type": "string"}},
                },
            },
        ]
    }


def _client() -> MCPClient:
    transport = FakeTransport(
        {
            "initialize": [_ok_init_result()],
            "tools/list": [_tools_list_result()],
        }
    )
    return MCPClient(transport=transport, name="github")


# ---------------------------------------------------------------------------
# MCPClient.get_tool_schema — lazy single-tool fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_tool_schema_returns_matching_definition() -> None:
    client = _client()
    await client.initialize()
    definition = await client.get_tool_schema("get_commit")
    assert definition.name == "get_commit"
    assert definition.input_schema["required"] == ["sha"]
    await client.close()


@pytest.mark.asyncio
async def test_get_tool_schema_raises_when_absent() -> None:
    client = _client()
    await client.initialize()
    with pytest.raises(MCPProtocolError, match="no tool named"):
        await client.get_tool_schema("nope")
    await client.close()


@pytest.mark.asyncio
async def test_get_tool_schema_reuses_cached_list() -> None:
    client = _client()
    await client.initialize()
    await client.get_tool_schema("get_commit")
    await client.get_tool_schema("list_repos")
    # tools/list is fetched once; both lookups reuse the cache.
    assert len(_list_calls(client)) == 1
    await client.close()


# ---------------------------------------------------------------------------
# register_mcp_tools — deferred vs eager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_deferred_hides_schema_until_hydrated() -> None:
    client = _client()
    registry = ToolRegistry()
    names = await register_deferred(registry, client)
    assert names == ["mcp__github__get_commit", "mcp__github__list_repos"]

    tool = registry.get_tool("mcp__github__get_commit")
    assert tool.is_deferred is True
    assert tool.is_hydrated is False
    # Schema is empty until hydrated; the tool is hidden from list_specs().
    assert tool.spec.function.parameters == {}
    assert registry.list_specs() == []

    await registry.hydrate_tool("mcp__github__get_commit")
    assert tool.is_hydrated is True
    assert tool.spec.function.parameters["required"] == ["sha"]
    # Only the hydrated tool appears in list_specs().
    specs = registry.list_specs()
    assert [s.function.name for s in specs] == ["mcp__github__get_commit"]
    await client.close()


@pytest.mark.asyncio
async def test_register_eager_keeps_full_schema() -> None:
    client = _client()
    registry = ToolRegistry()
    names = await register_eager(registry, client)
    assert names == ["mcp__github__get_commit", "mcp__github__list_repos"]

    tool = registry.get_tool("mcp__github__get_commit")
    assert tool.is_deferred is False
    assert tool.is_hydrated is True
    assert tool.spec.function.parameters["required"] == ["sha"]
    # Eager tools are always present in list_specs().
    assert len(registry.list_specs()) == 2
    await client.close()


# ---------------------------------------------------------------------------
# ToolRegistry.search_tools — discovery without loading schemas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_tools_matches_name_and_description() -> None:
    client = _client()
    registry = ToolRegistry()
    await register_deferred(registry, client)
    # Matches description ("commit").
    by_desc = registry.search_tools("commit")
    assert [r.name for r in by_desc] == ["mcp__github__get_commit"]
    # Matches name ("repos").
    by_name = registry.search_tools("repos")
    assert [r.name for r in by_name] == ["mcp__github__list_repos"]
    # Case-insensitive.
    assert registry.search_tools("COMMIT") == by_desc
    # No match.
    assert registry.search_tools("zzz") == []
    await client.close()


@pytest.mark.asyncio
async def test_search_tools_returns_hydration_state() -> None:
    client = _client()
    registry = ToolRegistry()
    await register_deferred(registry, client)
    results = registry.search_tools("commit")
    assert isinstance(results[0], ToolSearchResult)
    assert results[0].is_deferred is True
    assert results[0].is_hydrated is False
    await client.close()


@pytest.mark.asyncio
async def test_hydrate_tool_is_idempotent() -> None:
    client = _client()
    registry = ToolRegistry()
    await register_deferred(registry, client)
    tool = registry.get_tool("mcp__github__get_commit")
    await registry.hydrate_tool("mcp__github__get_commit")
    await registry.hydrate_tool("mcp__github__get_commit")
    assert tool.is_hydrated is True
    # tools/list fetched once (registration) — hydration reuses the cache.
    assert len(_list_calls(client)) == 1
    await client.close()


# ---------------------------------------------------------------------------
# search_tools agent tool — search + hydrate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_tools_tool_hydrates_matches() -> None:
    client = _client()
    registry = ToolRegistry()
    await register_deferred(registry, client)
    search = create_search_tools_tool(registry)
    out = await search.execute({"query": "commit"})
    assert "mcp__github__get_commit" in out
    assert "schema loaded" in out
    # The matched tool is now hydrated and visible to list_specs().
    assert registry.get_tool("mcp__github__get_commit").is_hydrated is True
    assert [s.function.name for s in registry.list_specs()] == ["mcp__github__get_commit"]
    await client.close()


@pytest.mark.asyncio
async def test_search_tools_tool_no_match() -> None:
    client = _client()
    registry = ToolRegistry()
    await register_deferred(registry, client)
    search = create_search_tools_tool(registry)
    out = await search.execute({"query": "zzz"})
    assert "No tools match" in out
    await client.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def register_deferred(registry: ToolRegistry, client: MCPClient) -> list[str]:
    from nullain.mcp import register_mcp_tools

    return await register_mcp_tools(registry, client, defer_schemas=True)


async def register_eager(registry: ToolRegistry, client: MCPClient) -> list[str]:
    from nullain.mcp import register_mcp_tools

    return await register_mcp_tools(registry, client)
