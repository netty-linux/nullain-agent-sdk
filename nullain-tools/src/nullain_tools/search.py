"""Nullain Tools — ``search_tools``: discover and hydrate deferred-schema tools (P4.26).

The agent calls ``search_tools(query)`` to find relevant tools by name or
description. For MCP tools registered with deferred schemas, the act of
searching hydrates the matched tools' schemas, so they become callable on the
next agent step (the loop re-assembles ``list_specs()`` every step). This is the
"tool search (schemas deferidos)" pattern: the LLM only ever sees full schemas
for tools it has chosen to use.
"""

from __future__ import annotations

from nullain.tools import RegisteredTool, ToolRegistry, tool


def create_search_tools_tool(registry: ToolRegistry) -> RegisteredTool:
    """Build the ``search_tools`` discovery tool bound to a registry.

    Args:
        registry: The registry to search. Matched deferred tools are hydrated
            in place, so their schemas appear in the next ``list_specs()``.
    """

    @tool(
        name="search_tools",
        description=(
            "Search available tools by name or description and load their schemas. "
            "Use this to discover MCP tools (prefixed mcp__) before calling them: "
            "searching hydrates the matched tools' schemas so they become callable. "
            "Returns a list of matching tool names, descriptions, and schema state."
        ),
        read_only=True,
    )
    async def search_tools(query: str, limit: int = 20) -> str:
        """Search the registry and hydrate matched deferred tools.

        Args:
            query: Substring to match against tool names and descriptions
                (case-insensitive).
            limit: Maximum number of matches to hydrate and report.
        """
        matches = registry.search_tools(query)
        if not matches:
            return f"No tools match '{query}'."
        # Hydrate the matched deferred tools (bounded) so their schemas load
        # into the next step's tool list. Non-deferred tools are already
        # hydrated; hydrate_tool is a no-op for them.
        for result in matches[:limit]:
            await registry.hydrate_tool(result.name)
        lines = [f"Found {len(matches)} tool(s) matching '{query}':"]
        for result in matches[:limit]:
            # Re-read the live hydration state (search_tools captured it before
            # hydration above).
            hydrated = registry.get_tool(result.name).is_hydrated
            state = "schema loaded" if hydrated else "schema deferred"
            lines.append(f"- {result.name} — {result.description} ({state})")
        if len(matches) > limit:
            lines.append(f"... and {len(matches) - limit} more (raise limit to see them)")
        return "\n".join(lines)

    return search_tools


__all__ = ["create_search_tools_tool"]
