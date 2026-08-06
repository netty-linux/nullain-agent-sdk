"""Nullain Agent SDK — LSP-backed tools (M11.2).

Four read-only tools expose the LSP client to the agent: ``lsp_diagnostics``,
``lsp_goto_definition``, ``lsp_find_references``, and ``lsp_hover``. Each tool
reads the target file, infers its language from the extension, routes to the
configured server for that language, and formats the server's response.

Fail-soft (same pattern as ``_load_mcp_clients``): an unavailable server, an
unsupported file type, or a protocol/transport failure is surfaced as an
informative :class:`ToolResult` error — it never raises out of the tool and
never derails the session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nullain.authority import Capability
from nullain.errors import LSPError
from nullain.lsp.client import LSPClient
from nullain.lsp.config import language_for_path
from nullain.lsp.types import (
    DiagnosticReport,
    Hover,
    Location,
    LocationLink,
    LocationResult,
    MarkupContent,
)
from nullain.tools import tool
from nullain.tools.result import ToolResult

#: Severity labels per the LSP DiagnosticSeverity enum.
_SEVERITY = {1: "error", 2: "warning", 3: "information", 4: "hint"}


def _format_location(loc: Location | LocationLink) -> str:
    """Format a single LSP Location (or LocationLink) as ``path:line:col``."""
    if isinstance(loc, LocationLink):
        start = loc.target_range.start
        return f"{loc.target_uri}:{start.line + 1}:{start.character + 1}"
    start = loc.range.start
    return f"{loc.uri}:{start.line + 1}:{start.character + 1}"


def _format_locations(result: LocationResult) -> str:
    """Format a definition/references result (Location | Location[] | null)."""
    if result is None:
        return "No results."
    if isinstance(result, (Location, LocationLink)):
        return _format_location(result)
    if not result:
        return "No results."
    return "\n".join(_format_location(loc) for loc in result)


def _format_hover(result: Hover | None) -> str:
    """Format a hover result (Hover | null) into readable text."""
    if result is None:
        return "No hover information."
    contents = result.contents
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        parts: list[str] = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(item.value)
        return "\n".join(p for p in parts if p) or "No hover information."
    if isinstance(contents, MarkupContent):
        return contents.value
    return "No hover information."


def _format_diagnostics(result: DiagnosticReport | None) -> str:
    """Format a pull-diagnostics result (``{kind, items}``) into readable text."""
    if result is None or not result.items:
        return "No diagnostics."
    lines: list[str] = []
    for d in result.items:
        start = d.range.start
        sev = _SEVERITY.get(d.severity or 0, "diagnostic")
        lines.append(f"{sev}:{start.line + 1}:{start.character + 1} [{d.source or ''}] {d.message}")
    return "\n".join(lines)


def _read_file(path: str) -> tuple[Path, str] | ToolResult:
    """Resolve and read a file, returning ``(path, text)`` or an error result."""
    p = Path(path).resolve()
    if not p.is_file():
        return ToolResult(
            output=f"Error: file not found: {path}",
            is_error=True,
            error_type="ToolError",
        )
    return p, p.read_text(encoding="utf-8", errors="replace")


def _route_client(clients: dict[str, LSPClient], path: Path) -> tuple[LSPClient | None, str | None]:
    """Return ``(client, language)`` for a path, or ``(None, reason)``."""
    language = language_for_path(str(path))
    if language is None:
        return None, f"unsupported file type: {path.suffix or '(no extension)'}"
    client = clients.get(language)
    if client is None:
        return None, f"no LSP server configured for language '{language}'"
    return client, None


def _fail_soft(err: LSPError, action: str) -> ToolResult:
    """Convert an LSP failure into an informative error result (fail-soft)."""
    return ToolResult(
        output=f"Error: LSP {action} failed: {err}",
        is_error=True,
        error_type="ToolError",
    )


def register_lsp_tools(
    registry: Any,
    clients: dict[str, LSPClient],
) -> list[str]:
    """Register the four LSP tools into a ToolRegistry.

    Args:
        registry: Target tool registry.
        clients: Map of language → initialized :class:`LSPClient`. Tools route
            a file to its server by extension → language → this map.

    Returns:
        The list of registered tool names.
    """
    registered: list[str] = []

    @tool(
        name="lsp_diagnostics",
        description=(
            "Return the LSP diagnostics (errors, warnings, hints) for a file. "
            "The language is inferred from the file extension and routed to the "
            "configured LSP server for that language."
        ),
        read_only=True,
        requires=frozenset({Capability.READ}),
    )
    async def lsp_diagnostics(path: str) -> str | ToolResult:
        read = _read_file(path)
        if isinstance(read, ToolResult):
            return read
        p, text = read
        client, reason = _route_client(clients, p)
        if client is None:
            return ToolResult(
                output=f"Error: {reason}.",
                is_error=True,
                error_type="ToolError",
            )
        try:
            result = await client.diagnostics(str(p), text)
        except LSPError as err:
            return _fail_soft(err, "diagnostics")
        return _format_diagnostics(result)

    @tool(
        name="lsp_goto_definition",
        description=(
            "Resolve the definition of the symbol at a position in a file. "
            "Positions are 0-based line and character offsets. Returns the "
            "location(s) of the definition."
        ),
        read_only=True,
        requires=frozenset({Capability.READ}),
    )
    async def lsp_goto_definition(path: str, line: int, col: int) -> str | ToolResult:
        read = _read_file(path)
        if isinstance(read, ToolResult):
            return read
        p, text = read
        client, reason = _route_client(clients, p)
        if client is None:
            return ToolResult(
                output=f"Error: {reason}.",
                is_error=True,
                error_type="ToolError",
            )
        try:
            result = await client.goto_definition(str(p), text, line, col)
        except LSPError as err:
            return _fail_soft(err, "goto_definition")
        return _format_locations(result)

    @tool(
        name="lsp_find_references",
        description=(
            "Find all references to the symbol at a position in a file. "
            "Positions are 0-based line and character offsets. Returns the "
            "locations of every reference."
        ),
        read_only=True,
        requires=frozenset({Capability.READ}),
    )
    async def lsp_find_references(path: str, line: int, col: int) -> str | ToolResult:
        read = _read_file(path)
        if isinstance(read, ToolResult):
            return read
        p, text = read
        client, reason = _route_client(clients, p)
        if client is None:
            return ToolResult(
                output=f"Error: {reason}.",
                is_error=True,
                error_type="ToolError",
            )
        try:
            result = await client.find_references(str(p), text, line, col)
        except LSPError as err:
            return _fail_soft(err, "find_references")
        return _format_locations(result)

    @tool(
        name="lsp_hover",
        description=(
            "Return hover information (type signature, docs) for the symbol at "
            "a position in a file. Positions are 0-based line and character "
            "offsets."
        ),
        read_only=True,
        requires=frozenset({Capability.READ}),
    )
    async def lsp_hover(path: str, line: int, col: int) -> str | ToolResult:
        read = _read_file(path)
        if isinstance(read, ToolResult):
            return read
        p, text = read
        client, reason = _route_client(clients, p)
        if client is None:
            return ToolResult(
                output=f"Error: {reason}.",
                is_error=True,
                error_type="ToolError",
            )
        try:
            result = await client.hover(str(p), text, line, col)
        except LSPError as err:
            return _fail_soft(err, "hover")
        return _format_hover(result)

    for t in (lsp_diagnostics, lsp_goto_definition, lsp_find_references, lsp_hover):
        registry.register(t)
        registered.append(t.name)
    return registered


__all__ = ["register_lsp_tools"]
