"""Nullain Agent SDK — Structured Tool Result.

A tool's outcome is carried as a typed :class:`ToolResult` rather than a bare
string so the agent loop can detect failure structurally (``is_error``) instead
of by fragile string-prefix matching. Legacy tools that return a plain ``str``
are bridged into a ``ToolResult`` by :meth:`RegisteredTool.execute`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolResult(BaseModel):
    """Structured outcome of a single tool execution.

    Attributes:
        output: The tool's textual output (may be empty).
        is_error: Whether the tool failed. True for raised ``ToolError`` and
            for tools that return ``ToolResult(is_error=True)``.
        error_type: Machine-readable error kind (e.g. ``"ToolError"``,
            ``"ToolPermissionError"``, ``"HookBlocked"``) when ``is_error``.
        metadata: Optional structured metadata attached by the tool.
    """

    model_config = ConfigDict(frozen=True)

    output: str = ""
    is_error: bool = False
    error_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])


__all__ = ["ToolResult"]
