"""Nullain Agent SDK — LSP client (M11.2).

A stdio JSON-RPC client for the Language Server Protocol (LSP 3.17), plus the
four read-only tools that expose it to the agent. Configuration lives under
``[lsp.servers.<lang>]`` in ``nullain.toml`` (see :mod:`nullain.lsp.config`).
"""

from nullain.lsp.client import LSPClient
from nullain.lsp.config import (
    EXTENSION_TO_LANGUAGE,
    LSPConfig,
    LSPServerConfig,
    language_for_path,
)
from nullain.lsp.tools import register_lsp_tools
from nullain.lsp.types import (
    Diagnostic,
    DiagnosticReport,
    Hover,
    Location,
    LocationLink,
    LocationResult,
    MarkupContent,
    Position,
    Range,
)

__all__ = [
    "EXTENSION_TO_LANGUAGE",
    "Diagnostic",
    "DiagnosticReport",
    "Hover",
    "LSPClient",
    "LSPConfig",
    "LSPServerConfig",
    "Location",
    "LocationLink",
    "LocationResult",
    "MarkupContent",
    "Position",
    "Range",
    "language_for_path",
    "register_lsp_tools",
]
