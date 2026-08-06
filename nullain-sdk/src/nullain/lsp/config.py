"""Nullain Agent SDK — LSP Client Configuration (M11.2).

The LSP client is configured declaratively in ``nullain.toml`` under the
``[lsp.servers.<lang>]`` section, mirroring the ``[mcp.servers.*]`` format: each
entry names a language (the section key) and the stdio command that serves it.
Tools route a file to its server by mapping the file extension to a language.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LSPServerConfig(BaseModel):
    """Configuration for a single LSP server launched via stdio.

    The server is spawned with ``[command, *args]`` as an explicit argv list
    (never a shell). The section key in ``[lsp.servers.<lang>]`` is the language
    this server serves (e.g. ``python``, ``typescript``); file extensions map to
    that language via :data:`EXTENSION_TO_LANGUAGE`.
    """

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class LSPConfig(BaseModel):
    """LSP client configuration: a named map of stdio LSP servers by language."""

    servers: dict[str, LSPServerConfig] = Field(default_factory=dict)


#: Default file-extension → language mapping used to route a file to its LSP
#: server. Operators can extend this by naming a server for a language that is
#: not listed here; the map is only consulted to derive the language from a
#: file's extension.
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".lua": "lua",
    ".sh": "bash",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
    ".xml": "xml",
}


def language_for_path(path: str) -> str | None:
    """Return the language for a file path, or ``None`` if unknown.

    The language is derived from the file's extension via
    :data:`EXTENSION_TO_LANGUAGE`. A path with no recognized extension yields
    ``None``, which the tools report as an unsupported file type.
    """
    from pathlib import Path

    return EXTENSION_TO_LANGUAGE.get(Path(path).suffix.lower())


__all__ = [
    "EXTENSION_TO_LANGUAGE",
    "LSPConfig",
    "LSPServerConfig",
    "language_for_path",
]
