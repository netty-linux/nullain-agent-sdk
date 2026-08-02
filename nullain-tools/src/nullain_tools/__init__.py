"""Nullain Tools — Package Initialization and Helper functions."""

from pathlib import Path

from nullain.tools import RegisteredTool, ToolRegistry

from nullain_tools.bash import create_bash_tool
from nullain_tools.filesystem import create_filesystem_tools
from nullain_tools.git import create_git_tools


def register_default_tools(registry: ToolRegistry, workspace_root: str | Path) -> None:
    """Register all built-in filesystem, bash, and git tools into a ToolRegistry."""
    tools: list[RegisteredTool] = []
    tools.extend(create_filesystem_tools(workspace_root))
    tools.append(create_bash_tool(workspace_root))
    tools.extend(create_git_tools(workspace_root))

    for t in tools:
        registry.register(t)


__version__ = "0.1.0"

__all__ = [
    "create_bash_tool",
    "create_filesystem_tools",
    "create_git_tools",
    "register_default_tools",
]
