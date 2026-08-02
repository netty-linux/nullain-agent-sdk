"""Nullain Tools — Package Initialization and Helper functions."""

from pathlib import Path

from nullain.tools import RegisteredTool, ToolRegistry

from nullain_tools.ask_user import AskUserCallback, create_ask_user_tool
from nullain_tools.bash import create_bash_tool
from nullain_tools.filesystem import create_filesystem_tools
from nullain_tools.git import create_git_tools
from nullain_tools.web import create_web_fetch_tool


def register_default_tools(
    registry: ToolRegistry,
    workspace_root: str | Path,
    ask_user_callback: AskUserCallback | None = None,
) -> None:
    """Register all built-in tools into a ToolRegistry.

    Args:
        registry: Target tool registry.
        workspace_root: Workspace root for filesystem/bash/git tools.
        ask_user_callback: Optional async callback backing the ``ask_user``
            tool. When None, ``ask_user`` is still registered but returns an
            error when invoked (graceful degradation for unattended runs).
    """
    tools: list[RegisteredTool] = []
    tools.extend(create_filesystem_tools(workspace_root))
    tools.append(create_bash_tool(workspace_root))
    tools.extend(create_git_tools(workspace_root))
    tools.append(create_web_fetch_tool())
    tools.append(create_ask_user_tool(ask_user_callback))

    for t in tools:
        registry.register(t)


__version__ = "0.1.0"

__all__ = [
    "AskUserCallback",
    "create_ask_user_tool",
    "create_bash_tool",
    "create_filesystem_tools",
    "create_git_tools",
    "create_web_fetch_tool",
    "register_default_tools",
]
