"""Nullain Tools — Package Initialization and Helper functions."""

from pathlib import Path

from nullain.memory import PersistentMemory
from nullain.tools import RegisteredTool, ToolRegistry
from nullain.tools.sandbox import Sandbox, SandboxOptions

from nullain_tools.ask_user import AskUserCallback, create_ask_user_tool
from nullain_tools.bash import create_bash_tool
from nullain_tools.filesystem import create_filesystem_tools
from nullain_tools.git import create_git_tools
from nullain_tools.memory import create_memory_tools
from nullain_tools.web import create_web_fetch_tool


def register_default_tools(
    registry: ToolRegistry,
    workspace_root: str | Path,
    ask_user_callback: AskUserCallback | None = None,
    persistent_memory: PersistentMemory | None = None,
    sandbox: Sandbox | None = None,
    sandbox_opts: SandboxOptions | None = None,
) -> None:
    """Register all built-in tools into a ToolRegistry.

    Args:
        registry: Target tool registry.
        workspace_root: Workspace root for filesystem/bash/git tools.
        ask_user_callback: Optional async callback backing the ``ask_user``
            tool. When None, ``ask_user`` is still registered but returns an
            error when invoked (graceful degradation for unattended runs).
        persistent_memory: Optional PersistentMemory; when provided, the
            ``save_memory`` / ``read_memory`` tools are registered so the agent
            can record durable facts.
        sandbox: Optional OS-level isolation adapter applied to the bash and
            git subprocess tools. When None (the default), those tools run
            unsandboxed — backward compatible with existing callers. The
            daemon passes ``select_sandbox(settings.sandbox)`` so the safe
            default (fail-closed) is honored in production.
        sandbox_opts: Options handed to the adapter (workspace root, allow
            paths, deny_network). Required when ``sandbox`` is provided.
    """
    tools: list[RegisteredTool] = []
    tools.extend(create_filesystem_tools(workspace_root))
    tools.append(create_bash_tool(workspace_root, sandbox=sandbox, sandbox_opts=sandbox_opts))
    tools.extend(create_git_tools(workspace_root, sandbox=sandbox, sandbox_opts=sandbox_opts))
    tools.append(create_web_fetch_tool())
    tools.append(create_ask_user_tool(ask_user_callback))
    if persistent_memory is not None:
        tools.extend(create_memory_tools(persistent_memory))

    for t in tools:
        registry.register(t)


__version__ = "0.1.0"

__all__ = [
    "AskUserCallback",
    "create_ask_user_tool",
    "create_bash_tool",
    "create_filesystem_tools",
    "create_git_tools",
    "create_memory_tools",
    "create_web_fetch_tool",
    "register_default_tools",
]
