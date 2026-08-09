"""Nullain Tools — Package Initialization and Helper functions."""

from pathlib import Path

from nullain.events import EventBus
from nullain.memory import PersistentMemory
from nullain.tools import RegisteredTool, ToolRegistry
from nullain.tools.sandbox import Sandbox, SandboxOptions

from nullain_tools.ask_user import AskUserCallback, create_ask_user_tool
from nullain_tools.bash import create_bash_tool
from nullain_tools.checkpoints import CheckpointStore
from nullain_tools.filesystem import FileAccessTracker, create_filesystem_tools
from nullain_tools.git import create_git_tools
from nullain_tools.memory import create_memory_tools
from nullain_tools.search import create_search_tools_tool
from nullain_tools.web import create_web_fetch_tool


def register_default_tools(
    registry: ToolRegistry,
    workspace_root: str | Path,
    ask_user_callback: AskUserCallback | None = None,
    persistent_memory: PersistentMemory | None = None,
    sandbox: Sandbox | None = None,
    sandbox_opts: SandboxOptions | None = None,
    file_access_tracker: FileAccessTracker | None = None,
    event_bus: EventBus | None = None,
    session_id: str = "default",
    checkpoint_store: CheckpointStore | None = None,
    bash_timeout: float = 300.0,
    web_fetch_headers: dict[str, str] | None = None,
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
        file_access_tracker: Session-scoped read tracker (M8). When None, a
            fresh instance is created per call (backward compatible).
        event_bus: Optional event bus; when provided, ``todo_write`` emits a
            ``TodoEvent`` on each update.
        session_id: Session id stamped on emitted ``TodoEvent``s.
        checkpoint_store: Session-scoped checkpoint store (M11). When provided,
            write tools snapshot pre-write state and an ``undo`` tool is
            registered. When None, writes proceed without snapshots (backward
            compatible).
        bash_timeout: Wall-clock timeout (seconds) for each bash/git
            subprocess (M20). Defaults to 300s — long enough for a real
            dependency install, test run, or build to finish rather than
            being killed mid-command.
        web_fetch_headers: HTTP headers ``web_fetch`` sends on every
            request, overriding the default bot-identifying ``User-Agent``
            (``nullain.config.WebFetchConfig``). None uses the defaults.
    """
    tools: list[RegisteredTool] = []
    tools.extend(
        create_filesystem_tools(
            workspace_root,
            file_access_tracker=file_access_tracker,
            event_bus=event_bus,
            session_id=session_id,
            checkpoint_store=checkpoint_store,
        )
    )
    tools.append(
        create_bash_tool(
            workspace_root, sandbox=sandbox, sandbox_opts=sandbox_opts, timeout=bash_timeout
        )
    )
    tools.extend(
        create_git_tools(
            workspace_root, sandbox=sandbox, sandbox_opts=sandbox_opts, timeout=bash_timeout
        )
    )
    tools.append(create_web_fetch_tool(headers=web_fetch_headers))
    tools.append(create_ask_user_tool(ask_user_callback))
    # Tool discovery (P4.26): lets the agent search for and hydrate deferred-
    # schema MCP tools. Registered last so it is always present.
    tools.append(create_search_tools_tool(registry))
    if persistent_memory is not None:
        tools.extend(create_memory_tools(persistent_memory))

    for t in tools:
        registry.register(t)


__version__ = "0.1.0"

__all__ = [
    "AskUserCallback",
    "CheckpointStore",
    "FileAccessTracker",
    "create_ask_user_tool",
    "create_bash_tool",
    "create_filesystem_tools",
    "create_git_tools",
    "create_memory_tools",
    "create_search_tools_tool",
    "create_web_fetch_tool",
    "register_default_tools",
]
