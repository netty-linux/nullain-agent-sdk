"""Nullain Agent SDK — Command Line Interface.

A thin ``argparse`` CLI that delegates to the :class:`Agent` facade. It is a
delegator, not a reimplementation: ``run`` and ``chat`` build an ``Agent`` and
call its methods; ``doctor`` probes the environment; ``mcp`` manages the MCP
servers declared in ``nullain.toml``.

Exit codes: ``0`` success, ``1`` runtime failure (a run that did not succeed, a
failed health check), ``2`` usage/config error. ``--json`` emits NDJSON (one
object per line) consumable by scripts. Secrets are never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from nullain import __version__
from nullain.agent import Agent, RunResult
from nullain.config import load_settings
from nullain.events import EventStore
from nullain.llm import OllamaCloudProvider
from nullain.tools.sandbox import select_sandbox
from nullain.tui import TUIRenderer

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2

#: Top-level TOML section header for MCP servers.
_MCP_SECTION = "[mcp.servers]"


# ---------------------------------------------------------------------------
# TOML editing for `mcp add|remove` (stdlib only, no new dependency)
# ---------------------------------------------------------------------------


def _find_config_path() -> Path:
    """Resolve the config file path (NULLAIN_CONFIG, then ./nullain.toml)."""
    env = os.environ.get("NULLAIN_CONFIG")
    if env:
        return Path(env)
    return Path("nullain.toml")


def _parse_mcp_servers(text: str) -> dict[str, dict[str, Any]]:
    """Parse the ``[mcp.servers.<name>]`` tables from a TOML string."""
    servers: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"^\[mcp\.servers\.([A-Za-z0-9_\-]+)\]\s*$", stripped)
        if m:
            name = m.group(1)
            current = name
            servers.setdefault(name, {})
            continue
        if current is None:
            continue
        kv = re.match(r"^([A-Za-z0-9_]+)\s*=\s*(.*)$", stripped)
        if not kv:
            continue
        key, raw = kv.group(1), kv.group(2)
        try:
            servers[current][key] = json.loads(raw)
        except json.JSONDecodeError:
            servers[current][key] = raw.strip("\"'")
    return servers


def _serialize_mcp_servers(servers: dict[str, dict[str, Any]]) -> str:
    """Serialize the ``[mcp.servers]`` section to TOML text."""
    lines = [_MCP_SECTION]
    for name, cfg in servers.items():
        lines.append(f"[mcp.servers.{name}]")
        lines.append(f'command = "{cfg.get("command", "")}"')
        args = cfg.get("args")
        if args:
            lines.append(f"args = {json.dumps(args)}")
        env = cfg.get("env")
        if env:
            lines.append(f"env = {json.dumps(env)}")
        lines.append(f"auto_approve = {str(bool(cfg.get('auto_approve', False))).lower()}")
        lines.append(f"enabled = {str(bool(cfg.get('enabled', True))).lower()}")
    return "\n".join(lines)


def _replace_mcp_section(text: str, new_section: str) -> str:
    """Replace the ``[mcp.servers]`` block in ``text`` with ``new_section``.

    Preserves everything before the section and every top-level section after
    it. If the section is absent, ``new_section`` is appended at the end.
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == _MCP_SECTION), None)
    if start is None:
        body = text.rstrip("\n")
        return f"{body}\n\n{new_section}\n" if body else f"{new_section}\n"
    # Find the next top-level section not under mcp.servers.
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^\[(?!mcp\.servers)", lines[i].strip()):
            end = i
            break
    prefix = "\n".join(lines[:start]).rstrip("\n")
    suffix = "\n".join(lines[end:]).lstrip("\n")
    parts = [prefix, new_section]
    if suffix:
        parts.append(suffix)
    return "\n\n".join(parts) + "\n"


def _edit_mcp_server(
    name: str,
    *,
    command: str | None = None,
    args: list[str] | None = None,
    auto_approve: bool | None = None,
    enabled: bool | None = None,
    remove: bool = False,
) -> None:
    """Add, update, or remove an MCP server entry in the config file."""
    path = _find_config_path()
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    servers = _parse_mcp_servers(text)
    if remove:
        servers.pop(name, None)
    else:
        entry = servers.get(name, {})
        if command is not None:
            entry["command"] = command
        if args is not None:
            entry["args"] = args
        if auto_approve is not None:
            entry["auto_approve"] = auto_approve
        if enabled is not None:
            entry["enabled"] = enabled
        servers[name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_replace_mcp_section(text, _serialize_mcp_servers(servers)), encoding="utf-8")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


async def _resolve_session_id(
    workspace: str, *, session_id: str | None, continue_session: bool
) -> str | None:
    """Resolve the effective session id for ``run``/``chat`` (M16).

    - An explicit ``--session <id>`` always wins.
    - ``--continue`` looks up the most recently used session id in this
      workspace's event store (``<workspace>/.nullain/sessions.db``) and
      resumes it; with no prior sessions, falls through to a fresh one
      (there is nothing to continue, so this is not an error).
    - Otherwise ``None`` — a fresh session id is generated internally by the
      loop, matching the pre-M16 default of always starting empty.
    """
    if session_id:
        return session_id
    if not continue_session:
        return None
    store = EventStore(Path(workspace).resolve() / ".nullain" / "sessions.db")
    try:
        return await store.get_latest_session_id()
    finally:
        await store.close()


async def _run(
    prompt: str,
    *,
    model: str | None,
    workspace: str,
    max_steps: int,
    json_output: bool,
    session_id: str | None = None,
    continue_session: bool = False,
) -> RunResult:
    """Execute a single prompt and return the structured result.

    Always drives the run through ``agent.stream()`` — a single pass over the
    event stream — rather than a separate ``agent.run()`` call, so the run
    executes exactly once regardless of output mode. ``--json`` emits NDJSON
    for scripts; otherwise the Rich ``TUIRenderer`` renders live (streamed
    text, tool-call spinners, colored diffs) instead of a bare final-text
    print, matching Claude Code / Gemini CLI's terminal UX.

    Session persistence (M16): a resolved session id (explicit ``--session``
    or the latest one via ``--continue``) is passed through to
    ``agent.stream()``, which resumes that session's history from the
    workspace's on-disk event store when it has prior events.
    """
    agent = Agent(workspace_root=workspace, model=model, max_steps=max_steps)
    resolved_session = await _resolve_session_id(
        workspace, session_id=session_id, continue_session=continue_session
    )
    result: RunResult | None = None
    renderer = None if json_output else TUIRenderer()
    async for item in agent.stream(prompt, session_id=resolved_session):
        if isinstance(item, RunResult):
            result = item
            if json_output:
                print(json.dumps({"type": "result", **item.model_dump()}))
            continue
        if json_output:
            print(
                json.dumps(
                    {
                        "type": "event",
                        "event_type": item.event_type,
                        "data": json.loads(item.model_dump_json()),
                    }
                )
            )
        elif renderer is not None:
            renderer.handle(item)
    if renderer is not None:
        renderer.finish()
    assert result is not None  # agent.stream() always yields a terminal RunResult
    return result


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


async def _chat(*, model: str | None, workspace: str, continue_session: bool = False) -> int:
    """Run an interactive multi-turn session with TTY permission approval.

    Each turn is driven through ``agent.stream()`` and rendered live via
    ``TUIRenderer`` — streamed text, tool-call spinners, and colored
    write_file/edit_file diffs — instead of only printing the final answer
    once the whole turn has finished.

    Session persistence (M16): every turn shares one ``session_id`` for the
    duration of this chat process, so turn 2 sees turn 1's exchange in its
    context (previously each turn generated its own fresh id — a chat
    session's later turns never actually saw earlier ones). With
    ``continue_session``, that shared id is the workspace's most recently
    used session instead of a new one, so a chat opened yesterday can be
    picked back up. The session id is echoed at startup so the user can
    pass it to ``nullain run --session <id>`` later if they want a
    non-interactive continuation.
    """
    agent = Agent(
        workspace_root=workspace,
        model=model,
        permission_callback=_tty_permission,
        ask_user_callback=_tty_ask_user,
    )
    session_id = await _resolve_session_id(
        workspace, session_id=None, continue_session=continue_session
    )
    resuming = session_id is not None
    if session_id is None:
        session_id = str(uuid.uuid4())
    renderer = TUIRenderer()
    print("Nullain chat. Type 'exit' or Ctrl-D to quit.")
    print(f"Session: {session_id}{' (resumed)' if resuming else ''}")
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            print()
            return EXIT_OK
        if not prompt.strip():
            continue
        if prompt.strip().lower() in ("exit", "quit"):
            return EXIT_OK
        renderer.reset()
        async for item in agent.stream(prompt, session_id=session_id):
            renderer.handle(item)
        renderer.finish()


async def _tty_permission(tool_name: str, description: str) -> bool:
    """Prompt the user on the TTY for an ASK-level permission request."""
    try:
        answer = input(f"Allow {tool_name}? ({description}) [y/N]: ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


async def _tty_ask_user(question: str) -> str:
    """Prompt the user on the TTY for an ``ask_user`` question."""
    try:
        return input(f"nullain: {question}\n> ")
    except EOFError:
        return "Error: user interaction channel closed (no answer)."


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


async def _doctor() -> int:
    """Run environment and health checks, returning a non-zero exit on failure."""
    checks: list[tuple[str, bool, str]] = []
    try:
        settings = load_settings()
        checks.append(("config", True, "nullain.toml parsed"))
    except Exception as err:
        checks.append(("config", False, str(err)))
        settings = None

    if settings is not None:
        provider = OllamaCloudProvider(
            api_key=settings.ollama_api_key, base_url=settings.ollama_base_url
        )
        try:
            healthy = await provider.health_check()
            checks.append(("provider", healthy, settings.ollama_base_url))
        except Exception as err:
            checks.append(("provider", False, str(err)))

        sandbox = select_sandbox(settings.sandbox)
        available = sandbox.available() if sandbox.required else True
        checks.append(("sandbox", available, sandbox.name))

        mcp_ok = True
        for name, cfg in settings.mcp.servers.items():
            if not cfg.enabled:
                continue
            if not cfg.command:
                mcp_ok = False
                checks.append((f"mcp:{name}", False, "no command configured"))
        if mcp_ok:
            checks.append(("mcp", True, f"{len(settings.mcp.servers)} server(s) declared"))

    rg = shutil.which("rg")
    checks.append(("ripgrep", rg is not None, rg or "not found"))

    failed = False
    for name, ok, detail in checks:
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {name}: {detail}")
        failed = failed or not ok
    return EXIT_RUNTIME if failed else EXIT_OK


# ---------------------------------------------------------------------------
# mcp
# ---------------------------------------------------------------------------


def _mcp_list() -> int:
    """List the MCP servers declared in the config file."""
    settings = load_settings()
    if not settings.mcp.servers:
        print("No MCP servers declared.")
        return EXIT_OK
    for name, cfg in settings.mcp.servers.items():
        args = " ".join(cfg.args) if cfg.args else ""
        print(
            f"{name}: {cfg.command} {args} (enabled={cfg.enabled}, auto_approve={cfg.auto_approve})"
        )
    return EXIT_OK


def _mcp_add(args: argparse.Namespace) -> int:
    """Add or update an MCP server entry in the config file."""
    _edit_mcp_server(
        args.name,
        command=args.command,
        args=args.args,
        auto_approve=args.auto_approve,
        enabled=not args.disabled,
    )
    print(f"Added/updated MCP server '{args.name}' in {_find_config_path()}.")
    return EXIT_OK


def _mcp_remove(args: argparse.Namespace) -> int:
    """Remove an MCP server entry from the config file."""
    _edit_mcp_server(args.name, remove=True)
    print(f"Removed MCP server '{args.name}' from {_find_config_path()}.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# argparse app
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse command tree."""
    parser = argparse.ArgumentParser(
        prog="nullain",
        description="Nullain Agent SDK — run, chat, and manage an agent.",
    )
    parser.add_argument("--version", action="version", version=f"Nullain Agent SDK v{__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print the SDK version").set_defaults(handler=_version_handler)

    p_run = sub.add_parser("run", help="Run a single prompt and print the result")
    p_run.add_argument("prompt", help="The prompt to act on")
    p_run.add_argument("--model", default=None, help="Model override")
    p_run.add_argument("--workspace", default=".", help="Workspace root")
    p_run.add_argument("--max-steps", type=int, default=25, help="Max ReAct steps")
    p_run.add_argument("--json", action="store_true", help="Emit NDJSON for piping")
    p_run.add_argument(
        "--session",
        default=None,
        help="Resume a specific session id (see nullain chat's startup line)",
    )
    p_run.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Resume the workspace's most recently used session",
    )
    p_run.set_defaults(handler=_run_handler)

    p_chat = sub.add_parser("chat", help="Start an interactive multi-turn session")
    p_chat.add_argument("--model", default=None, help="Model override")
    p_chat.add_argument("--workspace", default=".", help="Workspace root")
    p_chat.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Resume the workspace's most recently used session instead of starting fresh",
    )
    p_chat.set_defaults(handler=_chat_handler)

    p_doctor = sub.add_parser("doctor", help="Run environment and health checks")
    p_doctor.set_defaults(handler=_doctor_handler)

    p_mcp = sub.add_parser("mcp", help="Manage MCP servers in nullain.toml")
    mcp_sub = p_mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_sub.add_parser("list", help="List declared MCP servers").set_defaults(
        handler=_mcp_list_handler
    )
    p_add = mcp_sub.add_parser("add", help="Add or update an MCP server")
    p_add.add_argument("name", help="Server name")
    p_add.add_argument("--command", required=True, help="Launch command")
    p_add.add_argument("--args", nargs="*", default=None, help="Command arguments")
    p_add.add_argument("--auto-approve", action="store_true", help="Auto-approve tool calls")
    p_add.add_argument("--disabled", action="store_true", help="Register as disabled")
    p_add.set_defaults(handler=_mcp_add_handler)
    p_rm = mcp_sub.add_parser("remove", help="Remove an MCP server")
    p_rm.add_argument("name", help="Server name")
    p_rm.set_defaults(handler=_mcp_remove_handler)

    return parser


def _version_handler(args: argparse.Namespace) -> int:
    """Handler for ``nullain version``."""
    print(f"Nullain Agent SDK v{__version__}")
    return EXIT_OK


def _run_handler(args: argparse.Namespace) -> int:
    """Handler for ``nullain run``."""
    result = asyncio.run(
        _run(
            args.prompt,
            model=args.model,
            workspace=args.workspace,
            max_steps=args.max_steps,
            json_output=args.json,
            session_id=args.session,
            continue_session=args.continue_session,
        )
    )
    return EXIT_OK if result.success else EXIT_RUNTIME


def _chat_handler(args: argparse.Namespace) -> int:
    """Handler for ``nullain chat``."""
    return asyncio.run(
        _chat(model=args.model, workspace=args.workspace, continue_session=args.continue_session)
    )


def _doctor_handler(args: argparse.Namespace) -> int:
    """Handler for ``nullain doctor``."""
    return asyncio.run(_doctor())


def _mcp_list_handler(args: argparse.Namespace) -> int:
    """Handler for ``nullain mcp list``."""
    return _mcp_list()


def _mcp_add_handler(args: argparse.Namespace) -> int:
    """Handler for ``nullain mcp add``."""
    return _mcp_add(args)


def _mcp_remove_handler(args: argparse.Namespace) -> int:
    """Handler for ``nullain mcp remove``."""
    return _mcp_remove(args)


def app() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        sys.exit(EXIT_USAGE)
    sys.exit(handler(args))


if __name__ == "__main__":
    app()
