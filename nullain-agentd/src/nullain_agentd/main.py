"""Nullain Agent Daemon — Stdio NDJSON Daemon process."""

import asyncio
import contextlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, TextIO, cast

from nullain.agent import AgentLoop
from nullain.config import NullainSettings, load_settings
from nullain.events import BaseEvent, EventBus, EventStore
from nullain.hooks import HookManager
from nullain.llm import LLMProvider, OllamaCloudProvider
from nullain.mcp import MCPClient, StdioTransport, register_mcp_tools
from nullain.memory import EpisodicMemory, PersistentMemory
from nullain.protocol import (
    AgentEventPayload,
    AskUserRequestPayload,
    PermissionRequestPayload,
    ProtocolEnvelope,
)
from nullain.router import ModelRouter
from nullain.telemetry import configure_telemetry, get_logger
from nullain.tools import PermissionPolicy, ToolRegistry
from nullain_tools import register_default_tools

logger = get_logger(__name__)


async def _load_mcp_clients(settings: NullainSettings) -> list[MCPClient]:
    """Start the MCP servers declared in settings and return their clients.

    A server that fails to initialize is logged and skipped — one unavailable
    external tool server must not prevent the daemon from serving the rest of
    its tools. The error is surfaced as a structured log (AGENTS.md rule 5).
    """
    clients: list[MCPClient] = []
    for name, server_cfg in settings.mcp.servers.items():
        if not server_cfg.enabled:
            continue
        transport = StdioTransport(
            command=server_cfg.command,
            args=server_cfg.args,
            env=server_cfg.env,
        )
        client = MCPClient(transport=transport, name=name)
        try:
            await client.initialize()
            clients.append(client)
        except Exception as err:
            logger.warning(
                "mcp_server_init_failed",
                server=name,
                command=server_cfg.command,
                error=str(err),
            )
            with contextlib.suppress(Exception):
                await transport.close()
    return clients


async def run_agentd(
    *,
    provider: LLMProvider | None = None,
    input_reader: asyncio.StreamReader | None = None,
    output: TextIO | None = None,
    settings: NullainSettings | None = None,
    mcp_clients: list[MCPClient] | None = None,
    episodic_memory: EpisodicMemory | None = None,
    event_store: EventStore | None = None,
) -> None:
    """Async main loop reading NDJSON from stdin and writing responses to stdout.

    Keyword-only injection points exist for testability: ``provider``,
    ``input_reader``, ``output``, ``settings``, ``mcp_clients``,
    ``episodic_memory`` and ``event_store`` can be supplied to drive the daemon
    in-process without spawning a real Ollama client, MCP subprocesses, or
    touching the user's home-directory SQLite stores. In production all default
    to their real values.
    """
    configure_telemetry(log_level="INFO", json_format=True)

    # Settings: injected (tests) or resolved from NULLAIN_CONFIG / cwd toml.
    if settings is None:
        config_path = os.environ.get("NULLAIN_CONFIG")
        if config_path is None and Path("nullain.toml").exists():
            config_path = "nullain.toml"
        settings = load_settings(config_path)

    # Provider: injected (tests) or built from settings (production).
    if provider is None:
        provider = OllamaCloudProvider(
            api_key=settings.ollama_api_key,
            base_url=settings.ollama_base_url,
        )

    # MCP clients: injected (tests) or started from settings. Shared across
    # sessions — the subprocesses are expensive to spawn and persist.
    if mcp_clients is None:
        mcp_clients = await _load_mcp_clients(settings)

    router = ModelRouter(config=settings.router)
    hook_manager = HookManager(settings.hooks)
    if event_store is None:
        event_store = EventStore()
    await event_store.initialize()
    if episodic_memory is None:
        episodic_memory = EpisodicMemory()
    await episodic_memory.initialize()
    event_bus = EventBus()

    out: TextIO = output or sys.stdout

    async def emit_agent_event(ev: BaseEvent) -> None:
        """Forward agent events to stdout as NDJSON."""
        payload = AgentEventPayload(
            session_id=ev.session_id,
            event_type=ev.event_type,
            data=json.loads(ev.model_dump_json()),
        )
        env = ProtocolEnvelope(
            v=1,
            type="agent.event",
            payload=json.loads(payload.model_dump_json()),
        )
        out.write(env.model_dump_json() + "\n")
        out.flush()

    event_bus.subscribe("*", emit_agent_event)

    if input_reader is not None:
        reader = input_reader
    else:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    def write_envelope(env: ProtocolEnvelope) -> None:
        out.write(env.model_dump_json() + "\n")
        out.flush()

    async def permission_callback(tool_name: str, description: str) -> bool:
        """Emit a permission.request and await the matching permission.response.

        Blocks on stdin while the agent is paused awaiting approval. If the
        client closes the stream or sends an unparseable/non-response line,
        the action is DENIED (fail-closed).
        """
        request_id = str(uuid.uuid4())
        req_payload = PermissionRequestPayload(
            request_id=request_id,
            tool_name=tool_name,
            description=description,
        )
        req_env = ProtocolEnvelope(
            v=1,
            type="permission.request",
            payload=json.loads(req_payload.model_dump_json()),
        )
        write_envelope(req_env)

        line_bytes = await reader.readline()
        if not line_bytes:
            return False
        try:
            raw = cast(dict[str, Any], json.loads(line_bytes.decode("utf-8").strip()))
            resp_env = ProtocolEnvelope.model_validate(raw)
        except Exception:
            return False
        if resp_env.type != "permission.response":
            return False
        return bool(resp_env.payload.get("granted", False))

    async def ask_user_callback(question: str) -> str:
        """Emit an ask_user.request and await the matching ask_user.response.

        Fail-closed on EOF/parse error: returns an error string the agent can
        react to rather than hanging on a closed stdin.
        """
        request_id = str(uuid.uuid4())
        req_payload = AskUserRequestPayload(request_id=request_id, question=question)
        req_env = ProtocolEnvelope(
            v=1,
            type="ask_user.request",
            payload=json.loads(req_payload.model_dump_json()),
        )
        write_envelope(req_env)

        line_bytes = await reader.readline()
        if not line_bytes:
            return "Error: user interaction channel closed (no answer)."
        try:
            raw = cast(dict[str, Any], json.loads(line_bytes.decode("utf-8").strip()))
            resp_env = ProtocolEnvelope.model_validate(raw)
        except Exception:
            return "Error: unparseable response to ask_user."
        if resp_env.type != "ask_user.response":
            return "Error: expected ask_user.response."
        return str(resp_env.payload.get("answer", ""))

    ws_root = "."
    policy = PermissionPolicy(workspace_root=ws_root)
    registry: ToolRegistry = ToolRegistry(
        permission_policy=policy,
        permission_callback=permission_callback,
    )
    persistent_memory: PersistentMemory | None = None
    register_default_tools(registry, ws_root, ask_user_callback=ask_user_callback)

    try:
        while True:
            line_bytes = await reader.readline()
            if not line_bytes:
                break

            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue

            try:
                raw_dict = cast(dict[str, Any], json.loads(line))
                env = ProtocolEnvelope.model_validate(raw_dict)
            except Exception as err:
                err_env = ProtocolEnvelope(
                    v=1,
                    type="error",
                    payload={"message": f"Invalid NDJSON envelope: {err}"},
                )
                write_envelope(err_env)
                continue

            if env.type == "session.start":
                ws_root = str(env.payload.get("workspace_root", "."))
                policy = PermissionPolicy(workspace_root=ws_root)
                registry = ToolRegistry(
                    permission_policy=policy,
                    permission_callback=permission_callback,
                )
                persistent_memory = PersistentMemory(workspace_root=ws_root)
                register_default_tools(
                    registry,
                    ws_root,
                    ask_user_callback=ask_user_callback,
                    persistent_memory=persistent_memory,
                )
                # Register each MCP server's tools into the fresh per-session
                # registry. The shared client/subprocess persists across
                # sessions; only the lightweight wrappers are re-created.
                for client in mcp_clients:
                    server_cfg = settings.mcp.servers.get(client.name)
                    auto_approve = bool(server_cfg and server_cfg.auto_approve)
                    try:
                        await register_mcp_tools(registry, client, auto_approve=auto_approve)
                    except Exception as err:
                        logger.warning(
                            "mcp_tools_register_failed",
                            server=client.name,
                            error=str(err),
                        )

                resp_env = ProtocolEnvelope(
                    v=1,
                    type="session.started",
                    id=env.id,
                    payload={
                        "session_id": env.payload.get("session_id", "s1"),
                        "status": "ok",
                    },
                )
                write_envelope(resp_env)

            elif env.type == "user.message":
                prompt = str(env.payload.get("prompt", ""))
                sess_id = str(env.payload.get("session_id", "s1"))

                agent = AgentLoop(
                    provider=provider,
                    tools=registry,
                    event_bus=event_bus,
                    event_store=event_store,
                    router=router,
                    episodic_memory=episodic_memory,
                    hooks=hook_manager,
                    persistent_memory=persistent_memory,
                    workspace_root=Path(ws_root),
                )

                try:
                    res_text = await agent.run_streaming(prompt=prompt, session_id=sess_id)
                    end_env = ProtocolEnvelope(
                        v=1,
                        type="session.end",
                        id=env.id,
                        payload={
                            "session_id": sess_id,
                            "status": "completed",
                            "output": res_text,
                        },
                    )
                    out.write(end_env.model_dump_json() + "\n")
                    out.flush()
                except Exception as err:
                    err_env = ProtocolEnvelope(
                        v=1,
                        type="session.end",
                        id=env.id,
                        payload={
                            "session_id": sess_id,
                            "status": "error",
                            "error": str(err),
                        },
                    )
                    out.write(err_env.model_dump_json() + "\n")
                    out.flush()
    finally:
        for client in mcp_clients:
            with contextlib.suppress(Exception):
                await client.close()
        await episodic_memory.close()
        await event_store.close()


def main() -> None:
    """Entry point for nullain-agentd."""
    asyncio.run(run_agentd())


if __name__ == "__main__":
    main()
