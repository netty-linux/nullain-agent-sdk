"""Nullain Agent Daemon — Stdio NDJSON Daemon process."""

import asyncio
import contextlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, cast

from nullain.agent import AgentLoop
from nullain.authority import Capability
from nullain.config import NullainSettings, load_settings
from nullain.events import BaseEvent, EventBus, EventStore, repair_session_events
from nullain.hooks import HookManager
from nullain.llm import LLMProvider, OllamaCloudProvider
from nullain.lsp import LSPClient, register_lsp_tools
from nullain.mcp import MCPClient, StdioTransport, register_mcp_tools
from nullain.memory import EpisodicMemory, PersistentMemory
from nullain.plugins import PluginLoader, PreparedPlugin, select_verifier
from nullain.protocol import (
    AgentEventPayload,
    AskUserRequestPayload,
    PermissionRequestPayload,
    ProtocolEnvelope,
)
from nullain.router import ModelRouter
from nullain.telemetry import configure_telemetry, get_logger
from nullain.tools import PermissionPolicy, ToolRegistry
from nullain.tools.sandbox import SandboxOptions, select_sandbox
from nullain_tools import CheckpointStore, FileAccessTracker, register_default_tools

logger = get_logger(__name__)


@dataclass
class _SessionState:
    """Per-session mutable state (issue #43).

    Everything a running session needs that must NOT be shared with any
    other concurrent session — workspace root, tool registry, permission
    policy, and persistent memory. Found live: an earlier version of this
    daemon kept these as single closure-local variables reassigned on every
    ``session.start``, so a second concurrent session silently clobbered the
    first session's workspace/registry the moment it started — any
    ``user.message`` for session A, sent after session B started, ran
    against B's tools and workspace instead of A's. Keying this state by
    ``session_id`` in a dict (see ``run_agentd``'s ``sessions`` map) is the
    fix: shared collaborators (provider, MCP/LSP clients, prepared plugins,
    sandbox, router, event bus/store, episodic memory) still persist across
    sessions as before — only what genuinely must be per-session lives here.
    """

    workspace_root: str
    registry: ToolRegistry
    permission_policy: PermissionPolicy
    persistent_memory: PersistentMemory


def _sandbox_opts(settings: NullainSettings, ws_root: str) -> SandboxOptions:
    """Build per-session sandbox options from settings + the session workspace."""
    return SandboxOptions(
        workspace_root=Path(ws_root),
        allow_paths=[Path(p) for p in settings.sandbox.allow_paths],
        deny_network=settings.sandbox.deny_network,
    )


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


async def _load_lsp_clients(settings: NullainSettings) -> dict[str, LSPClient]:
    """Start the LSP servers declared in settings and return them by language.

    Mirrors :func:`_load_mcp_clients`: a server that fails to initialize is
    logged and skipped — one unavailable language server must not prevent the
    daemon from serving the rest of its tools. The error is surfaced as a
    structured log (AGENTS.md rule 5).
    """
    clients: dict[str, LSPClient] = {}
    for language, server_cfg in settings.lsp.servers.items():
        if not server_cfg.enabled:
            continue
        client = LSPClient(
            command=server_cfg.command,
            args=server_cfg.args,
            env=server_cfg.env,
            name=language,
        )
        try:
            await client.initialize()
            clients[language] = client
        except Exception as err:
            logger.warning(
                "lsp_server_init_failed",
                language=language,
                command=server_cfg.command,
                error=str(err),
            )
            with contextlib.suppress(Exception):
                await client.close()
    return clients


def _cap_set(names: list[str] | None) -> frozenset[Capability]:
    """Parse capability names from config; ``None``/empty => the universe (all).

    An unset operator grant means "no narrowing" (the plugin gets its full
    declared set, intersected at load), matching the Authority meet semantics
    where the universe is the identity element.
    """
    if not names:
        return frozenset(Capability)
    return frozenset(Capability(n.lower()) for n in names)


async def _load_plugins(settings: NullainSettings) -> list[PreparedPlugin]:
    """Verify, prepare, and initialize the plugins declared in settings.

    Mirrors :func:`_load_mcp_clients`: the plugin subprocesses are prepared once
    at startup and shared across sessions (the live MCP client persists); only
    the lightweight per-session tool wrappers are re-created. A plugin that
    fails any pipeline step (signature, SBOM, capability, MCP init) is logged
    and skipped — fail-closed per plugin, never fatal to the daemon. This is
    the AGENTS.md rule 5 structured-log discipline and the P4.25 trust policy:
    an unverified or over-capable plugin is refused, not silently degraded.
    """
    pcfg = settings.plugins
    if not pcfg.enabled:
        return []
    verifier = select_verifier()
    global_grant = _cap_set(pcfg.allowed_capabilities)
    prepared: list[PreparedPlugin] = []
    for name, entry in pcfg.entries.items():
        if not entry.enabled:
            continue
        # Per-entry grant narrows the operator's global grant (intersection) —
        # an entry may never widen what [plugins].allowed_capabilities grants.
        grant = global_grant
        if entry.allowed_capabilities is not None:
            grant = global_grant & _cap_set(entry.allowed_capabilities)
        loader = PluginLoader(
            verifier,
            require_signature=pcfg.require_signature,
            trusted_keys=dict(pcfg.trusted_keys),
            allowed_capabilities=grant,
            auto_approve_default=entry.auto_approve,
        )
        try:
            manifest = PluginLoader.parse_manifest_file(entry.manifest)
            prepared.append(await loader.prepare(manifest))
        except Exception as err:
            logger.warning(
                "plugin_load_failed",
                plugin=name,
                manifest=entry.manifest,
                error=str(err),
            )
    return prepared


async def run_agentd(
    *,
    provider: LLMProvider | None = None,
    input_reader: asyncio.StreamReader | None = None,
    output: TextIO | None = None,
    settings: NullainSettings | None = None,
    mcp_clients: list[MCPClient] | None = None,
    lsp_clients: dict[str, LSPClient] | None = None,
    prepared_plugins: list[PreparedPlugin] | None = None,
    episodic_memory: EpisodicMemory | None = None,
    event_store: EventStore | None = None,
) -> None:
    """Async main loop reading NDJSON from stdin and writing responses to stdout.

    Keyword-only injection points exist for testability: ``provider``,
    ``input_reader``, ``output``, ``settings``, ``mcp_clients``,
    ``prepared_plugins``, ``episodic_memory`` and ``event_store`` can be supplied
    to drive the daemon in-process without spawning a real Ollama client, MCP
    subprocesses, or touching the user's home-directory SQLite stores. In
    production all default to their real values.
    """
    configure_telemetry(log_level="INFO", json_format=True)

    # Settings: injected (tests) or resolved from NULLAIN_CONFIG / cwd toml.
    if settings is None:
        config_path = os.environ.get("NULLAIN_CONFIG")
        if config_path is None and Path("nullain.toml").exists():
            config_path = "nullain.toml"
        settings = load_settings(config_path)

    # Sandbox: select the platform adapter once. When isolation is enabled and
    # required but the adapter is unavailable on this host, fail-closed would
    # refuse every bash/git call at execution time — surface that at startup so
    # the operator can fix the host (or set sandbox.required=false to opt out).
    sandbox = select_sandbox(settings.sandbox)
    if settings.sandbox.enabled and sandbox.required and not sandbox.available():
        logger.warning(
            "sandbox_required_but_unavailable",
            sandbox=sandbox.name,
            hint="set [sandbox] required=false to run unconfined, or enable the adapter",
        )

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

    # LSP clients: injected (tests) or started from settings. Shared across
    # sessions like MCP clients — the language-server subprocesses persist.
    if lsp_clients is None:
        lsp_clients = await _load_lsp_clients(settings)

    # Plugins: injected (tests) or prepared from settings. Shared across
    # sessions like MCP clients — the verified manifest + live MCP server are
    # prepared once; per-session registration only re-creates the wrappers.
    if prepared_plugins is None:
        prepared_plugins = await _load_plugins(settings)

    router = ModelRouter(config=settings.router)
    hook_manager = HookManager(settings.hooks)
    if event_store is None:
        event_store = EventStore()
    await event_store.initialize()
    if episodic_memory is None:
        episodic_memory = EpisodicMemory()
    await episodic_memory.initialize()
    event_bus = EventBus()

    # A single registrar for per-session plugin registration. register() only
    # uses each PreparedPlugin's baked-in effective_capabilities/declarations
    # (computed at prepare time with the per-entry grant); the loader's own
    # grant/verifier are NOT consulted at register time, so one shared instance
    # is correct and the per-entry narrowing is preserved.
    plugin_registrar = PluginLoader(select_verifier())

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

    # Per-session state (issue #43): keyed by session_id, never overwritten by
    # a different session starting — see _SessionState's docstring for the
    # concurrency bug this replaces.
    sessions: dict[str, _SessionState] = {}

    def _build_worktree_registry(
        worktree_root: Path, sess_id: str, sess_event_bus: EventBus
    ) -> ToolRegistry:
        """Build a fresh registry rooted at a worktree directory (M11.4).

        Tools bind to their workspace root at creation time, so a
        worktree-isolated subagent needs a registry built against the worktree
        path rather than the parent's. Captures the same permission/ask
        callbacks, sandbox, and session id as the parent session's registry.
        """
        wt_root = str(worktree_root)
        wt_policy = PermissionPolicy(workspace_root=wt_root)
        wt_registry = ToolRegistry(
            permission_policy=wt_policy,
            permission_callback=permission_callback,
        )
        register_default_tools(
            wt_registry,
            wt_root,
            ask_user_callback=ask_user_callback,
            persistent_memory=PersistentMemory(workspace_root=wt_root),
            sandbox=sandbox,
            sandbox_opts=_sandbox_opts(settings, wt_root),
            file_access_tracker=FileAccessTracker(),
            event_bus=sess_event_bus,
            session_id=sess_id,
            checkpoint_store=CheckpointStore(wt_root),
            bash_timeout=settings.agent.bash_timeout,
        )
        return wt_registry

    async def _load_session_history(sess_id: str) -> list[BaseEvent] | None:
        """Load and repair prior events for ``sess_id`` from the shared event
        store, so a ``session.start`` on an id with prior history resumes
        that conversation after a daemon restart (issue #43's acceptance
        criterion) — mirrors ``Agent._load_session_history`` (issue #44),
        including the same orphaned-tool-result repair for sessions
        persisted by a build older than #24's compaction fix.
        """
        events = await event_store.get_session_events(sess_id)
        if not events:
            return None
        repaired, report = repair_session_events(sess_id, events)
        if report is None:
            return repaired
        logger.warning(
            "agentd_session_repaired",
            session_id=sess_id,
            re_paired_call_ids=report.re_paired_call_ids,
            dropped_call_ids=report.dropped_call_ids,
        )
        await event_store.append(report)
        await event_bus.publish(report)
        return [*repaired, report]

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
                sess_id = str(env.payload.get("session_id", "s1"))
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
                    sandbox=sandbox,
                    sandbox_opts=_sandbox_opts(settings, ws_root),
                    # M8: a fresh read-tracker per session so edit_file requires
                    # a prior read within the same session; todo_write mirrors
                    # progress to the client via TodoEvent on the shared bus.
                    file_access_tracker=FileAccessTracker(),
                    event_bus=event_bus,
                    session_id=sess_id,
                    # M11: a fresh checkpoint store per session so write tools
                    # snapshot pre-write state and the undo tool can restore it.
                    checkpoint_store=CheckpointStore(ws_root),
                    bash_timeout=settings.agent.bash_timeout,
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
                # Register each prepared plugin's tools into the per-session
                # registry. The verifier/capability pipeline already ran at
                # prepare time; this only re-creates the namespaced wrappers and
                # applies the manifest's per-tool requires/permission_level. A
                # plugin that drops every tool at this stage (e.g. the operator
                # narrowed the grant between prepare and register) is skipped
                # with a structured log rather than crashing the session.
                for prepared in prepared_plugins:
                    try:
                        await plugin_registrar.register(prepared, registry)
                    except Exception as err:
                        logger.warning(
                            "plugin_register_failed",
                            plugin=prepared.manifest.name,
                            error=str(err),
                        )
                # Register the LSP-backed read-only tools into the per-session
                # registry. The shared language-server subprocesses persist
                # across sessions (like MCP clients); only the lightweight
                # wrappers are re-created. Fail-soft: an unavailable server or
                # unsupported file type surfaces as a ToolResult error, never a
                # session crash.
                register_lsp_tools(registry, lsp_clients)

                # Keyed by session_id — a second session.start for a
                # DIFFERENT id no longer clobbers this one's state (issue
                # #43). Restarting the SAME session_id rebuilds its registry
                # fresh (correct: tool wrappers/read-trackers/checkpoint
                # stores are process-local and don't survive anyway), while
                # its conversation history still resumes below from the
                # durable event store.
                sessions[sess_id] = _SessionState(
                    workspace_root=ws_root,
                    registry=registry,
                    permission_policy=policy,
                    persistent_memory=persistent_memory,
                )

                resp_env = ProtocolEnvelope(
                    v=1,
                    type="session.started",
                    id=env.id,
                    payload={
                        "session_id": sess_id,
                        "status": "ok",
                    },
                )
                write_envelope(resp_env)

            elif env.type == "user.message":
                prompt = str(env.payload.get("prompt", ""))
                sess_id = str(env.payload.get("session_id", "s1"))

                state = sessions.get(sess_id)
                if state is None:
                    err_env = ProtocolEnvelope(
                        v=1,
                        type="session.end",
                        id=env.id,
                        payload={
                            "session_id": sess_id,
                            "status": "error",
                            "error": (f"unknown session_id {sess_id!r} — send session.start first"),
                        },
                    )
                    write_envelope(err_env)
                    continue

                events_history = await _load_session_history(sess_id)

                agent = AgentLoop(
                    provider=provider,
                    tools=state.registry,
                    event_bus=event_bus,
                    event_store=event_store,
                    router=router,
                    episodic_memory=episodic_memory,
                    hooks=hook_manager,
                    persistent_memory=state.persistent_memory,
                    workspace_root=Path(state.workspace_root),
                    # M11.4: enables isolation="worktree" subagents by re-rooting
                    # a fresh registry at the worktree directory. The lambda
                    # binds this session's id and event bus (B023: loop var
                    # captured as a default).
                    tool_factory=lambda p, _sid=sess_id, _bus=event_bus: _build_worktree_registry(
                        p, _sid, _bus
                    ),
                )

                try:
                    res_text = await agent.run_streaming(
                        prompt=prompt, session_id=sess_id, events_history=events_history
                    )
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
        for prepared in prepared_plugins:
            with contextlib.suppress(Exception):
                await prepared.client.close()
        await episodic_memory.close()
        await event_store.close()


def main() -> None:
    """Entry point for nullain-agentd."""
    asyncio.run(run_agentd())


if __name__ == "__main__":
    main()
