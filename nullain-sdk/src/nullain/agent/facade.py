"""Nullain Agent SDK — Agent facade (public entry point).

The :class:`Agent` class is the primary entry point for the vast majority of
users. It assembles the collaborators (LLM provider, tool registry, permission
policy, model router, sandbox, memory) with safe defaults so a caller can go
from zero to a running agent in a few lines without touching the internal
wiring. It never reimplements the loop — it delegates to :class:`AgentLoop`.

The facade is async-first (principle 5 of the engineering plan): ``run`` and
``stream`` are coroutines, and ``run_sync`` is a thin synchronous facade for
scripts that wraps ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

from nullain_tools import register_default_tools
from structlog.stdlib import BoundLogger

from nullain.agent.loop import AgentLoop
from nullain.agent.result import RunResult
from nullain.config import NullainSettings, load_settings
from nullain.events import BaseEvent, EventBus, EventStore, EventStorePort, repair_session_events
from nullain.hooks import HookManager, HooksConfig
from nullain.llm import LLMProvider, OllamaCloudProvider, OpenAICompatibleProvider
from nullain.memory import EpisodicMemory, PersistentMemory
from nullain.quota import QuotaChecker
from nullain.router import ModelRouter
from nullain.telemetry import get_logger
from nullain.tools import PermissionPolicy, ToolRegistry
from nullain.tools.sandbox import Sandbox, SandboxOptions, select_sandbox

logger: BoundLogger = get_logger("nullain.agent.facade")

#: Async callback invoked when a tool requires ASK-level permission. Receives
#: the tool name and a human-readable description; returns whether to allow.
PermissionCallback = Callable[[str, str], Awaitable[bool]]
#: Async callback backing the ``ask_user`` tool. Receives a question; returns
#: the user's answer.
AskUserCallback = Callable[[str], Awaitable[str]]


class _Unset:
    """Sentinel distinguishing "argument omitted" from "explicitly None".

    ``max_tokens=None`` is a meaningful, valid value (disable the token
    ceiling) — so ``max_tokens`` can't default to ``None`` to mean "fall
    back to settings.agent.max_tokens" without losing the ability to pass
    ``None`` on purpose. This sentinel is the omitted-argument default
    instead.
    """


_UNSET = _Unset()


class Agent:
    """High-level facade assembling a runnable agent with safe defaults.

    The constructor builds every collaborator from ``settings`` (loaded from
    ``nullain.toml`` when not injected): an ``OllamaCloudProvider``, a
    ``ToolRegistry`` populated with the built-in tools under a fail-closed
    ``PermissionPolicy``, a ``ModelRouter``, and the platform-selected sandbox.
    Each collaborator can be overridden by injection for tests or custom wiring.

    Attributes:
        settings: The resolved :class:`NullainSettings`.
        tools: The :class:`ToolRegistry` the agent executes against.
        event_bus: The :class:`EventBus` events are published to.
    """

    def __init__(
        self,
        *,
        settings: NullainSettings | None = None,
        provider: LLMProvider | None = None,
        tools: ToolRegistry | None = None,
        workspace_root: str | Path = ".",
        model: str | None = None,
        max_steps: int | None = None,
        max_tokens: int | _Unset | None = _UNSET,
        timeout: float | None = None,
        permission_callback: PermissionCallback | None = None,
        ask_user_callback: AskUserCallback | None = None,
        event_bus: EventBus | None = None,
        event_store: EventStorePort | None = None,
        episodic_memory: EpisodicMemory | None = None,
        persistent_memory: PersistentMemory | None = None,
        hooks: HookManager | HooksConfig | None = None,
        quota_checker: QuotaChecker | None = None,
    ) -> None:
        """Assemble the agent's collaborators with safe defaults.

        Args:
            settings: Resolved settings; when None, loaded from ``nullain.toml``
                (or ``NULLAIN_CONFIG``) via :func:`load_settings`.
            provider: LLM provider; when None, built from ``settings.llm.provider``
                (default ``"ollama"`` — ``OllamaCloudProvider`` from
                ``settings.ollama_*``; ``"openai"`` — ``OpenAICompatibleProvider``
                from ``settings.openai_*``, issue #40).
            tools: Tool registry; when None, a fresh registry is built with the
                built-in tools under a fail-closed permission policy.
            workspace_root: Workspace root for the filesystem/bash/git tools.
            model: Optional model override for the run.
            max_steps: Maximum ReAct steps per run. When None, uses
                ``settings.agent.max_steps`` (``[agent] max_steps`` in
                ``nullain.toml``; SDK default 25).
            max_tokens: Cumulative token budget ceiling for one run — not a
                per-message limit, so it scales with how much a task
                actually needs across every step. When omitted entirely,
                uses ``settings.agent.max_tokens`` (``[agent] max_tokens``
                in ``nullain.toml``; SDK default 2,000,000 — large enough
                for real multi-file feature work, several rounds of
                self-correction, and debugging, while still stopping a
                genuinely runaway loop). Pass ``None`` explicitly to disable
                the ceiling for this ``Agent`` regardless of what
                ``nullain.toml`` says — the run is then bounded only by
                ``max_steps`` and ``timeout``.
            timeout: Per-run timeout in seconds. When None, uses
                ``settings.agent.timeout`` (``[agent] timeout`` in
                ``nullain.toml``; SDK default 300.0).
            permission_callback: Async callback for ASK-level permission
                requests. When None, ASK resolves to DENY (fail-closed).
            ask_user_callback: Async callback backing the ``ask_user`` tool.
                When None, ``ask_user`` returns an error when invoked.
            event_bus: Event bus; when None, a fresh one is created.
            event_store: Persistent event store; when None, a SQLite store at
                ``<workspace_root>/.nullain/sessions.db`` is created (and
                initialized on first run), so a session's trajectory survives
                the process exiting — the same directory persistent memory
                and checkpoints already use. Pass ``EventStore(":memory:")``
                explicitly to opt out (e.g. for tests, or one-shot scripting
                where nothing should be written to disk).
            episodic_memory: Episodic memory; when None, a default SQLite one is
                created and initialized on first run.
            persistent_memory: Persistent memory; when None, one is created
                scoped to ``workspace_root``.
            hooks: Lifecycle hooks; a ``HooksConfig`` is wrapped in a
                ``HookManager``. When None, ``settings.hooks`` is used.
            quota_checker: Optional per-tenant quota enforcement (ADR-4,
                docs/FUSION_PLAN.md), consulted before each step. When None
                (the default), no quota enforcement occurs.
        """
        self._workspace_root = str(workspace_root)
        # Resolve settings relative to workspace_root, not the process's own
        # cwd: load_settings(None) only checks NULLAIN_CONFIG then
        # ./nullain.toml relative to cwd, which silently misses a
        # nullain.toml in workspace_root whenever the two differ (e.g. a
        # daemon or long-lived process operating on a workspace other than
        # where it was launched from). NULLAIN_CONFIG is resolved here first
        # so it still wins, matching load_settings(None)'s own precedence.
        if settings is not None:
            self._settings = settings
        else:
            config_path = os.environ.get("NULLAIN_CONFIG") or str(
                Path(self._workspace_root).resolve() / "nullain.toml"
            )
            self._settings = load_settings(config_path)
        self._model = model
        self._max_steps = max_steps if max_steps is not None else self._settings.agent.max_steps
        self._max_tokens = (
            self._settings.agent.max_tokens if isinstance(max_tokens, _Unset) else max_tokens
        )
        self._timeout = timeout if timeout is not None else self._settings.agent.timeout

        self._provider = provider or self._build_default_provider()
        self._sandbox: Sandbox = select_sandbox(self._settings.sandbox)
        self.event_bus = event_bus or EventBus()
        self._event_store = event_store or EventStore(
            Path(self._workspace_root).resolve() / ".nullain" / "sessions.db"
        )
        self._episodic_memory = episodic_memory or EpisodicMemory()
        self._persistent_memory = persistent_memory or PersistentMemory(
            workspace_root=self._workspace_root
        )
        self._router = ModelRouter(config=self._settings.router)
        if isinstance(hooks, HookManager):
            self._hooks = hooks
        else:
            self._hooks = HookManager(hooks or self._settings.hooks)
        self._quota_checker = quota_checker

        if tools is not None:
            self.tools = tools
        else:
            policy = PermissionPolicy(workspace_root=self._workspace_root)
            self.tools = ToolRegistry(
                permission_policy=policy,
                permission_callback=permission_callback,
            )
            register_default_tools(
                self.tools,
                self._workspace_root,
                ask_user_callback=ask_user_callback,
                persistent_memory=self._persistent_memory,
                sandbox=self._sandbox,
                sandbox_opts=SandboxOptions(
                    workspace_root=Path(self._workspace_root),
                    allow_paths=[Path(p) for p in self._settings.sandbox.allow_paths],
                    deny_network=self._settings.sandbox.deny_network,
                ),
                event_bus=self.event_bus,
                bash_timeout=self._settings.agent.bash_timeout,
            )

    def _build_default_provider(self) -> LLMProvider:
        """Build the provider named by ``settings.llm.provider`` (issue #40).

        ``"ollama"`` (default) builds ``OllamaCloudProvider`` from the
        top-level ``ollama_api_key``/``ollama_base_url`` settings, preserving
        every existing config's behavior exactly (this branch is what ran
        unconditionally before #40). ``"openai"`` builds
        ``OpenAICompatibleProvider`` from the top-level
        ``openai_api_key``/``openai_base_url`` settings — and, via
        ``openai_base_url``, works against any OpenAI-compatible endpoint,
        not only OpenAI itself. An unrecognized value raises rather than
        silently falling back to Ollama, since a typo'd provider name
        silently talking to the wrong (or no) API would be a confusing
        failure mode far downstream of the config file.
        """
        provider_name = self._settings.llm.provider
        if provider_name == "ollama":
            return OllamaCloudProvider(
                api_key=self._settings.ollama_api_key,
                base_url=self._settings.ollama_base_url,
            )
        if provider_name == "openai":
            return OpenAICompatibleProvider(
                api_key=self._settings.openai_api_key,
                base_url=self._settings.openai_base_url,
            )
        raise ValueError(
            f"Unknown [llm] provider {provider_name!r} in settings — expected 'ollama' or 'openai'"
        )

    @classmethod
    def from_settings(
        cls,
        settings: NullainSettings,
        *,
        provider: LLMProvider | None = None,
        tools: ToolRegistry | None = None,
        workspace_root: str | Path = ".",
        model: str | None = None,
        max_steps: int | None = None,
        max_tokens: int | _Unset | None = _UNSET,
        timeout: float | None = None,
        permission_callback: PermissionCallback | None = None,
        ask_user_callback: AskUserCallback | None = None,
        event_bus: EventBus | None = None,
        event_store: EventStorePort | None = None,
        episodic_memory: EpisodicMemory | None = None,
        persistent_memory: PersistentMemory | None = None,
        hooks: HookManager | HooksConfig | None = None,
        quota_checker: QuotaChecker | None = None,
    ) -> Agent:
        """Build an ``Agent`` from an already-resolved :class:`NullainSettings`.

        Args:
            settings: The resolved settings to assemble from.
            provider: Optional provider override.
            tools: Optional tool registry override.
            workspace_root: Optional workspace root override.
            model: Optional model override.
            max_steps: Optional max-steps override.
            max_tokens: Optional cumulative token-budget override; see the
                constructor's docstring for the omitted-vs-None distinction.
            timeout: Optional per-run timeout override.
            permission_callback: Optional permission callback override.
            ask_user_callback: Optional ask-user callback override.
            event_bus: Optional event bus override.
            event_store: Optional event store override.
            episodic_memory: Optional episodic memory override.
            persistent_memory: Optional persistent memory override.
            hooks: Optional hooks override.

        Returns:
            A configured :class:`Agent`.
        """
        return cls(
            settings=settings,
            provider=provider,
            tools=tools,
            workspace_root=workspace_root,
            model=model,
            max_steps=max_steps,
            max_tokens=max_tokens,
            timeout=timeout,
            permission_callback=permission_callback,
            ask_user_callback=ask_user_callback,
            event_bus=event_bus,
            event_store=event_store,
            episodic_memory=episodic_memory,
            persistent_memory=persistent_memory,
            hooks=hooks,
            quota_checker=quota_checker,
        )

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        provider: LLMProvider | None = None,
        tools: ToolRegistry | None = None,
        workspace_root: str | Path = ".",
        model: str | None = None,
        max_steps: int | None = None,
        max_tokens: int | _Unset | None = _UNSET,
        timeout: float | None = None,
        permission_callback: PermissionCallback | None = None,
        ask_user_callback: AskUserCallback | None = None,
        event_bus: EventBus | None = None,
        event_store: EventStorePort | None = None,
        episodic_memory: EpisodicMemory | None = None,
        persistent_memory: PersistentMemory | None = None,
        hooks: HookManager | HooksConfig | None = None,
        quota_checker: QuotaChecker | None = None,
    ) -> Agent:
        """Build an ``Agent`` from a ``nullain.toml`` config file path.

        Args:
            path: Path to the TOML configuration file.
            provider: Optional provider override.
            tools: Optional tool registry override.
            workspace_root: Optional workspace root override.
            model: Optional model override.
            max_steps: Optional max-steps override.
            max_tokens: Optional cumulative token-budget override; see the
                constructor's docstring for the omitted-vs-None distinction.
            timeout: Optional per-run timeout override.
            permission_callback: Optional permission callback override.
            ask_user_callback: Optional ask-user callback override.
            event_bus: Optional event bus override.
            event_store: Optional event store override.
            episodic_memory: Optional episodic memory override.
            persistent_memory: Optional persistent memory override.
            hooks: Optional hooks override.

        Returns:
            A configured :class:`Agent`.
        """
        return cls(
            settings=load_settings(path),
            provider=provider,
            tools=tools,
            workspace_root=workspace_root,
            model=model,
            max_steps=max_steps,
            max_tokens=max_tokens,
            timeout=timeout,
            permission_callback=permission_callback,
            ask_user_callback=ask_user_callback,
            event_bus=event_bus,
            event_store=event_store,
            episodic_memory=episodic_memory,
            persistent_memory=persistent_memory,
            hooks=hooks,
            quota_checker=quota_checker,
        )

    def _build_loop(self, session_id: str | None) -> AgentLoop:
        """Construct an :class:`AgentLoop` from the assembled collaborators."""
        return AgentLoop(
            provider=self._provider,
            tools=self.tools,
            event_bus=self.event_bus,
            event_store=self._event_store,
            model=self._model,
            router=self._router,
            episodic_memory=self._episodic_memory,
            hooks=self._hooks,
            persistent_memory=self._persistent_memory,
            workspace_root=self._workspace_root,
            max_steps=self._max_steps,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
            quota_checker=self._quota_checker,
        )

    async def run(self, prompt: str, session_id: str | None = None) -> RunResult:
        """Run the agent on a single prompt and return the structured result.

        When ``session_id`` names a session with prior events already in the
        event store, those events are loaded as ``events_history`` so the run
        continues the conversation (the model sees prior turns, tool
        results, etc.) instead of starting fresh with the same id — matching
        what "continuing a session" means. A ``session_id`` with no prior
        events (including a fresh, unset one) starts empty as before.

        Args:
            prompt: The user prompt to act on.
            session_id: Optional session identifier; a fresh one is generated
                when omitted.

        Returns:
            The :class:`RunResult` describing how the run terminated.
        """
        await self._event_store.initialize()
        await self._episodic_memory.initialize()
        events_history = await self._load_session_history(session_id)
        loop = self._build_loop(session_id)
        return await loop.run_result(prompt, session_id=session_id, events_history=events_history)

    async def _load_session_history(self, session_id: str | None) -> list[BaseEvent] | None:
        """Load prior events for ``session_id`` from the event store, if any.

        Applies the pre-#24 orphaned-tool-result repair pass (see
        ``nullain.events.repair``) before returning: a session persisted by a
        build older than #24's compaction-boundary fix can carry a
        ``CompactionEvent`` that split a tool-call turn from its results,
        which would otherwise replay into a message history Ollama Cloud
        rejects with ``400 invalid message content type: <nil>`` on every
        resume attempt. Uncorrupted sessions take the fast no-op path inside
        ``repair_session_events`` — no extra write, no extra event — so this
        is free for every session created after the fix.

        A repair is never silent: it is logged, published on the event bus,
        and persisted to the store as a ``SessionRepairedEvent`` so it shows
        up in that session's history and in ``nullain doctor``.
        """
        if session_id is None:
            return None
        events = await self._event_store.get_session_events(session_id)
        if not events:
            return None
        repaired, report = repair_session_events(session_id, events)
        if report is None:
            return repaired
        logger.warning(
            "Repaired corrupted session history",
            session_id=session_id,
            re_paired_call_ids=report.re_paired_call_ids,
            dropped_call_ids=report.dropped_call_ids,
        )
        await self._event_store.append(report)
        await self.event_bus.publish(report)
        return [*repaired, report]

    async def stream(
        self, prompt: str, session_id: str | None = None
    ) -> AsyncIterator[BaseEvent | RunResult]:
        """Stream the run's events, ending with the final :class:`RunResult`.

        Yields each :class:`BaseEvent` published to the event bus as the run
        progresses, then yields the terminal :class:`RunResult` once the loop
        finishes.

        Args:
            prompt: The user prompt to act on.
            session_id: Optional session identifier. As with :meth:`run`, a
                session_id with prior events in the event store resumes that
                conversation (see :meth:`run` for details).

        Yields:
            :class:`BaseEvent` instances during the run, then the final
            :class:`RunResult`.
        """
        await self._event_store.initialize()
        await self._episodic_memory.initialize()
        events_history = await self._load_session_history(session_id)
        queue: asyncio.Queue[BaseEvent] = asyncio.Queue()

        async def forward(ev: BaseEvent) -> None:
            queue.put_nowait(ev)

        self.event_bus.subscribe("*", forward)
        try:
            loop = self._build_loop(session_id)
            task = asyncio.create_task(
                loop.run_result(prompt, session_id=session_id, events_history=events_history)
            )
            while not task.done():
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=0.1)
                except TimeoutError:
                    continue
                yield ev
            result = await task
            # Drain any events published between the last poll and completion.
            while not queue.empty():
                yield queue.get_nowait()
            yield result
        finally:
            self.event_bus.unsubscribe("*", forward)

    def run_sync(self, prompt: str, session_id: str | None = None) -> RunResult:
        """Synchronous facade over :meth:`run` for scripts.

        Args:
            prompt: The user prompt to act on.
            session_id: Optional session identifier.

        Returns:
            The :class:`RunResult` describing how the run terminated.
        """
        return asyncio.run(self.run(prompt, session_id=session_id))

    async def close(self) -> None:
        """Release the SQLite connections opened by ``run``/``stream``.

        ``EventStore`` and ``EpisodicMemory`` each open an ``aiosqlite``
        connection lazily on first use, backed by a dedicated worker thread
        that stays alive until ``.close()`` runs. Nothing in ``run``/
        ``stream`` closes them automatically — a script that calls one of
        those and exits without calling ``close()`` first leaves those
        threads parked waiting on the event loop, which blocks the
        interpreter's own shutdown from completing cleanly (observed
        building the ``nullain-agent`` bridge: the process would sit
        indefinitely after a fully successful run, never producing an
        error, because two live ``aiosqlite`` worker threads kept the
        interpreter from exiting).

        Safe to call even if neither store was ever initialized (each
        ``close()`` is itself a no-op on an unopened connection). Idempotent.
        """
        await self._event_store.close()
        await self._episodic_memory.close()


__all__ = ["Agent", "AskUserCallback", "PermissionCallback"]
