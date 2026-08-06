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
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

from nullain_tools import register_default_tools

from nullain.agent.loop import AgentLoop
from nullain.agent.result import RunResult
from nullain.config import NullainSettings, load_settings
from nullain.events import BaseEvent, EventBus, EventStore
from nullain.hooks import HookManager, HooksConfig
from nullain.llm import LLMProvider, OllamaCloudProvider
from nullain.memory import EpisodicMemory, PersistentMemory
from nullain.router import ModelRouter
from nullain.tools import PermissionPolicy, ToolRegistry
from nullain.tools.sandbox import Sandbox, SandboxOptions, select_sandbox

#: Async callback invoked when a tool requires ASK-level permission. Receives
#: the tool name and a human-readable description; returns whether to allow.
PermissionCallback = Callable[[str, str], Awaitable[bool]]
#: Async callback backing the ``ask_user`` tool. Receives a question; returns
#: the user's answer.
AskUserCallback = Callable[[str], Awaitable[str]]


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
        max_steps: int = 25,
        timeout: float = 300.0,
        permission_callback: PermissionCallback | None = None,
        ask_user_callback: AskUserCallback | None = None,
        event_bus: EventBus | None = None,
        event_store: EventStore | None = None,
        episodic_memory: EpisodicMemory | None = None,
        persistent_memory: PersistentMemory | None = None,
        hooks: HookManager | HooksConfig | None = None,
    ) -> None:
        """Assemble the agent's collaborators with safe defaults.

        Args:
            settings: Resolved settings; when None, loaded from ``nullain.toml``
                (or ``NULLAIN_CONFIG``) via :func:`load_settings`.
            provider: LLM provider; when None, an ``OllamaCloudProvider`` is
                built from ``settings.ollama_*``.
            tools: Tool registry; when None, a fresh registry is built with the
                built-in tools under a fail-closed permission policy.
            workspace_root: Workspace root for the filesystem/bash/git tools.
            model: Optional model override for the run.
            max_steps: Maximum ReAct steps per run.
            timeout: Per-run timeout in seconds.
            permission_callback: Async callback for ASK-level permission
                requests. When None, ASK resolves to DENY (fail-closed).
            ask_user_callback: Async callback backing the ``ask_user`` tool.
                When None, ``ask_user`` returns an error when invoked.
            event_bus: Event bus; when None, a fresh one is created.
            event_store: Persistent event store; when None, an in-memory one is
                created and initialized on first run.
            episodic_memory: Episodic memory; when None, a default SQLite one is
                created and initialized on first run.
            persistent_memory: Persistent memory; when None, one is created
                scoped to ``workspace_root``.
            hooks: Lifecycle hooks; a ``HooksConfig`` is wrapped in a
                ``HookManager``. When None, ``settings.hooks`` is used.
        """
        self._settings = settings or load_settings()
        self._workspace_root = str(workspace_root)
        self._model = model
        self._max_steps = max_steps
        self._timeout = timeout

        self._provider = provider or OllamaCloudProvider(
            api_key=self._settings.ollama_api_key,
            base_url=self._settings.ollama_base_url,
        )
        self._sandbox: Sandbox = select_sandbox(self._settings.sandbox)
        self.event_bus = event_bus or EventBus()
        self._event_store = event_store or EventStore()
        self._episodic_memory = episodic_memory or EpisodicMemory()
        self._persistent_memory = persistent_memory or PersistentMemory(
            workspace_root=self._workspace_root
        )
        self._router = ModelRouter(config=self._settings.router)
        if isinstance(hooks, HookManager):
            self._hooks = hooks
        else:
            self._hooks = HookManager(hooks or self._settings.hooks)

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
        max_steps: int = 25,
        timeout: float = 300.0,
        permission_callback: PermissionCallback | None = None,
        ask_user_callback: AskUserCallback | None = None,
        event_bus: EventBus | None = None,
        event_store: EventStore | None = None,
        episodic_memory: EpisodicMemory | None = None,
        persistent_memory: PersistentMemory | None = None,
        hooks: HookManager | HooksConfig | None = None,
    ) -> Agent:
        """Build an ``Agent`` from an already-resolved :class:`NullainSettings`.

        Args:
            settings: The resolved settings to assemble from.
            provider: Optional provider override.
            tools: Optional tool registry override.
            workspace_root: Optional workspace root override.
            model: Optional model override.
            max_steps: Optional max-steps override.
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
            timeout=timeout,
            permission_callback=permission_callback,
            ask_user_callback=ask_user_callback,
            event_bus=event_bus,
            event_store=event_store,
            episodic_memory=episodic_memory,
            persistent_memory=persistent_memory,
            hooks=hooks,
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
        max_steps: int = 25,
        timeout: float = 300.0,
        permission_callback: PermissionCallback | None = None,
        ask_user_callback: AskUserCallback | None = None,
        event_bus: EventBus | None = None,
        event_store: EventStore | None = None,
        episodic_memory: EpisodicMemory | None = None,
        persistent_memory: PersistentMemory | None = None,
        hooks: HookManager | HooksConfig | None = None,
    ) -> Agent:
        """Build an ``Agent`` from a ``nullain.toml`` config file path.

        Args:
            path: Path to the TOML configuration file.
            provider: Optional provider override.
            tools: Optional tool registry override.
            workspace_root: Optional workspace root override.
            model: Optional model override.
            max_steps: Optional max-steps override.
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
            timeout=timeout,
            permission_callback=permission_callback,
            ask_user_callback=ask_user_callback,
            event_bus=event_bus,
            event_store=event_store,
            episodic_memory=episodic_memory,
            persistent_memory=persistent_memory,
            hooks=hooks,
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
            timeout=self._timeout,
        )

    async def run(self, prompt: str, session_id: str | None = None) -> RunResult:
        """Run the agent on a single prompt and return the structured result.

        Args:
            prompt: The user prompt to act on.
            session_id: Optional session identifier; a fresh one is generated
                when omitted.

        Returns:
            The :class:`RunResult` describing how the run terminated.
        """
        await self._event_store.initialize()
        await self._episodic_memory.initialize()
        loop = self._build_loop(session_id)
        return await loop.run_result(prompt, session_id=session_id)

    async def stream(
        self, prompt: str, session_id: str | None = None
    ) -> AsyncIterator[BaseEvent | RunResult]:
        """Stream the run's events, ending with the final :class:`RunResult`.

        Yields each :class:`BaseEvent` published to the event bus as the run
        progresses, then yields the terminal :class:`RunResult` once the loop
        finishes.

        Args:
            prompt: The user prompt to act on.
            session_id: Optional session identifier.

        Yields:
            :class:`BaseEvent` instances during the run, then the final
            :class:`RunResult`.
        """
        await self._event_store.initialize()
        await self._episodic_memory.initialize()
        queue: asyncio.Queue[BaseEvent] = asyncio.Queue()

        async def forward(ev: BaseEvent) -> None:
            queue.put_nowait(ev)

        self.event_bus.subscribe("*", forward)
        try:
            loop = self._build_loop(session_id)
            task = asyncio.create_task(loop.run_result(prompt, session_id=session_id))
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


__all__ = ["Agent", "AskUserCallback", "PermissionCallback"]
