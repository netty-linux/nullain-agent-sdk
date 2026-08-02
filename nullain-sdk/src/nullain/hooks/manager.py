"""Nullain Agent SDK — Lifecycle Hooks.

Mirrors Claude Code's hook lifecycle convention: external commands run at
fixed lifecycle points (``pre_tool``, ``post_tool``, ``stop``,
``pre_compact``), receive the event payload as JSON on stdin, and signal via
exit code:

* ``0``  — success; stdout (if any) is attached as additional context.
* ``2``  — **block**: for ``pre_tool`` the tool call is skipped, for
  ``pre_compact`` the compaction is skipped, for ``stop`` the stop is vetoed.
  stdout is surfaced as the block reason.
* other — non-blocking failure (logged, execution continues unchanged).

Hooks are configured declaratively in ``nullain.toml`` under ``[hooks]`` and
executed by :class:`HookManager`. The :class:`~nullain.agent.loop.AgentLoop`
consults the manager at each lifecycle point and honors ``blocked`` outcomes
for the pre-lifecycle hooks.
"""

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from nullain.telemetry import get_logger

logger = get_logger(__name__)

#: Exit code a hook uses to veto the lifecycle action it wraps.
BLOCK_EXIT_CODE = 2


class HookLifecycle(StrEnum):
    """The four lifecycle points where hooks can fire."""

    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    STOP = "stop"
    PRE_COMPACT = "pre_compact"


class HookConfig(BaseModel):
    """A single command hook bound to one lifecycle point.

    ``command`` is an argv list (never shell-interpreted). ``env`` is merged
    over the parent process environment when spawning.
    """

    command: list[str]
    timeout: float = 30.0
    env: dict[str, str] = Field(default_factory=dict)


class HooksConfig(BaseModel):
    """All hooks grouped by lifecycle point."""

    pre_tool: list[HookConfig] = Field(default_factory=lambda: list[HookConfig]())
    post_tool: list[HookConfig] = Field(default_factory=lambda: list[HookConfig]())
    stop: list[HookConfig] = Field(default_factory=lambda: list[HookConfig]())
    pre_compact: list[HookConfig] = Field(default_factory=lambda: list[HookConfig]())

    def has_any(self) -> bool:
        """True if at least one hook is configured for any lifecycle point."""
        return any(bool(getattr(self, lf.value)) for lf in HookLifecycle)


@dataclass(slots=True)
class HookOutcome:
    """Result of running a single hook command."""

    lifecycle: HookLifecycle
    exit_code: int
    stdout: str
    stderr: str
    blocked: bool
    additional_context: str | None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class HookManager:
    """Runs configured lifecycle hooks.

    The manager is safe to construct with no configuration (all hooks become
    no-ops), so it can be wired into the AgentLoop unconditionally.
    """

    def __init__(self, config: HooksConfig | None = None) -> None:
        self.config = config or HooksConfig()

    @property
    def enabled(self) -> bool:
        """True if any hook is configured (cheap pre-check for callers)."""
        return self.config.has_any()

    def for_lifecycle(self, lifecycle: HookLifecycle) -> list[HookConfig]:
        return list(getattr(self.config, lifecycle.value))

    async def run(
        self,
        lifecycle: HookLifecycle,
        payload: dict[str, Any],
    ) -> list[HookOutcome]:
        """Run all hooks for a lifecycle point in configured order.

        Stops at the first blocking outcome (exit 2): a block decision is
        terminal for that lifecycle point, so later hooks are skipped.
        Returns the outcomes of the hooks that actually ran.
        """
        hooks = self.for_lifecycle(lifecycle)
        if not hooks:
            return []
        full_payload = {**payload, "lifecycle": lifecycle.value}
        outcomes: list[HookOutcome] = []
        for hook in hooks:
            outcome = await self._run_one(hook, lifecycle, full_payload)
            outcomes.append(outcome)
            if outcome.blocked:
                logger.info(
                    "hook_blocked",
                    lifecycle=lifecycle.value,
                    command=hook.command,
                    reason=outcome.stdout.strip() or None,
                )
                break
        return outcomes

    async def _run_one(
        self,
        hook: HookConfig,
        lifecycle: HookLifecycle,
        payload: dict[str, Any],
    ) -> HookOutcome:
        cmd = hook.command
        if not cmd or not cmd[0]:
            return HookOutcome(lifecycle, 0, "", "", False, None)
        env = {**os.environ, **hook.env} if hook.env else None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as err:
            logger.warning("hook_spawn_failed", command=cmd, error=str(err))
            return HookOutcome(lifecycle, -1, "", str(err), False, None)

        stdin_data = json.dumps(payload).encode()
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=hook.timeout,
            )
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            logger.warning("hook_timeout", command=cmd, timeout=hook.timeout)
            return HookOutcome(lifecycle, -1, "", "timeout", False, None)

        exit_code = proc.returncode if proc.returncode is not None else -1
        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        blocked = exit_code == BLOCK_EXIT_CODE
        additional = stdout.strip() or None if exit_code == 0 else None
        if exit_code not in (0, BLOCK_EXIT_CODE):
            logger.warning(
                "hook_nonzero",
                command=cmd,
                exit_code=exit_code,
                stderr=stderr.strip() or None,
            )
        return HookOutcome(lifecycle, exit_code, stdout, stderr, blocked, additional)


__all__ = [
    "BLOCK_EXIT_CODE",
    "HookConfig",
    "HookLifecycle",
    "HookManager",
    "HookOutcome",
    "HooksConfig",
]
