"""Nullain Agent SDK — Workflow subagent spawner port (P4.27).

A :class:`WorkflowSpawner` runs one subagent and returns its final text. The
orchestrator depends on this port, not on :class:`~nullain.agent.loop.AgentLoop`
directly, so workflows are testable with a fake spawner and decoupled from the
loop. :func:`loop_spawner` adapts an ``AgentLoop`` (delegating to
``AgentLoop.spawn``, which enforces the P4.24 authority-intersection law).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nullain.authority import Authority


@runtime_checkable
class WorkflowSpawner(Protocol):
    """Port: run one subagent and return its final answer text."""

    async def __call__(
        self,
        prompt: str,
        *,
        label: str | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        authority: Authority | None = None,
    ) -> str:
        """Run a subagent.

        Args:
            prompt: The subtask prompt.
            label: Optional human-readable label for progress reporting.
            model: Explicit model (bypasses routing) when not None.
            max_steps: Step cap when not None.
            authority: Delegated authority (P4.24) when not None.

        Returns:
            The subagent's final answer text.
        """
        ...


def loop_spawner(loop: Any) -> WorkflowSpawner:
    """Adapt an :class:`~nullain.agent.loop.AgentLoop` into a :class:`WorkflowSpawner`.

    Each ``agent()`` call in a workflow delegates to ``loop.spawn``, so the
    subagent inherits the loop's provider/workspace and the P4.24 authority
    intersection applies.
    """

    async def spawn(
        prompt: str,
        *,
        label: str | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        authority: Authority | None = None,
    ) -> str:
        return await loop.spawn(
            prompt,
            model=model,
            max_steps=max_steps,
            authority=authority,
        )

    return spawn


__all__ = ["WorkflowSpawner", "loop_spawner"]
