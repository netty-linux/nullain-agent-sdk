"""Nullain Agent SDK — Workflow definition and runner (P4.27).

A :class:`Workflow` is a named, described Python function that orchestrates
subagents deterministically. ``run`` binds a :class:`WorkflowSpawner` and an
optional event bus into a :class:`WorkflowContext`, then awaits the function
and returns its result. The orchestration structure is fixed by the script;
only the subagent outputs are non-deterministic.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from nullain.events import EventBus
from nullain.workflow.context import WorkflowContext
from nullain.workflow.spawner import WorkflowSpawner

#: A workflow function: async, receives a WorkflowContext, returns any value.
WorkflowFn = Callable[[WorkflowContext], Awaitable[Any]]


class Workflow:
    """A deterministic orchestration of subagents.

    Args:
        name: Short identifier (used in progress events).
        description: One-line summary (shown in listings).
        fn: The async workflow function. Receives a :class:`WorkflowContext`
            and returns the workflow's result.
    """

    def __init__(self, name: str, description: str, fn: WorkflowFn) -> None:
        self.name = name
        self.description = description
        self.fn = fn

    async def run(
        self,
        spawner: WorkflowSpawner,
        *,
        args: Any = None,
        event_bus: EventBus | None = None,
        session_id: str = "workflow",
    ) -> Any:
        """Execute the workflow and return its result.

        Args:
            spawner: The subagent port backing ``ctx.agent()``.
            args: Input value exposed as ``ctx.args``.
            event_bus: Optional bus for progress events.
            session_id: Session id stamped on emitted events.

        Returns:
            Whatever the workflow function returns.
        """
        ctx = WorkflowContext(
            spawner,
            args,
            workflow_name=self.name,
            event_bus=event_bus,
            session_id=session_id,
        )
        return await self.fn(ctx)


__all__ = ["Workflow", "WorkflowFn"]
