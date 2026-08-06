"""Nullain Agent SDK — Workflow execution context (P4.27).

:class:`WorkflowContext` is the DSL surface handed to a workflow function. It
exposes the deterministic orchestration primitives — ``agent``, ``parallel``,
``pipeline`` — plus ``phase``/``log`` for progress and ``args`` for input. The
orchestration structure is fixed by the script; only the subagent outputs are
non-deterministic (they are LLM calls).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from nullain.authority import Authority
from nullain.events import EventBus, WorkflowAgentEvent, WorkflowLogEvent, WorkflowPhaseEvent
from nullain.workflow.spawner import WorkflowSpawner

#: A stage in a pipeline: receives (prev_result, original_item, index) and
#: returns the next result. May be sync or async.
PipelineStage = Callable[[Any, Any, int], Any]


class WorkflowContext:
    """Execution context passed to a workflow function.

    Args:
        spawner: The subagent port backing ``agent()``.
        args: The workflow's input value, exposed verbatim as ``ctx.args``.
        workflow_name: Name used in emitted progress events.
        event_bus: Optional bus for progress events (phase/log/agent).
        session_id: Session id stamped on emitted events.
    """

    def __init__(
        self,
        spawner: WorkflowSpawner,
        args: Any,
        *,
        workflow_name: str,
        event_bus: EventBus | None = None,
        session_id: str = "workflow",
    ) -> None:
        self.args = args
        self._spawner = spawner
        self._workflow_name = workflow_name
        self._event_bus = event_bus
        self._session_id = session_id
        self._phase: str | None = None

    # -- progress -----------------------------------------------------------

    async def phase(self, title: str) -> None:
        """Set the current phase (groups progress reporting)."""
        self._phase = title
        if self._event_bus is not None:
            await self._event_bus.publish(
                WorkflowPhaseEvent(
                    session_id=self._session_id,
                    workflow=self._workflow_name,
                    phase=title,
                )
            )

    async def log(self, message: str) -> None:
        """Emit a progress message."""
        if self._event_bus is not None:
            await self._event_bus.publish(
                WorkflowLogEvent(
                    session_id=self._session_id,
                    workflow=self._workflow_name,
                    message=message,
                )
            )

    # -- orchestration ------------------------------------------------------

    async def agent(
        self,
        prompt: str,
        *,
        label: str | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        authority: Authority | None = None,
    ) -> str:
        """Spawn a subagent and return its final answer text.

        The subagent runs with fresh context and (when ``authority`` is
        delegated) an authority intersected with its own surface and the policy
        (P4.24). ``label`` is used for progress reporting.
        """
        label = label or prompt[:60]
        await self._emit_agent(label, "started")
        try:
            output = await self._spawner(
                prompt,
                label=label,
                model=model,
                max_steps=max_steps,
                authority=authority,
            )
        except Exception:
            await self._emit_agent(label, "failed")
            raise
        await self._emit_agent(label, "completed", output=output)
        return output

    async def parallel(self, thunks: list[Callable[[], Awaitable[Any]]]) -> list[Any]:
        """Run thunks concurrently and await all before returning (a barrier).

        A thunk that throws resolves to ``None`` in the result array; the call
        itself never rejects. Results are returned in input order.
        """
        results = await asyncio.gather(*(t() for t in thunks), return_exceptions=True)
        return [r if not isinstance(r, BaseException) else None for r in results]

    async def pipeline(
        self,
        items: list[Any],
        *stages: PipelineStage,
    ) -> list[Any]:
        """Run each item through all stages with no barrier between stages.

        Item A can be in stage 3 while item B is still in stage 1. Each stage
        receives ``(prev_result, original_item, index)``. A stage that throws
        drops that item to ``None`` and skips its remaining stages.
        """

        async def run_item(item: Any, index: int) -> Any:
            result: Any = item
            for stage in stages:
                try:
                    result = stage(result, item, index)
                    if asyncio.iscoroutine(result):
                        result = await result
                except Exception:
                    return None
            return result

        return await asyncio.gather(*(run_item(item, i) for i, item in enumerate(items)))

    # -- internals ----------------------------------------------------------

    async def _emit_agent(self, label: str, status: str, output: str | None = None) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            WorkflowAgentEvent(
                session_id=self._session_id,
                workflow=self._workflow_name,
                label=label,
                status=status,  # type: ignore[arg-type]
                output=output,
            )
        )


__all__ = ["PipelineStage", "WorkflowContext"]
