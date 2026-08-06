"""Unit tests for P4.27 — workflow orchestrator (deterministic over subagents).

100% offline: a fake :class:`WorkflowSpawner` stands in for the LLM, so the
orchestration structure (agent / parallel / pipeline / phase / log) is exercised
without any provider. The determinism property under test: the script fixes the
fan-out/pipeline structure; only the subagent outputs vary.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from nullain.authority import Authority, Capability
from nullain.events import EventBus, WorkflowAgentEvent, WorkflowLogEvent, WorkflowPhaseEvent
from nullain.workflow import Workflow, WorkflowContext, loop_spawner


class FakeSpawner:
    """Scripted subagent port: returns a canned text per prompt."""

    def __init__(self, outputs: dict[str, str] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        prompt: str,
        *,
        label: str | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        authority: Authority | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "label": label,
                "model": model,
                "max_steps": max_steps,
                "authority": authority,
            }
        )
        return self.outputs.get(prompt, f"result:{prompt}")


def _ctx(spawner: FakeSpawner, *, bus: EventBus | None = None) -> WorkflowContext:
    return WorkflowContext(
        spawner,
        {"input": 1},
        workflow_name="wf",
        event_bus=bus,
        session_id="s1",
    )


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_returns_spawner_output() -> None:
    spawner = FakeSpawner({"do it": "done"})
    ctx = _ctx(spawner)
    assert await ctx.agent("do it") == "done"


@pytest.mark.asyncio
async def test_agent_passes_authority_and_model() -> None:
    spawner = FakeSpawner()
    ctx = _ctx(spawner)
    auth = Authority.only(frozenset({Capability.READ}), can_spawn=False)
    await ctx.agent("p", model="m1", max_steps=5, authority=auth)
    call = spawner.calls[0]
    assert call["model"] == "m1"
    assert call["max_steps"] == 5
    assert call["authority"] is auth


# ---------------------------------------------------------------------------
# parallel (barrier)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_runs_concurrently_and_barriers() -> None:
    spawner = FakeSpawner()
    ctx = _ctx(spawner)
    order: list[str] = []

    async def slow() -> str:
        await asyncio.sleep(0.01)
        order.append("slow")
        return "s"

    async def fast() -> str:
        order.append("fast")
        return "f"

    results = await ctx.parallel([slow, fast])
    # Barrier: both awaited before return; results in input order.
    assert results == ["s", "f"]
    assert order == ["fast", "slow"]  # fast finished first, but results are ordered


@pytest.mark.asyncio
async def test_parallel_resolves_thunk_exception_to_none() -> None:
    ctx = _ctx(FakeSpawner())

    async def boom() -> str:
        raise RuntimeError("boom")

    async def ok() -> str:
        return "ok"

    results = await ctx.parallel([boom, ok])
    assert results == [None, "ok"]  # the call never rejects


# ---------------------------------------------------------------------------
# pipeline (no barrier)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_runs_stages_per_item_no_barrier() -> None:
    ctx = _ctx(FakeSpawner())
    seen: list[tuple[Any, Any, int]] = []

    def stage1(prev: Any, item: Any, index: int) -> str:
        seen.append((prev, item, index))
        return f"{prev}-s1"

    def stage2(prev: Any, item: Any, index: int) -> str:
        seen.append((prev, item, index))
        return f"{prev}-s2"

    results = await ctx.pipeline(["a", "b"], stage1, stage2)
    assert results == ["a-s1-s2", "b-s1-s2"]
    # Each stage saw (prev, original_item, index).
    assert ("a", "a", 0) in seen
    assert ("a-s1", "a", 0) in seen


@pytest.mark.asyncio
async def test_pipeline_drops_item_when_stage_throws() -> None:
    ctx = _ctx(FakeSpawner())

    def stage1(prev: Any, item: Any, index: int) -> str:
        if item == "bad":
            raise ValueError("nope")
        return f"{prev}-ok"

    def stage2(prev: Any, item: Any, index: int) -> str:
        return f"{prev}-s2"

    results = await ctx.pipeline(["good", "bad"], stage1, stage2)
    # "bad" dropped to None and skipped stage2; "good" ran both stages.
    assert results == ["good-ok-s2", None]


@pytest.mark.asyncio
async def test_pipeline_supports_async_stages() -> None:
    ctx = _ctx(FakeSpawner())

    async def stage(prev: Any, item: Any, index: int) -> str:
        return f"{prev}-async"

    results = await ctx.pipeline(["x"], stage)
    assert results == ["x-async"]


# ---------------------------------------------------------------------------
# progress events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_and_log_emit_events() -> None:
    bus = EventBus()
    events: list[Any] = []

    async def handler(ev: Any) -> None:
        events.append(ev)

    bus.subscribe("*", handler)
    ctx = _ctx(FakeSpawner(), bus=bus)
    await ctx.phase("Scan")
    await ctx.log("hello")
    assert any(isinstance(e, WorkflowPhaseEvent) and e.phase == "Scan" for e in events)
    assert any(isinstance(e, WorkflowLogEvent) and e.message == "hello" for e in events)


@pytest.mark.asyncio
async def test_agent_emits_start_and_complete_events() -> None:
    bus = EventBus()
    events: list[Any] = []

    async def handler(ev: Any) -> None:
        events.append(ev)

    bus.subscribe("*", handler)
    ctx = _ctx(FakeSpawner({"p": "out"}), bus=bus)
    await ctx.agent("p", label="L")
    agent_events = [e for e in events if isinstance(e, WorkflowAgentEvent)]
    assert [e.status for e in agent_events] == ["started", "completed"]
    assert agent_events[1].output == "out"
    assert all(e.label == "L" for e in agent_events)


# ---------------------------------------------------------------------------
# Workflow runner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_run_returns_fn_result() -> None:
    async def fn(ctx: WorkflowContext) -> str:
        await ctx.phase("Scan")
        out = await ctx.agent("p")
        return f"final:{out}"

    wf = Workflow(name="wf", description="d", fn=fn)
    result = await wf.run(FakeSpawner({"p": "x"}), args={"a": 1})
    assert result == "final:x"


@pytest.mark.asyncio
async def test_workflow_context_exposes_args() -> None:
    async def fn(ctx: WorkflowContext) -> Any:
        return ctx.args

    wf = Workflow(name="wf", description="d", fn=fn)
    assert await wf.run(FakeSpawner(), args={"k": "v"}) == {"k": "v"}


# ---------------------------------------------------------------------------
# loop_spawner adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_spawner_delegates_to_agent_loop_spawn() -> None:
    class FakeLoop:
        def __init__(self) -> None:
            self.spawned: list[tuple[str, dict[str, Any]]] = []

        async def spawn(self, prompt: str, **kwargs: Any) -> str:
            self.spawned.append((prompt, kwargs))
            return f"spawned:{prompt}"

    loop = FakeLoop()
    spawner = loop_spawner(loop)
    out = await spawner("hello", model="m")
    assert out == "spawned:hello"
    # loop_spawner forwards model/max_steps/authority (None when unset).
    assert loop.spawned == [("hello", {"model": "m", "max_steps": None, "authority": None})]
