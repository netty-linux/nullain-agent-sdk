"""Unit and E2E tests for AgentLoop ReAct Core."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from nullain.agent import AgentLoop
from nullain.errors import BudgetExceededError
from nullain.events import BaseEvent, ErrorEvent, EventBus
from nullain.llm import (
    CompletionChunk,
    CompletionRequest,
    LLMProvider,
    TokenUsage,
    ToolCall,
)
from nullain.tools import ToolRegistry
from nullain_tools import register_default_tools


class FakeSequenceProvider(LLMProvider):
    """Fake LLM Provider that yields a scripted sequence of responses."""

    def __init__(self, responses: list[CompletionChunk]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        if self.call_count < len(self.responses):
            chunk = self.responses[self.call_count]
            self.call_count += 1
            return chunk
        return CompletionChunk(delta_text="Default finished")

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        chunk = await self.generate(request)
        yield chunk

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_agent_loop_e2e_create_facts_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    # Step 0: Spec generation response for Plan phase
    spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "Create FACTS.txt file", '
            '"steps": ["Write 3 facts"], '
            '"target_files": ["FACTS.txt"], '
            '"acceptance_criteria": ["FACTS.txt exists"]}'
        )
    )

    # Step 1: Model calls write_file
    chunk1 = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="call_write_1",
                name="write_file",
                arguments={"path": "FACTS.txt", "content": "1. Fact A\n2. Fact B\n3. Fact C"},
            )
        ],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )

    # Step 2: Model returns final response text
    chunk2 = CompletionChunk(
        delta_text="Finished creating FACTS.txt with 3 facts.",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    fake_provider = FakeSequenceProvider([spec_chunk, chunk1, chunk2])
    bus = EventBus()
    events_log: list[BaseEvent] = []

    async def track_events(ev: BaseEvent) -> None:
        events_log.append(ev)

    bus.subscribe("*", track_events)

    agent = AgentLoop(
        provider=fake_provider,
        tools=registry,
        event_bus=bus,
        max_steps=5,
        workspace_root=workspace,
    )

    result_text = await agent.run("Create FACTS.txt file")
    assert "Finished creating FACTS.txt" in result_text
    assert (workspace / "FACTS.txt").exists()
    assert "1. Fact A" in (workspace / "FACTS.txt").read_text()


@pytest.mark.asyncio
async def test_agent_loop_verify_fix_reverify(tmp_path: Path) -> None:
    """A failed VERIFY re-enters the Act phase instead of terminating (M12).

    First attempt answers without creating the target file (verify fails on
    the missing-file check); the injected VERIFY-CORRECTION message is
    answered by actually creating the file, and the second verify passes.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "Create FACTS.txt file", '
            '"steps": ["Write 3 facts"], '
            '"target_files": ["FACTS.txt"], '
            '"acceptance_criteria": ["FACTS.txt exists"]}'
        )
    )
    # Attempt 1: model answers immediately without creating the file.
    premature_final = CompletionChunk(
        delta_text="Done.",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    # Attempt 2 (after VERIFY-CORRECTION): model actually creates the file.
    fix_tool_call = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="call_write_1",
                name="write_file",
                arguments={"path": "FACTS.txt", "content": "1. Fact A"},
            )
        ],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )
    fixed_final = CompletionChunk(
        delta_text="Fixed: FACTS.txt now exists.",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    fake_provider = FakeSequenceProvider(
        [spec_chunk, premature_final, fix_tool_call, fixed_final]
    )
    bus = EventBus()
    events_log: list[BaseEvent] = []

    async def track_events(ev: BaseEvent) -> None:
        events_log.append(ev)

    bus.subscribe("*", track_events)

    agent = AgentLoop(
        provider=fake_provider,
        tools=registry,
        event_bus=bus,
        max_steps=10,
        workspace_root=workspace,
    )

    result = await agent.run_result("Create FACTS.txt file")
    assert result.status == "success"
    assert "Fixed: FACTS.txt now exists." in result.final_text
    assert (workspace / "FACTS.txt").exists()
    assert len(events_log) >= 4


@pytest.mark.asyncio
async def test_agent_loop_infinite_loop_prevention(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    # Endless but VARYING tool calls — distinct paths each step so loop detection
    # does not fire and the max_steps guard is what stops the loop.
    varying_chunks = [
        CompletionChunk(
            tool_calls=[
                ToolCall(
                    id=f"call_{i}",
                    name="read_file",
                    arguments={"path": f"non_existent_{i}.txt"},
                )
            ]
        )
        for i in range(10)
    ]

    fake_provider = FakeSequenceProvider(varying_chunks)
    bus = EventBus()
    events_log: list[BaseEvent] = []

    async def track_events(ev: BaseEvent) -> None:
        events_log.append(ev)

    bus.subscribe("*", track_events)

    agent = AgentLoop(
        provider=fake_provider,
        tools=registry,
        event_bus=bus,
        max_steps=3,
    )

    await agent.run("Do something infinite")

    # Verify MaxStepsExceeded error event was emitted. Each failed tool call
    # (read_file on a missing path) also emits its own ErrorEvent, so filter for
    # the max-steps guard specifically rather than counting all error events.
    max_steps_events = [
        e for e in events_log if isinstance(e, ErrorEvent) and e.error_type == "MaxStepsExceeded"
    ]
    assert len(max_steps_events) == 1
    assert "maximum step count" in max_steps_events[0].message


@pytest.mark.asyncio
async def test_run_result_cancelled_returns_cancelled_status(tmp_path: Path) -> None:
    """Regression (M10 D2): cancelling a run returns RunResult(status='cancelled')
    instead of propagating a raw CancelledError."""
    import asyncio

    from anyio import sleep

    class BlockingProvider(LLMProvider):
        async def generate(self, request: CompletionRequest) -> CompletionChunk:
            await sleep(3600)  # block until the task is cancelled
            raise AssertionError("generate should have been cancelled")

        async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
            yield await self.generate(request)

        async def health_check(self) -> bool:
            return True

    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)
    agent = AgentLoop(provider=BlockingProvider(), tools=registry)

    task = asyncio.create_task(agent.run_result("do something"))
    await sleep(0.05)
    task.cancel()
    result = await task

    assert result.status == "cancelled"
    assert not result.success
    assert "cancelled" in (result.error or "")


@pytest.mark.asyncio
async def test_streaming_tool_call_arguments_merged_across_chunks() -> None:
    """Regression (M10 D4): streaming tool-call argument fragments split across
    3+ chunks (mid-JSON-token) are merged into one complete call before the tool
    executes, instead of each partial fragment being dispatched separately."""
    from nullain.tools.decorator import tool

    received: dict[str, Any] = {}

    @tool(name="echo_args", description="echo args", read_only=True)
    async def echo_args(path: str, limit: int) -> str:
        received["path"] = path
        received["limit"] = limit
        return f"path={path} limit={limit}"

    class FragmentedStreamProvider(LLMProvider):
        def __init__(self) -> None:
            self.step = 0

        async def generate(self, request: CompletionRequest) -> CompletionChunk:
            # Plan phase (MEDIUM intent) asks for a spec; return a minimal one.
            return CompletionChunk(
                delta_text=(
                    '{"objective": "do it", "steps": ["do it"], '
                    '"target_files": [], "acceptance_criteria": ["done"]}'
                )
            )

        async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
            if self.step == 0:
                self.step += 1
                # Fragment the arguments JSON across 3 chunks, splitting mid-token.
                for frag in ('{"path": "a.tx', 't", "limi', 't": 3}'):
                    yield CompletionChunk(
                        tool_calls=[ToolCall(id="call_1", name="echo_args", arguments=frag)]
                    )
            else:
                yield CompletionChunk(delta_text="done", finish_reason="stop")

        async def health_check(self) -> bool:
            return True

    registry = ToolRegistry()
    registry.register(echo_args)
    agent = AgentLoop(provider=FragmentedStreamProvider(), tools=registry)

    result = await agent.run_streaming("do it")
    # The tool executed once with the merged, parsed arguments (the regression:
    # without the merge, each fragment would be dispatched as a separate call).
    assert received == {"path": "a.txt", "limit": 3}
    assert result == "done"


@pytest.mark.asyncio
async def test_agent_loop_detection_repeated_calls(tmp_path: Path) -> None:
    """Repeating the exact same tool call for ``loop_detection_threshold``
    consecutive steps is detected as a stuck loop: the run returns a
    ``loop_detected`` RunResult and emits a LoopDetected error event."""
    from nullain.agent import RunResult

    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    # Identical tool call every step
    stuck_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="stuck_1",
                name="read_file",
                arguments={"path": "missing.txt"},
            )
        ]
    )
    fake_provider = FakeSequenceProvider([stuck_chunk] * 10)
    bus = EventBus()
    events_log: list[BaseEvent] = []

    async def track_events(ev: BaseEvent) -> None:
        events_log.append(ev)

    bus.subscribe("*", track_events)

    agent = AgentLoop(
        provider=fake_provider,
        tools=registry,
        event_bus=bus,
        max_steps=25,
        loop_detection_threshold=3,
    )

    result = await agent.run_result("Do something stuck")
    assert isinstance(result, RunResult)
    assert result.status == "loop_detected"
    assert not result.success

    loop_errors = [
        e for e in events_log if isinstance(e, ErrorEvent) and e.error_type == "LoopDetected"
    ]
    assert len(loop_errors) == 1
    # Self-correction reflection injected
    user_msgs = [e for e in events_log if e.event_type == "user_message"]
    assert any("[SELF-CORRECTION]" in getattr(m, "content", "") for m in user_msgs)


@pytest.mark.asyncio
async def test_agent_loop_spawn_subagent_returns_text(tmp_path: Path) -> None:
    """spawn runs a child AgentLoop with fresh context and returns only its
    final text. The parent's event bus is NOT polluted by the child's
    internal tool/model events."""
    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    # Child provider: a spec chunk (LOW complexity path skipped by using
    # explicit model + a trivial prompt) then a final answer.
    spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "Subtask", "steps": ["reply"], '
            '"target_files": [], "acceptance_criteria": []}'
        )
    )
    final_chunk = CompletionChunk(delta_text="Sub-agent result: 42")

    # We need two distinct providers: parent and child. Parent just spawns
    # and returns the child's text — so the parent's own run is bypassed by
    # calling spawn directly.
    child_provider = FakeSequenceProvider([spec_chunk, final_chunk])

    parent = AgentLoop(
        provider=child_provider,  # shared; spawn reuses it for the child
        tools=registry,
        event_bus=EventBus(),
        model="sub-model",
    )

    parent_events: list[BaseEvent] = []

    async def track_parent(ev: BaseEvent) -> None:
        parent_events.append(ev)

    parent.event_bus.subscribe("*", track_parent)

    text = await parent.spawn("Do the subtask", model="sub-model")
    assert text == "Sub-agent result: 42"
    # Parent bus stays clean: child's internal events were isolated.
    assert parent_events == []


@pytest.mark.asyncio
async def test_agent_loop_token_budget_exceeded(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    high_usage_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="call_1",
                name="read_file",
                arguments={"path": "test.txt"},
            )
        ],
        usage=TokenUsage(prompt_tokens=500, completion_tokens=600, total_tokens=1100),
    )

    fake_provider = FakeSequenceProvider([high_usage_chunk] * 5)
    agent = AgentLoop(
        provider=fake_provider,
        tools=registry,
        max_steps=10,
        max_tokens=1000,
    )

    with pytest.raises(BudgetExceededError):
        await agent.run("Run budget test")


@pytest.mark.asyncio
async def test_agent_loop_streaming(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "Stream test", '
            '"steps": ["Stream reply"], '
            '"target_files": [], '
            '"acceptance_criteria": []}'
        )
    )
    stream_chunk1 = CompletionChunk(delta_text="Hello ")
    stream_chunk2 = CompletionChunk(delta_text="world!")

    fake_provider = FakeSequenceProvider([spec_chunk, stream_chunk1, stream_chunk2])
    bus = EventBus()
    events_log: list[BaseEvent] = []

    async def track_events(ev: BaseEvent) -> None:
        events_log.append(ev)

    bus.subscribe("*", track_events)

    agent = AgentLoop(
        provider=fake_provider,
        tools=registry,
        event_bus=bus,
        max_steps=5,
    )

    result = await agent.run_streaming("Stream test")
    assert "Hello world!" in result or "Hello" in result
    stream_deltas = [e for e in events_log if e.event_type == "stream_delta"]
    assert len(stream_deltas) >= 1


@pytest.mark.asyncio
async def test_agent_loop_clock_timeout(tmp_path: Path) -> None:
    from nullain.errors import NullainError
    from nullain.ports.clock import Clock

    class AdvancingClock(Clock):
        def __init__(self) -> None:
            self._time = 1000.0

        def now(self) -> float:
            self._time += 500.0  # advance by 500s on every call
            return self._time

    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    fake_provider = FakeSequenceProvider([CompletionChunk(delta_text="Thinking")])
    agent = AgentLoop(
        provider=fake_provider,
        tools=registry,
        clock=AdvancingClock(),
        timeout=10.0,
    )

    with pytest.raises(NullainError, match="timed out"):
        await agent.run("Timeout test")


@pytest.mark.asyncio
async def test_agent_loop_self_correction(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "Test recovery", '
            '"steps": ["Try read", "Recover"], '
            '"target_files": [], '
            '"acceptance_criteria": []}'
        )
    )

    # Tool call that fails (reading non-existent file)
    fail_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="call_fail",
                name="read_file",
                arguments={"path": "does_not_exist_123.txt"},
            )
        ]
    )
    recover_chunk = CompletionChunk(delta_text="Recovered from missing file")

    fake_provider = FakeSequenceProvider([spec_chunk, fail_chunk, recover_chunk])
    bus = EventBus()
    events_log: list[BaseEvent] = []

    async def track_events(ev: BaseEvent) -> None:
        events_log.append(ev)

    bus.subscribe("*", track_events)

    agent = AgentLoop(
        provider=fake_provider,
        tools=registry,
        event_bus=bus,
        model="gpt-4o",  # explicit model to bypass spec generation
        self_correction_max=2,
    )

    result = await agent.run("Test recovery")
    assert result == "Recovered from missing file"
    # Check that self-correction prompt was injected as a UserMessageEvent
    user_msgs = [e for e in events_log if e.event_type == "user_message"]
    assert any("[SELF-CORRECTION]" in getattr(m, "content", "") for m in user_msgs)


@pytest.mark.asyncio
async def test_agent_loop_context_window_exhausted_thrash(tmp_path: Path) -> None:
    """Thrashing protection: when compaction cannot bring the window under
    threshold for more than max_compaction_attempts consecutive steps, the loop
    raises ContextWindowExhaustedError instead of looping forever."""
    from nullain.context import ContextManager
    from nullain.errors import ContextWindowExhaustedError

    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    # Provider that always requests a (no-op) tool call so the loop keeps going.
    tool_chunk = CompletionChunk(
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "x.txt"})]
    )
    fake_provider = FakeSequenceProvider([tool_chunk] * 20)

    # Tiny window + low threshold -> compaction required every step, and the
    # system prompt alone keeps the window over threshold after compacting.
    cm = ContextManager(max_window_tokens=100, compaction_threshold=0.5)

    agent = AgentLoop(
        provider=fake_provider,
        tools=registry,
        context_manager=cm,
        max_steps=15,
        max_compaction_attempts=3,
        # Disable loop detection here: this test exercises compaction thrash
        # specifically, which requires the same call to repeat past the point
        # loop detection would otherwise stop it.
        loop_detection_threshold=100,
    )

    with pytest.raises(ContextWindowExhaustedError):
        await agent.run("Thrash test")


@pytest.mark.asyncio
async def test_agent_loop_prefers_real_usage_over_estimate(tmp_path: Path) -> None:
    """Regression (M10 D5): the compaction decision prefers the provider's real
    ``usage`` token count over the ``len//4`` estimate. A long tool-call argument
    inflates the estimate past the threshold, but the small real usage keeps the
    context under it, so no compaction is triggered."""
    from nullain.context import ContextManager
    from nullain.events import CompactionEvent

    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "Real usage test", "steps": ["do it"], '
            '"target_files": [], "acceptance_criteria": []}'
        )
    )
    # Long argument inflates the len//4 estimate; real usage stays small.
    # Threshold is 2000 (max_window_tokens=4000 * 0.5): above the ~1246-token
    # system-prompt estimate (so step 0 does not compact) but far below the
    # ~5000-token estimate of the long tool-call argument.
    long_path = "x" * 20000
    tool_chunk = CompletionChunk(
        tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": long_path})],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    final_chunk = CompletionChunk(
        delta_text="done",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    fake_provider = FakeSequenceProvider([spec_chunk, tool_chunk, final_chunk])
    bus = EventBus()
    events_log: list[BaseEvent] = []

    async def track_events(ev: BaseEvent) -> None:
        events_log.append(ev)

    bus.subscribe("*", track_events)

    agent = AgentLoop(
        provider=fake_provider,
        tools=registry,
        event_bus=bus,
        max_steps=5,
        context_manager=ContextManager(max_window_tokens=4000, compaction_threshold=0.5),
    )

    await agent.run("Real usage test")

    # The len//4 estimate of the long tool-call argument (~5000 tokens) would
    # exceed the 2000-token threshold, but the real usage (10 prompt tokens)
    # keeps the context under it — so the loop must NOT have compacted.
    compactions = [e for e in events_log if isinstance(e, CompactionEvent)]
    assert compactions == []
