"""Unit and E2E tests for AgentLoop ReAct Core."""

from collections.abc import AsyncGenerator
from pathlib import Path

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
    )

    result_text = await agent.run("Create FACTS.txt file")
    assert "Finished creating FACTS.txt" in result_text
    assert (workspace / "FACTS.txt").exists()
    assert "1. Fact A" in (workspace / "FACTS.txt").read_text()
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

    # Verify MaxStepsExceeded error event was emitted
    error_events = [e for e in events_log if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert "maximum step count" in error_events[0].message


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
