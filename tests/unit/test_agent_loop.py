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

    # Endless tool calls response
    endless_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="endless_1",
                name="read_file",
                arguments={"path": "non_existent.txt"},
            )
        ]
    )

    # Infinite sequence provider
    fake_provider = FakeSequenceProvider([endless_chunk] * 10)
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
