"""Unit and E2E tests for AgentLoop ReAct Core."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from nullain.agent import AgentLoop, SpawnTask
from nullain.authority import Authority
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
async def test_agent_loop_spec_generation_uses_structured_tool_call(tmp_path: Path) -> None:
    """Plan phase parses TaskSpec from a tool call, not free-text JSON (M12).

    The spec-generation request now offers an ``emit_task_spec`` tool; when
    the model responds with a tool call (rather than delta_text), the spec
    must be built from the call's arguments.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    spec_via_tool_call = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="call_spec_1",
                name="emit_task_spec",
                arguments={
                    "objective": "Create FACTS.txt file",
                    "steps": ["Write 3 facts"],
                    "target_files": ["FACTS.txt"],
                    "acceptance_criteria": ["FACTS.txt exists"],
                },
            )
        ]
    )
    write_call = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="call_write_1",
                name="write_file",
                arguments={"path": "FACTS.txt", "content": "1. Fact A"},
            )
        ],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )
    final_answer = CompletionChunk(
        delta_text="Finished creating FACTS.txt.",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    fake_provider = FakeSequenceProvider([spec_via_tool_call, write_call, final_answer])
    agent = AgentLoop(
        provider=fake_provider,
        tools=registry,
        max_steps=5,
        workspace_root=workspace,
    )

    result = await agent.run_result("Create FACTS.txt file")
    assert result.status == "success"
    assert (workspace / "FACTS.txt").exists()


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

    fake_provider = FakeSequenceProvider([spec_chunk, premature_final, fix_tool_call, fixed_final])
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
async def test_agent_loop_verify_retry_gets_fresh_correction_budget(tmp_path: Path) -> None:
    """Each verify-fix-reverify cycle gets its own self-correction budget (M14).

    self_correction_max=1: cycle 1 spends its one self-correction allowance
    recovering from a failed read_file, then answers without creating the
    target file (verify fails). Cycle 2 also hits a failed read_file — if
    correction_budget were not reset per cycle, this second error would get
    no [SELF-CORRECTION] reflection and the model would have no signal to
    retry. Both cycles' errors must be followed by a reflection, proving the
    budget was refreshed for cycle 2.
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
            '"acceptance_criteria": []}'
        )
    )
    # Cycle 1: a failing read (spends the sole self-correction allowance),
    # then a premature final answer without creating the file (verify fails).
    cycle1_fail_read = CompletionChunk(
        tool_calls=[
            ToolCall(id="r1", name="read_file", arguments={"path": "missing_1.txt"}),
        ]
    )
    cycle1_premature_final = CompletionChunk(
        delta_text="Done (nothing created).",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    # Cycle 2 (after VERIFY-CORRECTION): another failing read. If the budget
    # were not refreshed, correction_budget would already be 0 here and no
    # [SELF-CORRECTION] reflection would be injected for this error.
    cycle2_fail_read = CompletionChunk(
        tool_calls=[
            ToolCall(id="r2", name="read_file", arguments={"path": "missing_2.txt"}),
        ]
    )
    cycle2_fix_write = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="w1", name="write_file", arguments={"path": "FACTS.txt", "content": "1. Fact A"}
            )
        ],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )
    cycle2_final = CompletionChunk(
        delta_text="Fixed: FACTS.txt now exists.",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    fake_provider = FakeSequenceProvider(
        [
            spec_chunk,
            cycle1_fail_read,
            cycle1_premature_final,
            cycle2_fail_read,
            cycle2_fix_write,
            cycle2_final,
        ]
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
        max_steps=15,
        workspace_root=workspace,
        self_correction_max=1,
        verify_retry_max=2,
    )

    result = await agent.run_result("Create FACTS.txt file")
    assert result.status == "success"
    assert (workspace / "FACTS.txt").exists()

    user_msgs = [e for e in events_log if e.event_type == "user_message"]
    self_correction_msgs = [
        m for m in user_msgs if "[SELF-CORRECTION]" in getattr(m, "content", "")
    ]
    # One reflection per cycle's tool failure — cycle 2's would be missing if
    # correction_budget leaked from cycle 1 instead of being refreshed.
    assert len(self_correction_msgs) == 2


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


class KeyedFakeProvider(LLMProvider):
    """Fake provider that scripts a distinct response sequence per user prompt.

    Each concurrent sub-agent in a spawn_many fan-out sends a different
    first user message, so responses are looked up by which script's opening
    instruction text appears in the request — letting several concurrent
    child loops share one provider instance while each gets its own scripted
    answer.
    """

    def __init__(self, scripts: dict[str, list[CompletionChunk]]) -> None:
        self.scripts = scripts
        self.call_counts: dict[str, int] = dict.fromkeys(scripts, 0)

    def _match_key(self, request: CompletionRequest) -> str | None:
        for msg in request.messages:
            if msg.role != "user" or not msg.content:
                continue
            for key in self.scripts:
                if key in msg.content:
                    return key
        return None

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        key = self._match_key(request)
        if key is None:
            return CompletionChunk(delta_text="Default finished")
        idx = self.call_counts[key]
        responses = self.scripts[key]
        if idx < len(responses):
            self.call_counts[key] += 1
            return responses[idx]
        return CompletionChunk(delta_text="Default finished")

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        yield await self.generate(request)

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_agent_loop_spawn_many_runs_concurrently(tmp_path: Path) -> None:
    """spawn_many fans out several sub-agents and returns outcomes in order (M13)."""
    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    scripts = {
        "task-A": [
            CompletionChunk(
                delta_text='{"objective": "A", "steps": ["reply"], '
                '"target_files": [], "acceptance_criteria": []}'
            ),
            CompletionChunk(delta_text="Result A"),
        ],
        "task-B": [
            CompletionChunk(
                delta_text='{"objective": "B", "steps": ["reply"], '
                '"target_files": [], "acceptance_criteria": []}'
            ),
            CompletionChunk(delta_text="Result B"),
        ],
        "task-C": [
            CompletionChunk(
                delta_text='{"objective": "C", "steps": ["reply"], '
                '"target_files": [], "acceptance_criteria": []}'
            ),
            CompletionChunk(delta_text="Result C"),
        ],
    }
    provider = KeyedFakeProvider(scripts)

    parent = AgentLoop(provider=provider, tools=registry, model="sub-model")

    outcomes = await parent.spawn_many(
        [
            SpawnTask(prompt="task-A do something", model="sub-model"),
            SpawnTask(prompt="task-B do something", model="sub-model"),
            SpawnTask(prompt="task-C do something", model="sub-model"),
        ]
    )

    assert [o.text for o in outcomes] == ["Result A", "Result B", "Result C"]
    assert all(o.success for o in outcomes)


@pytest.mark.asyncio
async def test_agent_loop_spawn_many_isolates_task_failure(tmp_path: Path) -> None:
    """One task lacking SPAWN capability fails without affecting siblings (M13)."""
    registry = ToolRegistry()
    register_default_tools(registry, tmp_path)

    scripts = {
        "task-ok": [
            CompletionChunk(
                delta_text='{"objective": "ok", "steps": ["reply"], '
                '"target_files": [], "acceptance_criteria": []}'
            ),
            CompletionChunk(delta_text="Result OK"),
        ],
    }
    provider = KeyedFakeProvider(scripts)

    # Parent authority lacks SPAWN, so any delegated-authority task fails at
    # _child_registry's SPAWN-capability gate; the other task passes
    # authority=None and is unaffected.
    parent = AgentLoop(
        provider=provider,
        tools=registry,
        model="sub-model",
        authority=Authority(
            capabilities=frozenset(),
            allowed_tools=frozenset(),
            deny_patterns=frozenset(),
            can_spawn=False,
        ),
    )

    outcomes = await parent.spawn_many(
        [
            SpawnTask(prompt="task-ok do something", model="sub-model"),
            SpawnTask(
                prompt="task-blocked do something",
                model="sub-model",
                authority=Authority(
                    capabilities=frozenset(),
                    allowed_tools=frozenset(),
                    deny_patterns=frozenset(),
                    can_spawn=False,
                ),
            ),
        ]
    )

    assert outcomes[0].success
    assert outcomes[0].text == "Result OK"
    assert not outcomes[1].success
    assert outcomes[1].error is not None
    assert "SPAWN capability" in outcomes[1].error


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


def test_batch_by_conflict_disjoint_writes_run_together() -> None:
    """Writes to different files have no conflicting resource — one batch (M13)."""
    registry = ToolRegistry()
    register_default_tools(registry, Path("."))

    calls = [
        ToolCall(id="a", name="write_file", arguments={"path": "a.txt", "content": "A"}),
        ToolCall(id="b", name="write_file", arguments={"path": "b.txt", "content": "B"}),
        ToolCall(id="c", name="write_file", arguments={"path": "c.txt", "content": "C"}),
    ]

    batches = AgentLoop._batch_by_conflict(calls, registry)  # type: ignore[reportPrivateUsage]
    assert len(batches) == 1
    assert {tc.id for tc in batches[0]} == {"a", "b", "c"}


def test_batch_by_conflict_same_file_write_write_serializes() -> None:
    """Two writes to the same file must land in different batches (M13)."""
    registry = ToolRegistry()
    register_default_tools(registry, Path("."))

    calls = [
        ToolCall(id="a", name="write_file", arguments={"path": "same.txt", "content": "A"}),
        ToolCall(id="b", name="write_file", arguments={"path": "same.txt", "content": "B"}),
    ]

    batches = AgentLoop._batch_by_conflict(calls, registry)  # type: ignore[reportPrivateUsage]
    assert [{tc.id for tc in b} for b in batches] == [{"a"}, {"b"}]


def test_batch_by_conflict_read_then_write_same_file_serializes() -> None:
    """A read and a write on the same file conflict — read_only alone is not enough (M13)."""
    registry = ToolRegistry()
    register_default_tools(registry, Path("."))

    calls = [
        ToolCall(id="r", name="read_file", arguments={"path": "same.txt"}),
        ToolCall(id="w", name="write_file", arguments={"path": "same.txt", "content": "X"}),
    ]

    batches = AgentLoop._batch_by_conflict(calls, registry)  # type: ignore[reportPrivateUsage]
    assert [{tc.id for tc in b} for b in batches] == [{"r"}, {"w"}]


def test_batch_by_conflict_all_read_only_disjoint_paths_run_together() -> None:
    """Multiple reads of different files run in one batch, as before M13."""
    registry = ToolRegistry()
    register_default_tools(registry, Path("."))

    calls = [
        ToolCall(id="r1", name="read_file", arguments={"path": "a.txt"}),
        ToolCall(id="r2", name="read_file", arguments={"path": "b.txt"}),
    ]

    batches = AgentLoop._batch_by_conflict(calls, registry)  # type: ignore[reportPrivateUsage]
    assert len(batches) == 1
    assert {tc.id for tc in batches[0]} == {"r1", "r2"}


def test_batch_by_conflict_bash_calls_serialize_with_each_other() -> None:
    """bash has no path argument — two bash calls fall back to tool-scoped
    serialization rather than being assumed independent (M13)."""
    registry = ToolRegistry()
    register_default_tools(registry, Path("."))

    calls = [
        ToolCall(id="b1", name="bash", arguments={"command_args": ["echo", "1"]}),
        ToolCall(id="b2", name="bash", arguments={"command_args": ["echo", "2"]}),
    ]

    batches = AgentLoop._batch_by_conflict(calls, registry)  # type: ignore[reportPrivateUsage]
    assert [{tc.id for tc in b} for b in batches] == [{"b1"}, {"b2"}]


@pytest.mark.asyncio
async def test_agent_loop_disjoint_writes_execute_concurrently(tmp_path: Path) -> None:
    """End-to-end: writes to different files in one step both land on disk (M13)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "Create two files", '
            '"steps": ["Write a.txt and b.txt"], '
            '"target_files": ["a.txt", "b.txt"], '
            '"acceptance_criteria": []}'
        )
    )
    dual_write_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(id="w1", name="write_file", arguments={"path": "a.txt", "content": "A"}),
            ToolCall(id="w2", name="write_file", arguments={"path": "b.txt", "content": "B"}),
        ],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )
    final_chunk = CompletionChunk(
        delta_text="Created both files.",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    fake_provider = FakeSequenceProvider([spec_chunk, dual_write_chunk, final_chunk])
    agent = AgentLoop(
        provider=fake_provider,
        tools=registry,
        max_steps=5,
        workspace_root=workspace,
    )

    result = await agent.run_result("Create two files")
    assert result.status == "success"
    assert (workspace / "a.txt").read_text() == "A"
    assert (workspace / "b.txt").read_text() == "B"


@pytest.mark.asyncio
async def test_agent_loop_incremental_verify_flags_broken_command_before_final_answer(
    tmp_path: Path,
) -> None:
    """A write to a target_file runs verification_commands right away (M14).

    A fake bash tool fails the first time (right after the target file is
    written) and succeeds the second time. The failure must be surfaced as
    an [INCREMENTAL-VERIFY] reflection mid-run — before the model's final
    answer — not only discovered by the end-of-run VERIFY phase.
    """
    from nullain.agent.spec import BASH_NONZERO_PREFIX
    from nullain.tools.decorator import tool as tool_decorator
    from nullain.tools.result import ToolResult

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    call_count = {"n": 0}

    @tool_decorator(name="bash", description="fake bash for incremental verify", read_only=False)
    async def fake_bash(command_args: list[str]) -> ToolResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ToolResult(
                output=f"{BASH_NONZERO_PREFIX} 1\nAssertionError: expected 3 facts, got 1",
                is_error=True,
                error_type="ToolError",
            )
        return ToolResult(output="3 passed", is_error=False)

    registry.register(fake_bash)

    spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "Create FACTS.txt file", '
            '"steps": ["Write 3 facts"], '
            '"target_files": ["FACTS.txt"], '
            '"acceptance_criteria": [], '
            '"verification_commands": ["pytest"]}'
        )
    )
    write_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(id="w1", name="write_file", arguments={"path": "FACTS.txt", "content": "1."}),
        ],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )
    fix_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="w2",
                name="write_file",
                arguments={"path": "FACTS.txt", "content": "1.\n2.\n3."},
            )
        ],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )
    final_chunk = CompletionChunk(
        delta_text="Fixed: FACTS.txt has 3 facts.",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    fake_provider = FakeSequenceProvider([spec_chunk, write_chunk, fix_chunk, final_chunk])
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
    # 2 incremental runs (fail after first write, pass after fix) + 1 more
    # from the end-of-run VERIFY phase re-running verification_commands.
    assert call_count["n"] == 3

    user_msgs = [e for e in events_log if e.event_type == "user_message"]
    incremental_msgs = [m for m in user_msgs if "[INCREMENTAL-VERIFY]" in getattr(m, "content", "")]
    assert len(incremental_msgs) == 1
    assert "expected 3 facts" in getattr(incremental_msgs[0], "content", "")

    # The incremental-verify reflection must appear before the model's final
    # answer, not after — it should influence the next step, not just be
    # logged post-hoc.
    def _is_final_answer(e: BaseEvent) -> bool:
        return e.event_type == "model_response" and "Fixed: FACTS.txt" in (
            getattr(e, "content", "") or ""
        )

    final_answer_idx = next(i for i, e in enumerate(events_log) if _is_final_answer(e))
    incremental_idx = events_log.index(incremental_msgs[0])
    assert incremental_idx < final_answer_idx
