"""Unit tests for conversation checkpoint/rewind (M14)."""

from pathlib import Path

import pytest
from nullain.agent import AgentLoop, list_rewind_points, rewind_events
from nullain.events import (
    BaseEvent,
    ModelResponseEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from nullain.llm import CompletionChunk, CompletionRequest, LLMProvider, ToolCall
from nullain.tools import ToolRegistry
from nullain_tools import register_default_tools


def _events_for_two_step_trajectory() -> list[BaseEvent]:
    """A user message, a tool-call step, its result, and a final-answer step."""
    sess = "sess_1"
    user = UserMessageEvent(session_id=sess, content="Do the task")
    step1_call = ModelResponseEvent(
        session_id=sess,
        model="m",
        tool_calls=(ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"}),),
    )
    tc_ev = ToolCallEvent(session_id=sess, call_id="c1", tool_name="read_file", arguments={})
    tr_ev = ToolResultEvent(
        session_id=sess, call_id="c1", tool_name="read_file", output="ok", is_error=False
    )
    step2_final = ModelResponseEvent(session_id=sess, model="m", content="Done.")
    return [user, step1_call, tc_ev, tr_ev, step2_final]


def test_list_rewind_points_enumerates_one_per_model_response() -> None:
    events = _events_for_two_step_trajectory()
    points = list_rewind_points(events)

    assert [p.step for p in points] == [1, 2]
    assert points[0].had_tool_calls is True
    assert "read_file" in points[0].summary
    assert points[1].had_tool_calls is False
    assert points[1].summary == "Done."


def test_list_rewind_points_empty_trajectory() -> None:
    assert list_rewind_points([]) == []


def test_rewind_events_to_step_1_keeps_only_prefix_before_it() -> None:
    events = _events_for_two_step_trajectory()
    truncated = rewind_events(events, to_step=1)

    # Everything up to (not including) step 1's ModelResponseEvent survives —
    # here, just the initial UserMessageEvent.
    assert len(truncated) == 1
    assert isinstance(truncated[0], UserMessageEvent)


def test_rewind_events_to_step_2_keeps_step_1_and_its_result() -> None:
    events = _events_for_two_step_trajectory()
    truncated = rewind_events(events, to_step=2)

    # Everything before step 2's ModelResponseEvent survives: user message,
    # step 1's tool-call response, and its tool call/result events.
    assert len(truncated) == 4
    assert isinstance(truncated[-1], ToolResultEvent)


def test_rewind_events_rejects_step_below_one() -> None:
    events = _events_for_two_step_trajectory()
    with pytest.raises(ValueError, match="to_step must be >= 1"):
        rewind_events(events, to_step=0)


def test_rewind_events_rejects_step_beyond_trajectory() -> None:
    events = _events_for_two_step_trajectory()
    with pytest.raises(ValueError, match="exceeds the trajectory's 2 completed step"):
        rewind_events(events, to_step=3)


def test_rewind_events_empty_trajectory_rejects_any_step() -> None:
    with pytest.raises(ValueError, match="exceeds the trajectory's 0 completed step"):
        rewind_events([], to_step=1)


class FakeSequenceProvider(LLMProvider):
    """Fake provider yielding a scripted sequence of responses (mirrors
    the fixture used across test_agent_loop.py)."""

    def __init__(self, responses: list[CompletionChunk]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        if self.call_count < len(self.responses):
            chunk = self.responses[self.call_count]
            self.call_count += 1
            return chunk
        return CompletionChunk(delta_text="Default finished")

    async def stream(self, request: CompletionRequest):
        yield await self.generate(request)

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_agent_loop_rewind_and_run_resumes_from_truncated_context(tmp_path: Path) -> None:
    """rewind_and_run truncates the trajectory then runs a fresh prompt from it.

    First run ends badly (wrong file created). We rewind to before that
    step's response and resume with a corrective prompt on a fresh
    AgentLoop/provider (a different attempt, as a real rewind-and-retry
    would use). The resumed run must produce the right file without seeing
    the original bad tool call in its own trajectory.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ToolRegistry()
    register_default_tools(registry, workspace)

    spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "Create right.txt", "steps": ["write"], '
            '"target_files": [], "acceptance_criteria": []}'
        )
    )
    bad_write = CompletionChunk(
        tool_calls=[
            ToolCall(id="w1", name="write_file", arguments={"path": "wrong.txt", "content": "x"})
        ]
    )
    bad_final = CompletionChunk(delta_text="Wrote wrong.txt.")

    original_provider = FakeSequenceProvider([spec_chunk, bad_write, bad_final])
    original_agent = AgentLoop(
        provider=original_provider, tools=registry, model="m", workspace_root=workspace
    )

    captured: list[BaseEvent] = []

    async def _capture(ev: BaseEvent) -> None:
        captured.append(ev)

    original_agent.event_bus.subscribe("*", _capture)

    original_result = await original_agent.run_result("Create right.txt", session_id="sess_x")
    assert original_result.status == "success"
    assert (workspace / "wrong.txt").exists()

    points = original_agent.list_rewind_points(captured)
    assert len(points) == 2  # step 1: bad write's tool-call response; step 2: final answer
    assert points[0].had_tool_calls is True

    # Rewind to before step 1's response and retry with a corrective prompt.
    # The rewound run re-enters the pipeline from its Plan phase (a fresh
    # prompt is a fresh classification), so it needs its own spec response.
    retry_spec_chunk = CompletionChunk(
        delta_text=(
            '{"objective": "Create right.txt", "steps": ["write"], '
            '"target_files": [], "acceptance_criteria": []}'
        )
    )
    good_write = CompletionChunk(
        tool_calls=[
            ToolCall(id="w2", name="write_file", arguments={"path": "right.txt", "content": "y"})
        ]
    )
    good_final = CompletionChunk(delta_text="Wrote right.txt.")
    retry_provider = FakeSequenceProvider([retry_spec_chunk, good_write, good_final])
    retry_agent = AgentLoop(
        provider=retry_provider, tools=registry, model="m", workspace_root=workspace
    )

    resumed = await retry_agent.rewind_and_run(
        events_history=captured,
        to_step=1,
        prompt="Actually, create right.txt instead.",
        session_id="sess_x",
    )

    assert resumed.status == "success"
    assert (workspace / "right.txt").exists()
    assert "Wrote right.txt." in resumed.final_text


def test_agent_loop_exposes_list_rewind_points_as_instance_method() -> None:
    """AgentLoop.list_rewind_points is a thin pass-through — no network/tools needed."""
    registry = ToolRegistry()

    class _NoopProvider(LLMProvider):
        async def generate(self, request: CompletionRequest) -> CompletionChunk:
            return CompletionChunk(delta_text="unused")

        async def stream(self, request: CompletionRequest):
            yield CompletionChunk(delta_text="unused")

        async def health_check(self) -> bool:
            return True

    agent = AgentLoop(provider=_NoopProvider(), tools=registry)
    events = _events_for_two_step_trajectory()
    assert agent.list_rewind_points(events) == list_rewind_points(events)
