"""Unit tests for Agent session persistence (M16)."""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from nullain.agent import Agent
from nullain.config import NullainSettings
from nullain.events import EventStore
from nullain.llm import ChatMessage, CompletionChunk, CompletionRequest, LLMProvider


class FakeSequenceProvider(LLMProvider):
    """Scripted provider yielding one response per call, then a default."""

    def __init__(self, responses: list[CompletionChunk]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.seen_messages: list[list[ChatMessage]] = []

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        self.seen_messages.append(list(request.messages))
        if self.call_count < len(self.responses):
            chunk = self.responses[self.call_count]
            self.call_count += 1
            return chunk
        return CompletionChunk(delta_text="Default finished")

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        yield await self.generate(request)

    async def health_check(self) -> bool:
        return True


def _agent(tmp_path: Path, provider: LLMProvider, event_store: EventStore | None = None) -> Agent:
    return Agent(
        settings=NullainSettings(),
        provider=provider,
        workspace_root=tmp_path,
        model="m",
        event_store=event_store,
    )


@pytest.mark.asyncio
async def test_agent_defaults_to_persistent_sqlite_event_store(tmp_path: Path) -> None:
    """Agent() with no event_store override creates <workspace>/.nullain/sessions.db,
    not an in-memory store — so a session survives the process exiting."""
    provider = FakeSequenceProvider([CompletionChunk(delta_text="hi")])
    agent = _agent(tmp_path, provider)

    await agent.run("Say hi", session_id="sess-1")

    db_path = tmp_path / ".nullain" / "sessions.db"
    assert db_path.exists()


@pytest.mark.asyncio
async def test_agent_run_resumes_session_history(tmp_path: Path) -> None:
    """A second Agent.run() with the same session_id sees the first turn's
    events in its message history — the model's second request must include
    the first turn's user message and answer, not start from scratch.

    Prompts use a "format" keyword so the deterministic intent heuristic
    classifies them LOW complexity — skipping the Plan phase's own provider
    call, which would otherwise consume the scripted final-answer response
    meant for the Act phase.
    """
    provider = FakeSequenceProvider(
        [
            CompletionChunk(delta_text="First answer."),
            CompletionChunk(delta_text="Second answer."),
        ]
    )
    agent = _agent(tmp_path, provider)

    result1 = await agent.run("format this question", session_id="sess-resume")
    assert result1.final_text == "First answer."

    result2 = await agent.run("format that question", session_id="sess-resume")
    assert result2.final_text == "Second answer."

    # The second call's request must include the first turn's exchange.
    second_call_messages = provider.seen_messages[-1]
    contents = [m.content for m in second_call_messages]
    assert any(c and "format this question" in c for c in contents)
    assert any(c and "First answer." in c for c in contents)
    assert any(c and "format that question" in c for c in contents)


@pytest.mark.asyncio
async def test_agent_run_fresh_session_id_starts_empty(tmp_path: Path) -> None:
    """A session_id with no prior events (including a brand new one) starts
    with no history — resuming only kicks in when there is something to
    resume."""
    provider = FakeSequenceProvider([CompletionChunk(delta_text="hi")])
    agent = _agent(tmp_path, provider)

    await agent.run("format this", session_id="brand-new-session")

    first_call_messages = provider.seen_messages[0]
    # Only the system prompt + this turn's user message — no prior turns.
    user_messages = [m for m in first_call_messages if m.role == "user"]
    assert len(user_messages) == 1
    assert user_messages[0].content == "format this"


@pytest.mark.asyncio
async def test_agent_run_none_session_id_does_not_resume(tmp_path: Path) -> None:
    """session_id=None (a fresh id is generated internally) must not attempt
    to load history — there is nothing to key a lookup on before the loop
    generates the id itself."""
    provider = FakeSequenceProvider(
        [CompletionChunk(delta_text="First."), CompletionChunk(delta_text="Second.")]
    )
    agent = _agent(tmp_path, provider)

    await agent.run("format first")
    await agent.run("format second")  # no session_id passed either time

    second_call_messages = provider.seen_messages[-1]
    user_messages = [m for m in second_call_messages if m.role == "user"]
    assert len(user_messages) == 1
    assert user_messages[0].content == "format second"


@pytest.mark.asyncio
async def test_agent_stream_also_resumes_session_history(tmp_path: Path) -> None:
    """Agent.stream() resumes history the same way Agent.run() does."""
    from nullain.agent.result import RunResult

    provider = FakeSequenceProvider(
        [CompletionChunk(delta_text="First."), CompletionChunk(delta_text="Second.")]
    )
    agent = _agent(tmp_path, provider)

    async for _item in agent.stream("format first", session_id="sess-stream"):
        pass

    async for item in agent.stream("format second", session_id="sess-stream"):
        if isinstance(item, RunResult):
            assert item.final_text == "Second."

    second_call_messages = provider.seen_messages[-1]
    contents = [m.content for m in second_call_messages]
    assert any(c and "format first" in c for c in contents)


@pytest.mark.asyncio
async def test_explicit_memory_event_store_disables_persistence(tmp_path: Path) -> None:
    """Passing EventStore(':memory:') explicitly opts out of the on-disk
    default — no sessions.db file should be created."""
    provider = FakeSequenceProvider([CompletionChunk(delta_text="hi")])
    agent = _agent(tmp_path, provider, event_store=EventStore(":memory:"))

    await agent.run("Say hi", session_id="sess-mem")

    db_path = tmp_path / ".nullain" / "sessions.db"
    assert not db_path.exists()
