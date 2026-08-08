"""Unit tests for Agent.close() — releasing SQLite connections after run/stream.

Regression coverage for a real bug found integrating the SDK into
nullain-agent's AgentBridge (docs/FUSION_PLAN.md): EventStore and
EpisodicMemory each open a lazy aiosqlite connection backed by a worker
thread that stays alive until closed explicitly. Nothing in run()/stream()
closed them, so a script that ran the agent and exited without calling
close() would hang indefinitely at interpreter shutdown — no error, no
timeout, just two live aiosqlite worker threads blocking the process from
exiting. Never surfaced before because every existing test either used
":memory:" stores across a single pytest session (no process exit in
between) or never exercised a real file-backed store end to end.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from nullain.agent import Agent
from nullain.config import NullainSettings
from nullain.llm import CompletionChunk, CompletionRequest, LLMProvider


class FakeProvider(LLMProvider):
    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        return CompletionChunk(delta_text="Default finished")

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        yield await self.generate(request)

    async def health_check(self) -> bool:
        return True


def _agent(tmp_path: Path) -> Agent:
    return Agent(
        settings=NullainSettings(),
        provider=FakeProvider(),
        workspace_root=tmp_path,
        model="m",
    )


@pytest.mark.asyncio
async def test_close_releases_connections_after_run(tmp_path: Path) -> None:
    """After run(), both stores hold an open aiosqlite connection; close()
    must release both so no worker thread is left behind."""
    agent = _agent(tmp_path)
    await agent.run("Say hi", session_id="sess-close-1")

    assert agent._event_store._conn is not None  # type: ignore[reportPrivateUsage]
    assert agent._episodic_memory._db is not None  # type: ignore[reportPrivateUsage]

    await agent.close()

    assert agent._event_store._conn is None  # type: ignore[reportPrivateUsage]
    assert agent._episodic_memory._db is None  # type: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_close_is_safe_before_any_run(tmp_path: Path) -> None:
    """close() on a freshly constructed Agent (neither store ever
    initialized) must not raise — both underlying close() calls are
    no-ops on an unopened connection."""
    agent = _agent(tmp_path)
    await agent.close()  # must not raise


@pytest.mark.asyncio
async def test_close_is_idempotent(tmp_path: Path) -> None:
    """Calling close() twice in a row must not raise."""
    agent = _agent(tmp_path)
    await agent.run("Say hi", session_id="sess-close-2")
    await agent.close()
    await agent.close()  # must not raise


@pytest.mark.asyncio
async def test_close_releases_connections_after_stream(tmp_path: Path) -> None:
    """stream() opens the same two connections as run() — close() must
    release them on that path too."""
    agent = _agent(tmp_path)
    async for _item in agent.stream("Say hi", session_id="sess-close-3"):
        pass

    assert agent._event_store._conn is not None  # type: ignore[reportPrivateUsage]
    await agent.close()
    assert agent._event_store._conn is None  # type: ignore[reportPrivateUsage]
