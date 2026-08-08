"""Unit tests for QuotaChecker — per-tenant quota enforcement (ADR-4).

Covers the Protocol contract itself, and AgentLoop's integration point:
quota_checker.check() is consulted alongside max_tokens before each step,
a denial raises QuotaExceededError and maps to RunResult.status ==
"quota_exceeded" (distinct from "budget", the token-ceiling status).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from nullain.agent import Agent
from nullain.config import NullainSettings
from nullain.llm import CompletionChunk, CompletionRequest, LLMProvider
from nullain.quota import QuotaChecker, QuotaExceededError


class _FakeProvider(LLMProvider):
    """Always returns a final answer on the first call — enough steps to
    exercise the pre-step quota check without needing tool calls."""

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        return CompletionChunk(delta_text="Final answer.", finish_reason="stop")

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        yield await self.generate(request)

    async def health_check(self) -> bool:
        return True


class _AlwaysDeniesChecker:
    """A QuotaChecker that denies every session unconditionally."""

    def __init__(self) -> None:
        self.checked_sessions: list[str] = []

    async def check(self, session_id: str) -> None:
        self.checked_sessions.append(session_id)
        raise QuotaExceededError(f"quota exhausted for {session_id}")


class _AlwaysAllowsChecker:
    """A QuotaChecker that allows every session — sanity-checks the
    integration doesn't break the happy path."""

    def __init__(self) -> None:
        self.checked_sessions: list[str] = []

    async def check(self, session_id: str) -> None:
        self.checked_sessions.append(session_id)


def _agent(tmp_path: Path, quota_checker: QuotaChecker | None) -> Agent:
    return Agent(
        settings=NullainSettings(),
        provider=_FakeProvider(),
        workspace_root=tmp_path,
        model="m",
        quota_checker=quota_checker,
    )


def test_always_denies_checker_satisfies_quota_checker_protocol() -> None:
    assert isinstance(_AlwaysDeniesChecker(), QuotaChecker)


@pytest.mark.asyncio
async def test_denied_quota_produces_quota_exceeded_status(tmp_path: Path) -> None:
    checker = _AlwaysDeniesChecker()
    agent = _agent(tmp_path, checker)

    result = await agent.run("Say hi", session_id="tenant-a")

    assert result.status == "quota_exceeded"
    assert result.success is False
    assert "tenant-a" in (result.error or "")
    await agent.close()


@pytest.mark.asyncio
async def test_denied_quota_checks_the_correct_session_id(tmp_path: Path) -> None:
    checker = _AlwaysDeniesChecker()
    agent = _agent(tmp_path, checker)

    await agent.run("Say hi", session_id="tenant-b")

    assert checker.checked_sessions == ["tenant-b"]
    await agent.close()


@pytest.mark.asyncio
async def test_no_quota_checker_means_no_enforcement(tmp_path: Path) -> None:
    """The default (quota_checker=None) must behave identically to before
    ADR-4 — a run succeeds with no quota checks at all."""
    agent = _agent(tmp_path, None)

    result = await agent.run("Say hi", session_id="tenant-c")

    assert result.status == "success"
    await agent.close()


@pytest.mark.asyncio
async def test_allowing_checker_does_not_block_a_successful_run(tmp_path: Path) -> None:
    checker = _AlwaysAllowsChecker()
    agent = _agent(tmp_path, checker)

    result = await agent.run("Say hi", session_id="tenant-d")

    assert result.status == "success"
    assert checker.checked_sessions == ["tenant-d"]
    await agent.close()


@pytest.mark.asyncio
async def test_quota_exceeded_status_is_distinct_from_budget_status(tmp_path: Path) -> None:
    """quota_exceeded (tenant/billing budget) and budget (per-run token
    ceiling) must not collapse into the same RunResult.status — a caller
    branching on status needs to tell them apart."""
    checker = _AlwaysDeniesChecker()
    agent = Agent(
        settings=NullainSettings(),
        provider=_FakeProvider(),
        workspace_root=tmp_path,
        model="m",
        quota_checker=checker,
        max_tokens=1,  # would also trip the token ceiling if checked first
    )

    result = await agent.run("Say hi", session_id="tenant-e")

    # Both checks happen in the same method; max_tokens is checked first
    # in the implementation, but starting at 0 accumulated tokens means
    # the quota check (not the token ceiling) is what actually fires here.
    assert result.status == "quota_exceeded"
    await agent.close()
