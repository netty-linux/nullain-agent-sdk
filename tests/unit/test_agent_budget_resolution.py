"""Tests for Agent's max_steps/max_tokens/timeout resolution (M18).

Raised by real usage: the SDK's 100k-token AgentLoop default was too tight
for real coding tasks (multi-file features, several rounds of
self-correction, debugging a larger codebase) and cut work off mid-task —
and neither Agent nor the CLI exposed any way to raise it without editing
AgentLoop's own default. Agent now resolves max_steps/max_tokens/timeout
from settings.agent (nullain.toml's [agent] section) when not passed
explicitly, with a much larger default (2,000,000) and an explicit
opt-out (max_tokens=None) to disable the ceiling entirely.
"""

from pathlib import Path

import pytest
from nullain.agent import Agent
from nullain.config import AgentConfig, NullainSettings
from nullain.llm import CompletionChunk, CompletionRequest, LLMProvider


class _NoopProvider(LLMProvider):
    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        return CompletionChunk(delta_text="unused")

    async def stream(self, request: CompletionRequest):
        yield CompletionChunk(delta_text="unused")

    async def health_check(self) -> bool:
        return True


def test_agent_defaults_to_settings_agent_budget(tmp_path: Path) -> None:
    """With nothing passed, Agent uses settings.agent.* (default 100/2M/300)."""
    agent = Agent(provider=_NoopProvider(), workspace_root=str(tmp_path))
    assert agent._max_steps == 100  # type: ignore[reportPrivateUsage]
    assert agent._max_tokens == 2_000_000  # type: ignore[reportPrivateUsage]
    assert agent._timeout == 300.0  # type: ignore[reportPrivateUsage]


def test_agent_reads_agent_section_from_workspace_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A workspace nullain.toml's [agent] section is used when nothing is
    passed explicitly to Agent()."""
    monkeypatch.delenv("NULLAIN_CONFIG", raising=False)
    (tmp_path / "nullain.toml").write_text(
        "[agent]\nmax_steps = 60\nmax_tokens = 8000000\ntimeout = 900.0\n", encoding="utf-8"
    )
    agent = Agent(provider=_NoopProvider(), workspace_root=str(tmp_path))
    assert agent._max_steps == 60  # type: ignore[reportPrivateUsage]
    assert agent._max_tokens == 8_000_000  # type: ignore[reportPrivateUsage]
    assert agent._timeout == 900.0  # type: ignore[reportPrivateUsage]


def test_agent_explicit_max_steps_overrides_config(tmp_path: Path) -> None:
    settings = NullainSettings(agent=AgentConfig(max_steps=60))
    agent = Agent(
        provider=_NoopProvider(), workspace_root=str(tmp_path), settings=settings, max_steps=10
    )
    assert agent._max_steps == 10  # type: ignore[reportPrivateUsage]


def test_agent_explicit_max_tokens_overrides_config(tmp_path: Path) -> None:
    settings = NullainSettings(agent=AgentConfig(max_tokens=8_000_000))
    agent = Agent(
        provider=_NoopProvider(),
        workspace_root=str(tmp_path),
        settings=settings,
        max_tokens=500_000,
    )
    assert agent._max_tokens == 500_000  # type: ignore[reportPrivateUsage]


def test_agent_explicit_none_max_tokens_disables_ceiling_regardless_of_config(
    tmp_path: Path,
) -> None:
    """max_tokens=None is distinct from "omitted" — it disables the ceiling
    even when nullain.toml configures a finite one."""
    settings = NullainSettings(agent=AgentConfig(max_tokens=8_000_000))
    agent = Agent(
        provider=_NoopProvider(), workspace_root=str(tmp_path), settings=settings, max_tokens=None
    )
    assert agent._max_tokens is None  # type: ignore[reportPrivateUsage]


def test_agent_omitted_max_tokens_falls_back_to_config_not_none(tmp_path: Path) -> None:
    """Omitting max_tokens entirely must NOT be treated as max_tokens=None —
    it must fall back to settings.agent.max_tokens. This is the crux of the
    _Unset sentinel: a plain `= None` default would make this
    indistinguishable from the explicit-disable case above."""
    settings = NullainSettings(agent=AgentConfig(max_tokens=8_000_000))
    agent = Agent(provider=_NoopProvider(), workspace_root=str(tmp_path), settings=settings)
    assert agent._max_tokens == 8_000_000  # type: ignore[reportPrivateUsage]
