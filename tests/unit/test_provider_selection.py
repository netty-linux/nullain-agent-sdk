"""Tests for Agent's config-driven provider selection (issue #40).

`settings.llm.provider` picks which LLMProvider Agent() builds by default
when none is injected — "ollama" (default, preserving every existing
config's behavior exactly) or "openai" (OpenAICompatibleProvider, works
against any OpenAI-compatible endpoint via openai_base_url).
"""

from pathlib import Path

import pytest
from nullain.agent import Agent
from nullain.config import LLMConfig, NullainSettings
from nullain.llm import OllamaCloudProvider, OpenAICompatibleProvider


def test_agent_defaults_to_ollama_provider(tmp_path: Path) -> None:
    agent = Agent(settings=NullainSettings(), workspace_root=tmp_path)
    assert isinstance(agent._provider, OllamaCloudProvider)  # pyright: ignore[reportPrivateUsage]


def test_agent_selects_ollama_provider_explicitly(tmp_path: Path) -> None:
    settings = NullainSettings(
        llm=LLMConfig(provider="ollama"),
        ollama_api_key="ollama-key",
        ollama_base_url="https://my.ollama.example",
    )
    agent = Agent(settings=settings, workspace_root=tmp_path)
    provider = agent._provider  # pyright: ignore[reportPrivateUsage]
    assert isinstance(provider, OllamaCloudProvider)
    assert provider.api_key == "ollama-key"
    assert provider.base_url == "https://my.ollama.example"


def test_agent_selects_openai_provider(tmp_path: Path) -> None:
    settings = NullainSettings(
        llm=LLMConfig(provider="openai"),
        openai_api_key="sk-test",
        openai_base_url="https://api.openai.com",
    )
    agent = Agent(settings=settings, workspace_root=tmp_path)
    provider = agent._provider  # pyright: ignore[reportPrivateUsage]
    assert isinstance(provider, OpenAICompatibleProvider)
    assert not isinstance(provider, OllamaCloudProvider)
    assert provider.api_key == "sk-test"
    assert provider.base_url == "https://api.openai.com"


def test_agent_selects_openai_provider_against_a_non_openai_base_url(tmp_path: Path) -> None:
    """The whole point of the generic adapter: any OpenAI-compatible
    endpoint works via base_url alone, no provider-specific code."""
    settings = NullainSettings(
        llm=LLMConfig(provider="openai"),
        openai_api_key="sk-or-test",
        openai_base_url="https://openrouter.ai/api",
    )
    agent = Agent(settings=settings, workspace_root=tmp_path)
    provider = agent._provider  # pyright: ignore[reportPrivateUsage]
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == "https://openrouter.ai/api"


def test_agent_unknown_provider_name_raises(tmp_path: Path) -> None:
    settings = NullainSettings(llm=LLMConfig(provider="anthropic"))
    with pytest.raises(ValueError, match=r"Unknown \[llm\] provider"):
        Agent(settings=settings, workspace_root=tmp_path)


def test_explicit_provider_argument_overrides_config_selection(tmp_path: Path) -> None:
    """An explicitly injected provider always wins over settings.llm.provider
    — config-driven selection only kicks in when provider=None."""
    injected = OpenAICompatibleProvider(base_url="https://custom.example")
    settings = NullainSettings(llm=LLMConfig(provider="ollama"))
    agent = Agent(settings=settings, provider=injected, workspace_root=tmp_path)
    assert agent._provider is injected  # pyright: ignore[reportPrivateUsage]
