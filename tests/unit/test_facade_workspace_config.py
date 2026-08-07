"""Regression tests: Agent() reads <workspace_root>/nullain.toml (M17).

Found via manual testing of the first-run setup wizard: the wizard wrote
nullain.toml with the user's chosen models and API key, but Agent() (built
with no explicit `settings=`) still used the built-in defaults and no key —
because Agent.__init__ called load_settings() with no config_path, which
only ever checked NULLAIN_CONFIG or ./nullain.toml relative to the
*process's* cwd, never workspace_root. In the CLI this happened to work only
when cwd == workspace (the common case), silently breaking whenever they
differ (e.g. `nullain run --workspace ../other-project`).
"""

from pathlib import Path

import pytest
from nullain.agent import Agent
from nullain.llm import CompletionChunk, CompletionRequest, LLMProvider


class _NoopProvider(LLMProvider):
    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        return CompletionChunk(delta_text="unused")

    async def stream(self, request: CompletionRequest):
        yield CompletionChunk(delta_text="unused")

    async def health_check(self) -> bool:
        return True


def test_agent_reads_workspace_nullain_toml_regardless_of_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Agent(workspace_root=X) must read X/nullain.toml even when the
    process's cwd is somewhere else entirely."""
    monkeypatch.delenv("NULLAIN_CONFIG", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("NULLAIN_OLLAMA_API_KEY", raising=False)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nullain.toml").write_text(
        "\n".join(
            [
                'ollama_api_key = "workspace-key"',
                "[router.tiers.balanced]",
                'models = ["custom-model:cloud"]',
            ]
        ),
        encoding="utf-8",
    )

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # process cwd is NOT the workspace

    agent = Agent(provider=_NoopProvider(), workspace_root=str(workspace))
    assert agent._settings.ollama_api_key == "workspace-key"  # type: ignore[reportPrivateUsage]
    assert agent._settings.router.tiers["balanced"].models == [  # type: ignore[reportPrivateUsage]
        "custom-model:cloud"
    ]


def test_agent_falls_back_to_env_when_no_workspace_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NULLAIN_CONFIG", raising=False)
    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")

    workspace = tmp_path / "workspace"
    workspace.mkdir()  # no nullain.toml here

    agent = Agent(provider=_NoopProvider(), workspace_root=str(workspace))
    assert agent._settings.ollama_api_key == "env-key"  # type: ignore[reportPrivateUsage]


def test_agent_nullain_config_env_var_wins_over_workspace_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NULLAIN_CONFIG is an explicit global override and must still win over
    a workspace-local nullain.toml, matching load_settings' own precedence."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nullain.toml").write_text('ollama_api_key = "workspace-key"\n', encoding="utf-8")

    override = tmp_path / "override.toml"
    override.write_text('ollama_api_key = "override-key"\n', encoding="utf-8")
    monkeypatch.setenv("NULLAIN_CONFIG", str(override))

    agent = Agent(provider=_NoopProvider(), workspace_root=str(workspace))
    assert agent._settings.ollama_api_key == "override-key"  # type: ignore[reportPrivateUsage]


def test_agent_explicit_settings_bypasses_workspace_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicitly-passed settings= is used as-is — no workspace/env lookup."""
    from nullain.config import NullainSettings

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nullain.toml").write_text('ollama_api_key = "workspace-key"\n', encoding="utf-8")

    explicit = NullainSettings(ollama_api_key="explicit-key")
    agent = Agent(provider=_NoopProvider(), workspace_root=str(workspace), settings=explicit)
    assert agent._settings.ollama_api_key == "explicit-key"  # type: ignore[reportPrivateUsage]
