"""Tests for declarative settings loading from nullain.toml and environment."""

from pathlib import Path

import pytest
from nullain.config import MCPServerConfig, NullainSettings, load_settings


def test_load_settings_from_toml(tmp_path: Path) -> None:
    """Router tiers, ollama endpoint and MCP servers all parse from TOML."""
    toml = tmp_path / "nullain.toml"
    toml.write_text(
        """
ollama_base_url = "https://my.ollama.example"
ollama_api_key = "secret-key"

[router]
fallback_chain = ["fast", "balanced", "deep"]

[router.tiers.fast]
models = ["m_fast_a", "m_fast_b"]
max_context = 8000

[router.tiers.deep]
models = ["m_deep"]

[mcp.servers.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/ws"]
auto_approve = true
enabled = true

[mcp.servers.git]
command = "uvx"
args = ["mcp-server-git"]
"""
    )

    settings = load_settings(toml)

    assert settings.ollama_base_url == "https://my.ollama.example"
    assert settings.ollama_api_key == "secret-key"
    assert settings.router.fallback_chain == ["fast", "balanced", "deep"]
    assert settings.router.tiers["fast"].models == ["m_fast_a", "m_fast_b"]
    assert settings.router.tiers["fast"].max_context == 8000
    assert settings.router.tiers["deep"].models == ["m_deep"]
    assert settings.mcp.servers["filesystem"].command == "npx"
    assert settings.mcp.servers["filesystem"].args == [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/ws",
    ]
    assert settings.mcp.servers["filesystem"].auto_approve is True
    assert settings.mcp.servers["filesystem"].enabled is True
    # auto_approve defaults to False when omitted.
    assert settings.mcp.servers["git"].auto_approve is False


def test_load_settings_defaults_when_no_cwd_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no config path and no ./nullain.toml in cwd, defaults are used."""
    monkeypatch.delenv("NULLAIN_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # empty dir — no nullain.toml here
    settings = load_settings(None)
    assert settings.ollama_base_url == "https://ollama.com"
    assert settings.ollama_api_key is None
    assert settings.mcp.servers == {}
    # Default router tiers are populated.
    assert "fast" in settings.router.tiers
    # Default agent budget (M18/M19): large enough for real multi-file
    # feature work, not the old 25 steps / 100k tokens that cut off
    # long-running coding tasks early.
    assert settings.agent.max_steps == 100
    assert settings.agent.max_tokens == 2_000_000
    assert settings.agent.timeout == 300.0


def test_load_settings_agent_section_from_toml(tmp_path: Path) -> None:
    """[agent] overrides the default budget/limits."""
    toml = tmp_path / "nullain.toml"
    toml.write_text(
        """
[agent]
max_steps = 50
max_tokens = 5000000
timeout = 600.0
"""
    )
    settings = load_settings(toml)
    assert settings.agent.max_steps == 50
    assert settings.agent.max_tokens == 5_000_000
    assert settings.agent.timeout == 600.0


def test_agent_config_max_tokens_none_disables_ceiling() -> None:
    """AgentConfig accepts max_tokens=None directly (no ceiling).

    TOML has no `null` literal — a user disables the ceiling by omitting
    the key entirely (which already keeps the default 2,000,000, so that's
    not the right way to test "disabled" either) or by constructing
    AgentConfig programmatically. This test covers the Python-level
    contract that None is a valid, meaningful value for the field.
    """
    from nullain.config import AgentConfig

    config = AgentConfig(max_tokens=None)
    assert config.max_tokens is None


def test_load_settings_no_path_reads_cwd_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: load_settings(None) previously never read ./nullain.toml,
    so every caller that didn't pass an explicit path (Agent's default,
    `nullain doctor`, `nullain mcp list`) silently ignored a nullain.toml
    the user (or the first-run setup wizard) had just written in the
    current directory. It must now be read the same way _find_config_path's
    CLI helpers already resolve it: NULLAIN_CONFIG, else ./nullain.toml."""
    monkeypatch.delenv("NULLAIN_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nullain.toml").write_text('ollama_api_key = "from-cwd-toml"\n', encoding="utf-8")

    settings = load_settings(None)
    assert settings.ollama_api_key == "from-cwd-toml"


def test_load_settings_no_path_respects_nullain_config_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NULLAIN_CONFIG still takes precedence over ./nullain.toml when both
    could apply."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nullain.toml").write_text('ollama_api_key = "from-cwd"\n', encoding="utf-8")
    other = tmp_path / "elsewhere.toml"
    other.write_text('ollama_api_key = "from-env-path"\n', encoding="utf-8")
    monkeypatch.setenv("NULLAIN_CONFIG", str(other))

    settings = load_settings(None)
    assert settings.ollama_api_key == "from-env-path"


def test_mcp_server_config_defaults() -> None:
    cfg = MCPServerConfig(command="runme")
    assert cfg.args == []
    assert cfg.env == {}
    assert cfg.auto_approve is False
    assert cfg.enabled is True


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """NULLAIN_-prefixed env vars override defaults (pydantic-settings)."""
    monkeypatch.setenv("NULLAIN_OLLAMA_API_KEY", "env-key")
    monkeypatch.setenv("NULLAIN_OLLAMA_BASE_URL", "https://env.ollama.example")
    settings = NullainSettings()
    assert settings.ollama_api_key == "env-key"
    assert settings.ollama_base_url == "https://env.ollama.example"


def test_settings_accepts_bare_ollama_api_key_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: docs/configuration.md documents OLLAMA_API_KEY (no prefix)
    as the way to set the key, but every other setting requires the
    NULLAIN_ prefix — the bare name silently did nothing. ollama_api_key now
    accepts both via AliasChoices, matching the documented behavior and the
    convention other CLIs use for their API-key env var (e.g.
    ANTHROPIC_API_KEY, no tool-specific prefix)."""
    monkeypatch.delenv("NULLAIN_OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_API_KEY", "bare-key")
    settings = NullainSettings()
    assert settings.ollama_api_key == "bare-key"


def test_settings_prefixed_ollama_api_key_wins_over_bare(monkeypatch: pytest.MonkeyPatch) -> None:
    """When both env vars are set, the NULLAIN_-prefixed one takes precedence
    (consistent with every other setting being NULLAIN_-prefixed)."""
    monkeypatch.setenv("NULLAIN_OLLAMA_API_KEY", "prefixed-key")
    monkeypatch.setenv("OLLAMA_API_KEY", "bare-key")
    settings = NullainSettings()
    assert settings.ollama_api_key == "prefixed-key"
