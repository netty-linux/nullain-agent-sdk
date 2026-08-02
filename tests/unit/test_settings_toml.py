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


def test_load_settings_defaults_when_no_path() -> None:
    """With no config path, defaults are used (cwd toml is NOT auto-read here)."""
    settings = load_settings(None)
    assert settings.ollama_base_url == "https://ollama.com"
    assert settings.ollama_api_key is None
    assert settings.mcp.servers == {}
    # Default router tiers are populated.
    assert "fast" in settings.router.tiers


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
