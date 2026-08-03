"""Nullain Agent SDK — Declarative Settings Loader."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from nullain.hooks import HooksConfig


class TierConfig(BaseModel):
    """Configuration for a specific model tier."""

    models: list[str]
    max_context: int = 32000


class RouterConfig(BaseModel):
    """Router configuration holding tier maps and fallback policies."""

    tiers: dict[str, TierConfig] = Field(
        default_factory=lambda: {
            "fast": TierConfig(models=["gpt-oss:20b"], max_context=32000),
            "balanced": TierConfig(
                models=["qwen3-coder:480b-cloud", "gpt-oss:120b"], max_context=128000
            ),
            "deep": TierConfig(models=["deepseek-v4-pro"], max_context=128000),
        }
    )
    fallback_chain: list[str] = Field(default_factory=lambda: ["deep", "balanced", "fast"])


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server launched via stdio.

    The server is spawned with ``[command, *args]`` as an explicit argv list
    (never a shell). ``auto_approve`` controls the permission level the
    registry assigns to the server's tools: True = ALLOW, False = ASK (default,
    gating tool calls through the human approval loop).
    """

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    auto_approve: bool = False
    enabled: bool = True


class MCPConfig(BaseModel):
    """MCP client configuration: a named map of stdio MCP servers."""

    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class SandboxConfig(BaseModel):
    """OS-level subprocess sandbox configuration.

    - ``enabled`` (default True): when False, the runner uses the NoSandbox
      adapter (no isolation, explicit opt-out; the PermissionPolicy path checks
      still apply).
    - ``required`` (default True): when True and the platform's real adapter
      reports ``available() == False``, the runner raises
      :class:`~nullain.errors.SandboxUnavailableError` (fail-closed) rather than
      executing the subprocess without isolation.
    - ``allow_paths``: extra paths (beyond the workspace root) the sandbox
      permits the child to read/write.
    - ``deny_network`` (default True): request network isolation when the
      platform adapter supports it.
    """

    enabled: bool = True
    required: bool = True
    allow_paths: list[str] = Field(default_factory=list)
    deny_network: bool = True


class NullainSettings(BaseSettings):
    """Root application settings loaded from nullain.toml or environment."""

    model_config = SettingsConfigDict(
        env_prefix="NULLAIN_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    router: RouterConfig = Field(default_factory=RouterConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    ollama_api_key: str | None = None
    ollama_base_url: str = "https://ollama.com"


def load_settings(config_path: str | Path | None = None) -> NullainSettings:
    """Load settings from optional nullain.toml or environment."""
    if config_path and Path(config_path).exists():
        import tomllib

        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return NullainSettings.model_validate(data)
    return NullainSettings()


__all__ = [
    "MCPConfig",
    "MCPServerConfig",
    "NullainSettings",
    "RouterConfig",
    "SandboxConfig",
    "TierConfig",
    "load_settings",
]
