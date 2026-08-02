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


class NullainSettings(BaseSettings):
    """Root application settings loaded from nullain.toml or environment."""

    model_config = SettingsConfigDict(
        env_prefix="NULLAIN_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    router: RouterConfig = Field(default_factory=RouterConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
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


__all__ = ["NullainSettings", "RouterConfig", "TierConfig", "load_settings"]
