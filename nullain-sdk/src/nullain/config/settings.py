"""Nullain Agent SDK — Declarative Settings Loader."""

import os
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from nullain.hooks import HooksConfig
from nullain.lsp.config import LSPConfig, LSPServerConfig


class LLMConfig(BaseModel):
    """Selects which LLM provider the ``Agent`` facade builds by default when
    none is injected (issue #40).

    ``provider = "ollama"`` (default) builds ``OllamaCloudProvider`` from the
    top-level ``ollama_api_key``/``ollama_base_url`` settings.
    ``provider = "openai"`` builds ``OpenAICompatibleProvider`` from the
    top-level ``openai_api_key``/``openai_base_url`` settings — and, via
    ``openai_base_url``, works against any OpenAI-compatible endpoint
    (OpenRouter, Together, Groq, vLLM, LM Studio, ...), not only OpenAI
    itself. Both providers' credentials live as direct ``NullainSettings``
    fields (not nested under ``[llm]``) for the same reason
    ``ollama_api_key``/``ollama_base_url`` already do: ``BaseSettings``'
    ``AliasChoices``-based bare-env-var resolution (``OLLAMA_API_KEY``,
    ``OPENAI_API_KEY`` — the names every provider's own SDK/CLI already
    expects, with no ``NULLAIN_`` prefix required) only applies to fields
    declared directly on the ``BaseSettings`` root; a field nested inside a
    plain ``BaseModel`` sub-section does not inherit it (confirmed live: an
    earlier ``[llm.openai]`` nested design silently dropped both the TOML
    value and the env var).

    This selects ONE provider for the whole agent. Per-tier provider routing
    (fast/balanced/deep each on a different provider) is tracked as a
    follow-up, not yet implemented — ``ModelRouter`` still only resolves a
    model name, not a ``(provider, model)`` pair.
    """

    provider: str = "ollama"


class AgentConfig(BaseModel):
    """Per-run budget/limits for ``AgentLoop`` (M18).

    ``max_tokens`` is a *cumulative* ceiling across every step of one
    ``run()`` — not a per-message limit — so it scales with how much a task
    actually needs, not with any single model call's context window. The
    100k default `AgentLoop` shipped with was sized for short single-shot
    tasks and cuts off well before a real coding task (multi-file feature
    work, debugging a SaaS codebase, several rounds of self-correction) can
    finish; 2M gives a task room to run to actual completion while still
    stopping a genuinely runaway loop before it can spend an unbounded
    amount on one run. Set to ``null`` in TOML (``None`` here) to disable
    the token ceiling entirely — the run is then bounded only by
    ``max_steps`` and ``timeout``.

    ``max_steps`` had the same problem: 25 ReAct iterations is enough for a
    single-file edit but not for a real session touching several files
    (write a component, adjust an env file, re-run, fix an error, ...) —
    exactly the shape of task this SDK targets. 100 gives that room while
    loop detection (a separate, independent guard) still catches genuine
    thrashing well before the step cap would.
    """

    max_steps: int = 100
    max_tokens: int | None = 2_000_000
    timeout: float = 300.0
    #: Lowest task complexity that still runs the Plan phase (the
    #: ``emit_task_spec`` call preceding the Act loop). ``"medium"`` — the
    #: default — plans for MEDIUM and HIGH, which is what `AgentLoop` has
    #: always done. ``"high"`` plans only for explicitly complex work, and
    #: ``"never"`` disables the phase outright.
    #:
    #: Worth tuning for a deployment whose tasks are mostly conversational:
    #: ``IntentParser`` falls back to MEDIUM whenever no keyword heuristic
    #: matches and no ``router.classifier_model`` is configured, so in a
    #: general-purpose chat product effectively *every* turn plans, and a
    #: plan for "what's the weather" costs a model round-trip and invites a
    #: spec that doesn't fit the request. Coding deployments should keep
    #: the default — planning is what makes multi-file work coherent.
    plan_complexity_threshold: Literal["medium", "high", "never"] = "medium"
    # Wall-clock timeout for a single bash/git subprocess (M20). The 120s
    # default `execute_subprocess` shipped with cuts off real coding-session
    # commands early — dependency installs, full test suites, builds — well
    # before they finish, well before the ReAct loop even gets a chance to
    # decide the command is stuck. 300s gives those room; a genuinely hung
    # process is still killed rather than left running forever.
    bash_timeout: float = 300.0


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
    #: Optional model used by :class:`~nullain.router.intent.IntentParser` to
    #: classify a task when the deterministic heuristics are not confident
    #: (M11.3). When None, the parser stays heuristic-only.
    classifier_model: str | None = None


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


class WebFetchConfig(BaseModel):
    """HTTP headers the ``web_fetch`` tool sends on every request.

    Defaults identify the SDK honestly as an automated client
    (``User-Agent: Nullain-Agent-SDK/0.1 (+web_fetch)``) — many sites
    (news outlets, paywalled content, some API docs) reject requests from
    an unrecognized bot User-Agent with 401/403/429, which is the site's
    own anti-scraping policy working as intended, not a bug to route
    around by spoofing a browser. An operator who has a legitimate reason
    to send different headers (a site they own, an internal service, a
    documented partnership) can override any of these explicitly via
    ``nullain.toml``'s ``[web_fetch]`` section — this is an intentional,
    visible opt-in, not a silent default change.
    """

    user_agent: str = "Nullain-Agent-SDK/0.1 (+web_fetch)"
    accept: str = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    accept_language: str = "en-US,en;q=0.9"
    #: Base URL of a self-hosted SearXNG instance (e.g.
    #: ``http://searxng:8080`` on an internal Docker network) — when set,
    #: ``web_search`` tries it first and falls back to DuckDuckGo scraping
    #: on any failure (unreachable, timeout, malformed response). None
    #: (the default) uses DuckDuckGo directly, unchanged from before this
    #: setting existed. A self-hosted meta-search instance aggregates
    #: multiple upstream engines and isn't subject to any single engine's
    #: bot detection the way scraping DuckDuckGo directly is — but running
    #: it is the operator's own infrastructure commitment, hence opt-in
    #: rather than a new default.
    searxng_base_url: str | None = None
    #: When True, ``web_fetch`` renders every fetch with a real headless
    #: browser (Crawl4AI/Playwright) before falling back to plain httpx —
    #: solves pages whose content only appears after JavaScript runs, and
    #: as a side effect of looking like a real browser, some (not all)
    #: sites that block bare HTTP clients. Requires the SDK's ``crawl``
    #: extra and a Chromium install on the host; opt-in because of that
    #: infrastructure cost (~600MB, browser memory per instance), not a
    #: new default. False (the default) is unchanged from before this
    #: setting existed.
    use_crawl4ai: bool = False


class PluginEntryConfig(BaseModel):
    """Configuration for a single plugin entry.

    The plugin's launch command, capabilities, tools, and SBOM live in the
    SIGNED manifest referenced by ``manifest`` (so they cannot be substituted
    without breaking the signature). The operator config only references the
    manifest and optionally narrows the granted capabilities / approval mode.
    """

    enabled: bool = True
    manifest: str  # path to the signed manifest JSON
    auto_approve: bool = False  # permission_level for the plugin's tools (ASK default)
    # Per-plugin narrowing of the global capability grant. None = use the global
    # [plugins].allowed_capabilities; a list intersects further (P4.24 meet).
    allowed_capabilities: list[str] | None = None


class PluginsConfig(BaseModel):
    """Plugin system configuration (P4.25).

    - ``enabled`` (default True): master switch for plugin loading.
    - ``require_signature`` (default True): fail-closed — an unsigned plugin is
      refused unless this is explicitly False (trusted-local opt-in). A signed
      plugin with no verifier backend installed is always refused.
    - ``trusted_keys``: map of key_id -> base64 Ed25519 public key. A signature
      whose key_id is not in this map does not verify.
    - ``allowed_capabilities``: the global capability grant (values are
      Capability strings: read/write/exec/network/spawn). A plugin tool is
      registered only if its required capabilities are a subset of the plugin's
      declared capabilities ∩ this grant. Empty (default) => no plugin tools
      load — the operator must explicitly grant capabilities (deny by default).
    - ``entries``: named plugin entries, each referencing a signed manifest.
    """

    enabled: bool = True
    require_signature: bool = True
    trusted_keys: dict[str, str] = Field(default_factory=dict)
    allowed_capabilities: list[str] = Field(default_factory=list)
    entries: dict[str, PluginEntryConfig] = Field(default_factory=dict)


class NullainSettings(BaseSettings):
    """Root application settings loaded from nullain.toml or environment."""

    model_config = SettingsConfigDict(
        env_prefix="NULLAIN_",
        env_nested_delimiter="__",
        extra="ignore",
        # Required for ollama_api_key's AliasChoices below to still accept
        # the plain field name ("ollama_api_key") when settings are loaded
        # from a nullain.toml dict via model_validate() rather than from
        # environment variables — without this, a validation_alias makes the
        # field only reachable by that alias, breaking `ollama_api_key = ...`
        # in the TOML file itself.
        populate_by_name=True,
    )

    agent: AgentConfig = Field(default_factory=AgentConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    lsp: LSPConfig = Field(default_factory=LSPConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    web_fetch: WebFetchConfig = Field(default_factory=WebFetchConfig)
    # Accepts both the prefixed env var (NULLAIN_OLLAMA_API_KEY, consistent
    # with every other setting here) and the bare OLLAMA_API_KEY — the name
    # docs/configuration.md always documented and the one other CLIs' API-key
    # env vars follow (e.g. ANTHROPIC_API_KEY, no tool-specific prefix). The
    # prefixed alias is listed first so it wins if both happen to be set.
    ollama_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("NULLAIN_OLLAMA_API_KEY", "OLLAMA_API_KEY")
    )
    ollama_base_url: str = "https://ollama.com"
    # Same AliasChoices pattern as ollama_api_key above (issue #40):
    # NULLAIN_OPENAI_API_KEY wins if both it and the bare OPENAI_API_KEY —
    # the name every OpenAI-compatible SDK/CLI already expects — are set.
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("NULLAIN_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    # Defaults to OpenAI itself; pointing this at any other OpenAI-compatible
    # chat-completions endpoint (OpenRouter, Together, Groq, vLLM, LM
    # Studio, ...) is the entire mechanism for using a different provider —
    # no other setting changes.
    openai_base_url: str = "https://api.openai.com"
    #: Provider selection (issue #40) — see LLMConfig's docstring.
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_settings(config_path: str | Path | None = None) -> NullainSettings:
    """Load settings from a nullain.toml file, or the environment alone.

    ``config_path`` resolution, in order:

    1. The explicit argument, if given.
    2. ``NULLAIN_CONFIG``, if set.
    3. ``./nullain.toml`` (relative to the current working directory), if it
       exists.

    A ``config_path`` that does not resolve to an existing file (including
    when none of the above applies) falls back to environment-only settings
    — this is not an error, since a deployment may configure everything via
    env vars and have no TOML file at all. Every caller building an
    ``Agent`` (or anything else that touches ``ollama_api_key``) should call
    this the same way — the previous default of "only look at env vars
    unless a path is explicitly passed" meant most call sites silently never
    read a ``nullain.toml`` the wizard or a `mcp add` had just written.
    """
    if config_path is None:
        env_path = os.environ.get("NULLAIN_CONFIG")
        config_path = env_path if env_path else "nullain.toml"
    if config_path and Path(config_path).exists():
        import tomllib

        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return NullainSettings.model_validate(data)
    return NullainSettings()


__all__ = [
    "AgentConfig",
    "LSPConfig",
    "LSPServerConfig",
    "MCPConfig",
    "MCPServerConfig",
    "NullainSettings",
    "PluginEntryConfig",
    "PluginsConfig",
    "RouterConfig",
    "SandboxConfig",
    "TierConfig",
    "WebFetchConfig",
    "load_settings",
]
