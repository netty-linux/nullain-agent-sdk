"""Nullain Agent SDK — Production Agentic Framework for Python."""

import importlib.metadata

from nullain.agent import Agent, AgentLoop, RunResult, RunStatus
from nullain.authority import Authority, Capability
from nullain.config import NullainSettings, load_settings
from nullain.errors import (
    BudgetExceededError,
    ContextError,
    MCPError,
    NoModelAvailableError,
    NullainError,
    PluginError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RouterError,
    SpecValidationError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
)
from nullain.events import (
    BaseEvent,
    CompactionEvent,
    Conversation,
    ErrorEvent,
    EventBus,
    ModelResponseEvent,
    SpecCreatedEvent,
    SpecVerifiedEvent,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from nullain.llm import LLMProvider, OllamaCloudProvider
from nullain.tools import ToolRegistry, tool

__version__ = importlib.metadata.version("nullain-sdk")

# The public API is deliberately minimal (AGENTS.md rule 11). Everything else
# remains importable by its full module path (e.g. ``from nullain.plugins
# import PluginLoader``) but is not part of the top-level ``__all__`` surface
# under SemVer. See docs/api-stability.md for the policy.
__all__ = [
    "Agent",
    "AgentLoop",
    "Authority",
    "BaseEvent",
    "BudgetExceededError",
    "Capability",
    "CompactionEvent",
    "ContextError",
    "Conversation",
    "ErrorEvent",
    "EventBus",
    "LLMProvider",
    "MCPError",
    "ModelResponseEvent",
    "NoModelAvailableError",
    "NullainError",
    "NullainSettings",
    "OllamaCloudProvider",
    "PluginError",
    "ProviderAuthenticationError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "RouterError",
    "RunResult",
    "RunStatus",
    "SpecCreatedEvent",
    "SpecValidationError",
    "SpecVerifiedEvent",
    "StreamDeltaEvent",
    "ToolCallEvent",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolPermissionError",
    "ToolRegistry",
    "ToolResultEvent",
    "UserMessageEvent",
    "load_settings",
    "tool",
]
