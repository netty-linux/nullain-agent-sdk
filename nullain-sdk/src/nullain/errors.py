"""Nullain Agent SDK — Exceptions Hierarchy.

All domain exceptions inherit from NullainError.
"""

from typing import Any


class NullainError(Exception):
    """Base exception for all Nullain SDK errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ProviderError(NullainError):
    """Base exception for LLM provider errors."""

    pass


class ProviderTimeoutError(ProviderError):
    """Raised when an LLM provider call times out."""

    pass


class ProviderRateLimitError(ProviderError):
    """Raised when hitting LLM provider rate limits (e.g. HTTP 429)."""

    pass


class ProviderAuthenticationError(ProviderError):
    """Raised when provider authentication fails (e.g. invalid API key)."""

    pass


class ToolError(NullainError):
    """Base exception for tool execution or validation errors."""

    pass


class ToolPermissionError(ToolError):
    """Raised when tool execution is denied by PermissionPolicy."""

    pass


class ToolExecutionError(ToolError):
    """Raised when a tool execution fails inside sandbox."""

    pass


class ToolNotFoundError(ToolError):
    """Raised when attempting to execute an unregistered tool."""

    pass


class RouterError(NullainError):
    """Base exception for model routing errors."""

    pass


class NoModelAvailableError(RouterError):
    """Raised when no suitable model is available in the requested tier."""

    pass


class BudgetExceededError(NullainError):
    """Raised when task token or cost budget is exceeded."""

    pass


class ContextError(NullainError):
    """Raised when context assembly or compaction fails."""

    pass


class SpecValidationError(NullainError):
    """Raised when Plan/Act task specification fails validation."""

    pass


__all__ = [
    "BudgetExceededError",
    "ContextError",
    "NoModelAvailableError",
    "NullainError",
    "ProviderAuthenticationError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "RouterError",
    "SpecValidationError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolPermissionError",
]
