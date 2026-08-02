"""Nullain Agent SDK — LLM Provider Port Interface."""

from collections.abc import AsyncGenerator
from typing import Protocol, runtime_checkable

from nullain.llm.types import CompletionChunk, CompletionRequest


@runtime_checkable
class LLMProvider(Protocol):
    """Abstract protocol for LLM Providers (Ports & Adapters)."""

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        """Generate a complete response from the LLM model.

        Args:
            request: Completion request specification.

        Returns:
            The complete CompletionChunk response.
        """
        ...

    def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        """Stream response chunks asynchronously from the LLM model.

        Args:
            request: Completion request specification.

        Yields:
            CompletionChunk objects sequentially.
        """
        ...

    async def health_check(self) -> bool:
        """Check availability and connectivity of the provider.

        Returns:
            True if healthy, False otherwise.
        """
        ...


__all__ = ["LLMProvider"]
