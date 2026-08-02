"""Shared pytest fixtures for Nullain Agent SDK test suite."""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from nullain.events import EventBus
from nullain.llm import CompletionChunk, CompletionRequest
from nullain.llm.provider import LLMProvider
from nullain.tools import ToolRegistry
from nullain_tools import register_default_tools


class FakeProvider(LLMProvider):
    """Configurable fake LLM provider for unit testing."""

    def __init__(self, responses: list[CompletionChunk] | None = None) -> None:
        """Initialize with optional scripted responses."""
        self.responses = list(responses or [])
        self.call_count = 0
        self.requests: list[CompletionRequest] = []

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        """Return next scripted response or default."""
        self.requests.append(request)
        if self.call_count < len(self.responses):
            chunk = self.responses[self.call_count]
            self.call_count += 1
            return chunk
        return CompletionChunk(delta_text="Done")

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        """Stream scripted response as single chunk."""
        chunk = await self.generate(request)
        yield chunk

    async def health_check(self) -> bool:
        """Always healthy in tests."""
        return True


@pytest.fixture
def fake_provider() -> FakeProvider:
    """Create a FakeProvider with no pre-loaded responses."""
    return FakeProvider()


@pytest.fixture
def event_bus() -> EventBus:
    """Create a fresh EventBus instance."""
    return EventBus()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create an isolated workspace directory for testing."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def tool_registry(workspace: Path) -> ToolRegistry:
    """Create ToolRegistry populated with default filesystem tools."""
    registry = ToolRegistry()
    register_default_tools(registry, workspace)
    return registry
