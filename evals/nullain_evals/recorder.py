"""Nullain Agent SDK evals — records a live provider's responses to a fixture.

``RecordingProvider`` wraps a real ``LLMProvider`` (e.g. ``OllamaCloudProvider``)
transparently — every call is forwarded to the wrapped provider and the
response is both returned to the caller and appended to an in-memory log.
After a live eval run completes, :meth:`RecordingProvider.save` writes that
log to ``evals/fixtures/<task_id>.json`` in exactly the format
``ReplayProvider.from_fixture`` reads, so a task run once against the real
API can be replayed deterministically forever after.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

from nullain.llm.provider import LLMProvider
from nullain.llm.types import CompletionChunk, CompletionRequest

from nullain_evals.replay import dump_responses


class RecordingProvider:
    """``LLMProvider`` decorator that transparently records every response
    from the wrapped real provider, in call order."""

    def __init__(self, wrapped: LLMProvider) -> None:
        self._wrapped = wrapped
        self.recorded: list[CompletionChunk] = []

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        chunk = await self._wrapped.generate(request)
        self.recorded.append(chunk)
        return chunk

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        # Recorded as the fully-drained chunk sequence collapsed into
        # whatever the wrapped provider's own stream yields — mirrors
        # generate() so replay (which only ever emits one chunk per call,
        # see ReplayProvider.stream) stays consistent with what was recorded.
        last: CompletionChunk | None = None
        async for chunk in self._wrapped.stream(request):
            last = chunk
        if last is None:
            raise RuntimeError("wrapped provider's stream() yielded no chunks to record")
        self.recorded.append(last)
        yield last

    async def health_check(self) -> bool:
        return await self._wrapped.health_check()

    def save(self, path: str | Path) -> None:
        """Write everything recorded so far to a fixture file."""
        dump_responses(self.recorded, path)


__all__ = ["RecordingProvider"]
