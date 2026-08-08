"""Nullain Agent SDK evals — offline replay LLM provider.

``ReplayProvider`` implements the SDK's ``LLMProvider`` port by replaying a
pre-recorded, ordered sequence of ``CompletionChunk`` responses instead of
calling a real API — this is what makes ``make evals`` deterministic and
CI-safe (issue #45's "offline mode"). It has no network access at all: every
call is a list index into a fixture loaded from JSON, so a task's outcome
depends only on the SDK's own logic, never on API availability, rate limits,
or model nondeterminism.

Recordings live at ``evals/fixtures/<task_id>.json`` — a JSON array of
``CompletionChunk``-shaped objects, in the exact order the real ``AgentLoop``
would have requested them for that task. ``record_from_live`` (used by the
live-mode runner) captures a real run's responses into this same format, so
a task authored/verified live can be replayed offline forever after without
re-touching the API.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path

from nullain.llm.types import CompletionChunk, CompletionRequest


class ReplayExhaustedError(RuntimeError):
    """Raised when the agent makes more provider calls than the fixture has
    recorded responses for — always a real bug (a task fixture out of sync
    with the current harness behavior, or an agent stuck retrying), never
    something a grader should silently paper over."""


class ReplayProvider:
    """``LLMProvider`` that replays a fixed, ordered list of responses.

    Each call to :meth:`generate` (or a full drain of :meth:`stream`) consumes
    exactly one recorded response, in order — matching how
    ``FakeSequenceProvider`` is already used across the SDK's own test suite
    (see ``tests/unit/test_facade_session_persistence.py``), just loaded from
    a JSON fixture file instead of constructed inline.
    """

    def __init__(self, responses: list[CompletionChunk]) -> None:
        #: The recorded response sequence, in replay order — public so a
        #: test can assert against a loaded fixture's contents directly.
        self.responses = list(responses)
        self.call_count = 0
        #: Every request this provider was asked to answer, in order — lets
        #: a grader or the recorder inspect exactly what the agent sent
        #: without needing to instrument the loop itself.
        self.seen_requests: list[CompletionRequest] = []

    @classmethod
    def from_fixture(cls, path: str | Path) -> ReplayProvider:
        """Load a recorded response sequence from a JSON fixture file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        responses = [CompletionChunk.model_validate(item) for item in data]
        return cls(responses)

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        self.seen_requests.append(request)
        if self.call_count >= len(self.responses):
            raise ReplayExhaustedError(
                f"replay fixture exhausted after {self.call_count} call(s); "
                "the agent requested another completion the fixture doesn't "
                "cover — either the fixture is stale or the agent is looping"
            )
        chunk = self.responses[self.call_count]
        self.call_count += 1
        return chunk

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        # Fixtures are recorded as complete (non-streamed) chunks — replay
        # yields the whole response as a single chunk, matching how a
        # generate()-only provider is already treated elsewhere in the SDK
        # (e.g. FakeSequenceProvider.stream in the facade session tests).
        yield await self.generate(request)

    async def health_check(self) -> bool:
        return True


def dump_responses(responses: list[CompletionChunk], path: str | Path) -> None:
    """Write a recorded response sequence to a JSON fixture file, matching
    the exact schema :meth:`ReplayProvider.from_fixture` reads back."""
    payload = [chunk.model_dump(mode="json") for chunk in responses]
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["ReplayExhaustedError", "ReplayProvider", "dump_responses"]
