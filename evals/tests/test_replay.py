"""Tests for the offline replay provider (issue #45: "replay determinism")."""

from __future__ import annotations

from pathlib import Path

import pytest
from nullain.llm.types import ChatMessage, CompletionChunk, CompletionRequest
from nullain_evals.replay import ReplayExhaustedError, ReplayProvider, dump_responses


def _req() -> CompletionRequest:
    return CompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])


@pytest.mark.asyncio
async def test_generate_replays_in_order() -> None:
    provider = ReplayProvider(
        [CompletionChunk(delta_text="first"), CompletionChunk(delta_text="second")]
    )
    r1 = await provider.generate(_req())
    r2 = await provider.generate(_req())
    assert r1.delta_text == "first"
    assert r2.delta_text == "second"


@pytest.mark.asyncio
async def test_generate_records_seen_requests() -> None:
    provider = ReplayProvider([CompletionChunk(delta_text="ok")])
    req = _req()
    await provider.generate(req)
    assert provider.seen_requests == [req]


@pytest.mark.asyncio
async def test_exhausted_fixture_raises() -> None:
    provider = ReplayProvider([CompletionChunk(delta_text="only one")])
    await provider.generate(_req())
    with pytest.raises(ReplayExhaustedError):
        await provider.generate(_req())


@pytest.mark.asyncio
async def test_stream_yields_the_single_recorded_chunk() -> None:
    provider = ReplayProvider([CompletionChunk(delta_text="streamed")])
    chunks = [c async for c in provider.stream(_req())]
    assert len(chunks) == 1
    assert chunks[0].delta_text == "streamed"


@pytest.mark.asyncio
async def test_health_check_always_true() -> None:
    assert await ReplayProvider([]).health_check() is True


def test_dump_and_load_roundtrip(tmp_path: Path) -> None:
    original = [
        CompletionChunk(delta_text="a", finish_reason="stop"),
        CompletionChunk(delta_text="", tool_calls=None),
    ]
    fixture = tmp_path / "task.json"
    dump_responses(original, fixture)

    provider = ReplayProvider.from_fixture(fixture)
    assert len(provider.responses) == 2
    assert provider.responses[0].delta_text == "a"
    assert provider.responses[0].finish_reason == "stop"
