"""Unit tests for EventStorePort — the shared contract between
SQLiteEventStore and PostgresEventStore (docs/FUSION_PLAN.md ADR-2).

Both adapters must give identical resume semantics: append events, read
them back in insertion order, list/resolve sessions the same way. These
tests run the same assertions against SQLiteEventStore (the real,
file-backed adapter — no mocking needed, it's already fast and local) to
pin down the contract PostgresEventStore is required to match; see
test_postgres_event_store.py for PostgresEventStore's own tests against a
mocked asyncpg.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nullain.events import EventStorePort, SQLiteEventStore, UserMessageEvent


def test_sqlite_event_store_satisfies_the_port_protocol() -> None:
    assert isinstance(SQLiteEventStore(":memory:"), EventStorePort)


@pytest.mark.asyncio
async def test_append_and_get_session_events_round_trip(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "sessions.db")
    event = UserMessageEvent(session_id="s1", content="hello")
    await store.append(event)

    events = await store.get_session_events("s1")
    assert len(events) == 1
    assert events[0].content == "hello"  # type: ignore[attr-defined]
    await store.close()


@pytest.mark.asyncio
async def test_get_session_events_preserves_insertion_order(tmp_path: Path) -> None:
    """Ordering is by seq (insertion), not by timestamp — appending events
    with out-of-order timestamps must still come back in append order."""
    store = SQLiteEventStore(tmp_path / "sessions.db")
    await store.append(UserMessageEvent(session_id="s1", content="first", timestamp=100.0))
    await store.append(UserMessageEvent(session_id="s1", content="second", timestamp=1.0))
    await store.append(UserMessageEvent(session_id="s1", content="third", timestamp=50.0))

    events = await store.get_session_events("s1")
    assert [e.content for e in events] == ["first", "second", "third"]  # type: ignore[attr-defined]
    await store.close()


@pytest.mark.asyncio
async def test_get_session_events_for_unknown_session_is_empty(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "sessions.db")
    events = await store.get_session_events("never-existed")
    assert events == []
    await store.close()


@pytest.mark.asyncio
async def test_get_latest_session_id_reflects_most_recent_append(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "sessions.db")
    await store.append(UserMessageEvent(session_id="s1", content="a"))
    await store.append(UserMessageEvent(session_id="s2", content="b"))
    assert await store.get_latest_session_id() == "s2"
    await store.close()


@pytest.mark.asyncio
async def test_get_latest_session_id_on_empty_store_is_none(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "sessions.db")
    assert await store.get_latest_session_id() is None
    await store.close()


@pytest.mark.asyncio
async def test_list_session_ids_oldest_created_first(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "sessions.db")
    await store.append(UserMessageEvent(session_id="s2", content="x"))
    await store.append(UserMessageEvent(session_id="s1", content="y"))
    await store.append(UserMessageEvent(session_id="s2", content="z"))  # s2 again

    assert await store.list_session_ids() == ["s2", "s1"]
    await store.close()


@pytest.mark.asyncio
async def test_close_is_idempotent(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "sessions.db")
    await store.initialize()
    await store.close()
    await store.close()  # must not raise
