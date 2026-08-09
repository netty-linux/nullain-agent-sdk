"""Unit tests for PostgresEventStore — mocked asyncpg (no live Postgres/
Supabase instance in CI; the adapter's own logic under test is the SQL it
issues, auto-initialize-on-first-use, and error wrapping, not asyncpg's or
Postgres's own behavior)."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest
from nullain.errors import NullainError
from nullain.events import EventStorePort, UserMessageEvent
from nullain.events.postgres_store import EventStoreConnectionError, PostgresEventStore


class _FakeAcquireCtx:
    """Mimics asyncpg's `pool.acquire()` async context manager."""

    def __init__(self, conn: MagicMock) -> None:
        self._conn = conn

    async def __aenter__(self) -> MagicMock:
        return self._conn

    async def __aexit__(self, *exc_info: object) -> None:
        return None


@pytest.fixture
def fake_asyncpg(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Installs a fake `asyncpg` module so PostgresEventStore's lazy
    `importlib.import_module("asyncpg")` resolves without the real
    dependency installed."""
    fake_module = ModuleType("asyncpg")

    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.executemany = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_FakeAcquireCtx(conn))
    pool.close = AsyncMock()

    fake_module.create_pool = AsyncMock(return_value=pool)  # type: ignore[attr-defined]
    fake_module._conn = conn  # type: ignore[attr-defined]
    fake_module._pool = pool  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "asyncpg", fake_module)
    return fake_module


def test_missing_asyncpg_dependency_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "asyncpg", None)
    with pytest.raises(ImportError, match=r"pip install nullain-sdk\[postgres\]"):
        PostgresEventStore(dsn="postgresql://localhost/test")


def test_satisfies_event_store_port_protocol(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    assert isinstance(store, EventStorePort)


def test_event_store_connection_error_is_a_nullain_error() -> None:
    assert issubclass(EventStoreConnectionError, NullainError)


@pytest.mark.asyncio
async def test_initialize_creates_pool_and_schema(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()

    fake_asyncpg.create_pool.assert_called_once()
    assert fake_asyncpg._conn.execute.call_count == 2  # CREATE TABLE + CREATE INDEX


@pytest.mark.asyncio
async def test_initialize_is_idempotent(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()
    await store.initialize()
    fake_asyncpg.create_pool.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_wraps_failure_in_connection_error(fake_asyncpg: ModuleType) -> None:
    fake_asyncpg.create_pool.side_effect = RuntimeError("connection refused")
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    with pytest.raises(EventStoreConnectionError, match="Failed to initialize"):
        await store.initialize()


@pytest.mark.asyncio
async def test_append_auto_initializes_and_enqueues(fake_asyncpg: ModuleType) -> None:
    """append() returns once the event is queued, not once it's written —
    the write happens on the background drain task, verified here via
    flush() (the explicit synchronization point for a caller that needs
    the durably-written guarantee back)."""
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    event = UserMessageEvent(session_id="s1", content="hello")

    await store.append(event)
    fake_asyncpg.create_pool.assert_called_once()  # auto-initialized
    await store.flush()

    insert_calls = fake_asyncpg._conn.executemany.call_args_list
    assert len(insert_calls) == 1
    sql, rows = insert_calls[0].args
    assert "INSERT INTO events" in sql
    assert len(rows) == 1
    assert rows[0][0] == event.id
    assert rows[0][1] == "s1"
    await store.close()


@pytest.mark.asyncio
async def test_append_drain_failure_is_logged_not_raised(fake_asyncpg: ModuleType) -> None:
    """A Postgres-side failure during the background drain must never
    surface as an exception from append() — append() already returned
    successfully (the event was accepted onto the queue) by the time the
    drain even runs. There is no caller left to raise to."""
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()
    fake_asyncpg._conn.executemany.side_effect = RuntimeError("timeout")

    await store.append(UserMessageEvent(session_id="s1", content="x"))
    await store.flush()  # must not raise despite the drain failing
    await store.close()


@pytest.mark.asyncio
async def test_append_after_close_raises(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()
    await store.close()
    with pytest.raises(EventStoreConnectionError, match="closed"):
        await store.append(UserMessageEvent(session_id="s1", content="x"))


@pytest.mark.asyncio
async def test_flush_waits_for_all_queued_events_to_drain(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()

    for i in range(5):
        await store.append(UserMessageEvent(session_id="s1", content=f"msg-{i}"))
    await store.flush()

    written_ids = [
        row[0] for call in fake_asyncpg._conn.executemany.call_args_list for row in call.args[1]
    ]
    assert len(written_ids) == 5
    await store.close()


@pytest.mark.asyncio
async def test_close_flushes_pending_events_before_closing_pool(fake_asyncpg: ModuleType) -> None:
    """No event loss on shutdown: close() must drain whatever is still
    queued before it stops the drain task and closes the pool."""
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()

    await store.append(UserMessageEvent(session_id="s1", content="last one"))
    await store.close()

    written_ids = [
        row[0] for call in fake_asyncpg._conn.executemany.call_args_list for row in call.args[1]
    ]
    assert len(written_ids) == 1
    fake_asyncpg._pool.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_is_idempotent_with_drain_task(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()
    await store.append(UserMessageEvent(session_id="s1", content="x"))
    await store.close()
    await store.close()  # must not raise
    fake_asyncpg._pool.close.assert_called_once()


@pytest.mark.asyncio
async def test_reads_flush_pending_writes_first(fake_asyncpg: ModuleType) -> None:
    """A resume/list/latest-session read must observe every append() that
    happened-before it — never silently miss events still sitting in the
    queue. Verified indirectly: after append() (no explicit flush), a
    read call still triggers the drain (executemany called) before the
    read's own SELECT runs."""
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()
    await store.append(UserMessageEvent(session_id="s1", content="x"))

    await store.list_session_ids()

    assert fake_asyncpg._conn.executemany.call_count == 1
    await store.close()


@pytest.mark.asyncio
async def test_seq_ordering_survives_concurrent_appends(fake_asyncpg: ModuleType) -> None:
    """The invariant the whole module depends on: seq is BIGSERIAL,
    assigned at INSERT time, and every read path orders by it. A
    concurrent multi-connection drain would let batches race and write
    events out of the order they were actually appended in, silently
    corrupting resume/replay. The single sequential drain task must
    preserve strict append-order regardless of how concurrently append()
    itself was called."""
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()

    events = [UserMessageEvent(session_id="s1", content=f"msg-{i}") for i in range(50)]
    # Concurrent append() calls — asyncio.gather runs these interleaved,
    # not sequentially, to actually exercise the ordering guarantee
    # rather than trivially satisfying it via sequential test code.
    await asyncio.gather(*(store.append(e) for e in events))
    await store.close()

    written_ids = [
        row[0] for call in fake_asyncpg._conn.executemany.call_args_list for row in call.args[1]
    ]
    assert written_ids == [e.id for e in events]


@pytest.mark.asyncio
async def test_drain_batches_do_not_exceed_configured_batch_size(fake_asyncpg: ModuleType) -> None:
    """A single executemany() call must never exceed _DRAIN_BATCH_SIZE —
    confirms batching actually bounds each write instead of accumulating
    an unbounded batch when events arrive faster than the drain interval."""
    from nullain.events.postgres_store import (
        _DRAIN_BATCH_SIZE,  # type: ignore[reportPrivateUsage]
    )

    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()

    events = [
        UserMessageEvent(session_id="s1", content=f"msg-{i}") for i in range(_DRAIN_BATCH_SIZE + 20)
    ]
    await asyncio.gather(*(store.append(e) for e in events))
    await store.close()

    for call in fake_asyncpg._conn.executemany.call_args_list:
        _, rows = call.args
        assert len(rows) <= _DRAIN_BATCH_SIZE


@pytest.mark.asyncio
async def test_get_latest_session_id_maps_row_to_string(fake_asyncpg: ModuleType) -> None:
    fake_asyncpg._conn.fetchrow.return_value = {"session_id": "s-latest"}
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    assert await store.get_latest_session_id() == "s-latest"


@pytest.mark.asyncio
async def test_get_latest_session_id_empty_store_is_none(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    assert await store.get_latest_session_id() is None


@pytest.mark.asyncio
async def test_get_latest_session_id_wraps_failure(fake_asyncpg: ModuleType) -> None:
    fake_asyncpg._conn.fetchrow.side_effect = RuntimeError("timeout")
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    with pytest.raises(EventStoreConnectionError, match="Failed to query latest session"):
        await store.get_latest_session_id()


@pytest.mark.asyncio
async def test_list_session_ids_maps_rows_to_strings(fake_asyncpg: ModuleType) -> None:
    fake_asyncpg._conn.fetch.return_value = [{"session_id": "s1"}, {"session_id": "s2"}]
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    assert await store.list_session_ids() == ["s1", "s2"]


@pytest.mark.asyncio
async def test_list_session_ids_wraps_failure(fake_asyncpg: ModuleType) -> None:
    fake_asyncpg._conn.fetch.side_effect = RuntimeError("timeout")
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    with pytest.raises(EventStoreConnectionError, match="Failed to list sessions"):
        await store.list_session_ids()


@pytest.mark.asyncio
async def test_get_session_events_deserializes_rows(fake_asyncpg: ModuleType) -> None:
    event = UserMessageEvent(session_id="s1", content="hi there")
    fake_asyncpg._conn.fetch.return_value = [
        {"event_type": event.event_type, "payload": event.model_dump_json()}
    ]
    store = PostgresEventStore(dsn="postgresql://localhost/test")

    events = await store.get_session_events("s1")

    assert len(events) == 1
    assert isinstance(events[0], UserMessageEvent)
    assert events[0].content == "hi there"


@pytest.mark.asyncio
async def test_get_session_events_empty_for_unknown_session(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    assert await store.get_session_events("no-such-session") == []


@pytest.mark.asyncio
async def test_get_session_events_wraps_failure(fake_asyncpg: ModuleType) -> None:
    fake_asyncpg._conn.fetch.side_effect = RuntimeError("timeout")
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    with pytest.raises(EventStoreConnectionError, match="Failed to fetch session events"):
        await store.get_session_events("s1")


@pytest.mark.asyncio
async def test_close_releases_pool(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()
    await store.close()
    fake_asyncpg._pool.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_before_initialize_is_a_noop(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.close()  # must not raise
    fake_asyncpg._pool.close.assert_not_called()


@pytest.mark.asyncio
async def test_close_is_idempotent(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()
    await store.close()
    await store.close()  # must not raise
