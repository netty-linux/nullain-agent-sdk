"""Unit tests for PostgresEventStore — mocked asyncpg (no live Postgres/
Supabase instance in CI; the adapter's own logic under test is the SQL it
issues, auto-initialize-on-first-use, and error wrapping, not asyncpg's or
Postgres's own behavior)."""

from __future__ import annotations

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
async def test_append_auto_initializes_and_inserts(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    event = UserMessageEvent(session_id="s1", content="hello")

    await store.append(event)

    fake_asyncpg.create_pool.assert_called_once()  # auto-initialized
    insert_calls = [
        c for c in fake_asyncpg._conn.execute.call_args_list if "INSERT INTO events" in c.args[0]
    ]
    assert len(insert_calls) == 1
    args = insert_calls[0].args
    assert args[1] == event.id
    assert args[2] == "s1"


@pytest.mark.asyncio
async def test_append_wraps_failure_in_connection_error(fake_asyncpg: ModuleType) -> None:
    store = PostgresEventStore(dsn="postgresql://localhost/test")
    await store.initialize()  # succeed first, so only the append's own INSERT fails
    fake_asyncpg._conn.execute.side_effect = RuntimeError("timeout")
    with pytest.raises(EventStoreConnectionError, match="Failed to append"):
        await store.append(UserMessageEvent(session_id="s1", content="x"))


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
