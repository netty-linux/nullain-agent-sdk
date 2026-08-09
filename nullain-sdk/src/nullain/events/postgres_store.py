"""Nullain Agent SDK — Postgres/Supabase Event Store Adapter.

`PostgresEventStore` is the production `EventStorePort` adapter
(docs/FUSION_PLAN.md ADR-2): same append-only `events` table shape as
`SQLiteEventStore`, same `seq`-ordered resume semantics, backed by
Postgres (Supabase or any other Postgres instance) instead of a local
SQLite file — for a deployment where the agent process is one of several
replicas that all need to see the same session history, not just the
process that happens to be holding the local `.nullain/sessions.db` file.

Chosen over dual-write/mirroring to SQLite (ADR-2's "opção a", rejected):
two writers means two sources of truth and a real failure mode (local
write succeeds, remote write fails — replay now diverges from what
actually happened). `PostgresEventStore` is a drop-in replacement, not an
addition — a deployment picks exactly one `EventStorePort` adapter via
`nullain.toml`'s config for its whole lifetime, matching how
`EmbeddingProvider`/`VectorStore` are chosen in `nullain.rag`.

`asyncpg` is an optional dependency (`pip install nullain-sdk[postgres]`),
resolved lazily via `importlib.import_module` — same pattern as
`nullain.plugins.signing`'s optional `cryptography` import and
`nullain.rag`'s `fastembed`/`qdrant-client` imports, keeping pyright's
strict mode clean without the dependency installed.

`append()` enqueues onto an in-process `asyncio.Queue` and returns
immediately; a single background drain task writes queued events to
Postgres in batches via `executemany`. `AgentLoop._emit()` previously
awaited one `pool.acquire()` + one INSERT round-trip per event, inline
in the agent's critical path — against a pooled connection (Supabase's
documented deployment target) that's plausibly 20-100ms per event,
several times per step. Queueing removes that latency from the hot path
entirely.

This changes what a successful `append()` return means: "accepted for
write," not "durably written" — a real semantic relaxation from the
`EventStorePort` contract's prior implication (see `port.py`'s
docstring). `append()` still raises `EventStoreConnectionError`, but
only for enqueue-time failures (store closed, queue full past the
configured timeout); a Postgres-side failure during the background
drain is logged, never raised to a caller that already returned.
`flush()` is the escape hatch for a caller that needs "durably written"
back — call it, then trust the return.

The single-writer-task design is load-bearing, not incidental: `seq` is
`BIGSERIAL`, assigned at INSERT time, and every read path
(`get_session_events`, `list_session_ids`, `get_latest_session_id`)
depends on `ORDER BY seq ASC` reflecting true insertion order for
resume/replay to be correct. A concurrent multi-connection drain would
let two batches race to acquire `seq` values in an order that doesn't
match when their events were actually appended, silently corrupting
that invariant. `pool_max_size` stays available for read-path
concurrency; the write path never uses more than one connection at a
time by construction.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
from typing import Any

from nullain.errors import NullainError
from nullain.events.store import EVENT_CLASS_MAP
from nullain.events.types import BaseEvent
from nullain.telemetry import get_logger

logger = get_logger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    seq BIGSERIAL PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL
)
"""
_CREATE_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_session_id ON events (session_id)"

_INSERT_SQL = (
    "INSERT INTO events (id, session_id, timestamp, event_type, payload) "
    "VALUES ($1, $2, $3, $4, $5)"
)

#: Maximum events buffered before `append()` blocks the caller. Bounded
#: (not unbounded) so a sustained Postgres outage can't grow unbounded
#: process memory; blocking (not dropping) because a dropped event is a
#: silent gap in an event-sourced trajectory — the caller waiting briefly
#: is the correct backpressure response, not data loss.
_DEFAULT_QUEUE_MAXSIZE = 1000

#: How many queued events one drain iteration writes via `executemany`.
_DRAIN_BATCH_SIZE = 100

#: How long the drain loop waits for more events to accumulate into a
#: batch before writing whatever it already has — bounds worst-case
#: append-to-durable latency during low-traffic periods.
_DRAIN_INTERVAL_SECONDS = 0.5


class EventStoreConnectionError(NullainError):
    """Raised when PostgresEventStore cannot reach or query the database,
    or when `append()` cannot even enqueue an event (store closed, queue
    full). Never raised for a background drain failure after `append()`
    has already returned — see the module docstring's note on `append()`'s
    relaxed "accepted," not "written," return semantics."""


class PostgresEventStore:
    """Postgres-backed `EventStorePort` adapter — Supabase or any Postgres.

    Args:
        dsn: A `postgresql://` connection string (Supabase's pooled
            connection string works directly — see docs/FUSION_PLAN.md's
            Supabase schema section for the surrounding `sessions`/`users`/
            `metadata`/`traces` tables this `events` table lives alongside;
            those are relational-side concerns this adapter does not own).
        pool_min_size: Minimum connections `asyncpg` keeps open.
        pool_max_size: Maximum connections `asyncpg` may open. The
            background drain task only ever uses one connection at a time
            (see module docstring on why); the rest of this pool remains
            available for concurrent reads.
        queue_maxsize: Maximum events buffered in `append()`'s queue
            before it blocks the caller. See `_DEFAULT_QUEUE_MAXSIZE`.

    Raises:
        ImportError: `asyncpg` is not installed
            (`pip install nullain-sdk[postgres]`).
    """

    def __init__(
        self,
        dsn: str,
        *,
        pool_min_size: int = 1,
        pool_max_size: int = 5,
        queue_maxsize: int = _DEFAULT_QUEUE_MAXSIZE,
    ) -> None:
        # importlib (not a static import) so the optional `asyncpg` extra
        # is resolved at runtime and does not trip static analysis when
        # absent — same pattern as nullain.plugins.signing's optional
        # `cryptography` import and nullain.rag's fastembed/qdrant-client.
        try:
            self._asyncpg: Any = importlib.import_module("asyncpg")
        except ImportError as exc:
            raise ImportError(
                "PostgresEventStore requires the 'postgres' extra: "
                "pip install nullain-sdk[postgres]"
            ) from exc

        self._dsn = dsn
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool: Any = None
        self._queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self._drain_task: asyncio.Task[None] | None = None
        self._closed = False

    async def initialize(self) -> None:
        """Create the connection pool and schema, and start the background
        drain task. Idempotent."""
        if self._pool is not None:
            return
        try:
            self._pool = await self._asyncpg.create_pool(
                dsn=self._dsn,
                min_size=self._pool_min_size,
                max_size=self._pool_max_size,
            )
            async with self._pool.acquire() as conn:
                await conn.execute(_CREATE_TABLE_SQL)
                await conn.execute(_CREATE_INDEX_SQL)
        except Exception as exc:
            raise EventStoreConnectionError(
                f"Failed to initialize Postgres event store: {exc}"
            ) from exc
        self._drain_task = asyncio.create_task(self._drain_loop())

    async def _ensure_pool(self) -> Any:
        """Return the connection pool, initializing it first if needed.
        Centralizes the "auto-initialize on first use" behavior every
        method needs, and gives pyright one place to see `self._pool` is
        non-None afterward instead of re-deriving it at every call site."""
        if self._pool is None:
            await self.initialize()
        assert self._pool is not None  # initialize() always sets it or raises
        return self._pool

    async def _drain_loop(self) -> None:
        """Single sequential writer: pulls queued events in batches and
        writes them via `executemany`. Runs until cancelled by `close()`.
        A batch write failure is logged, never raised — there is no
        caller left to raise to, since `append()` already returned when
        the event was enqueued. This is the intentional trade this
        adapter makes for taking Postgres latency out of the agent loop's
        hot path; a caller needing a hard durability guarantee back
        should call `flush()` and treat that as the synchronization
        point, not assume every `append()` implies a completed write."""
        assert self._pool is not None
        while True:
            try:
                batch = [await self._queue.get()]
            except asyncio.CancelledError:
                break
            deadline = asyncio.get_event_loop().time() + _DRAIN_INTERVAL_SECONDS
            while len(batch) < _DRAIN_BATCH_SIZE:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except TimeoutError:
                    break
            await self._write_batch(batch)
            for _ in batch:
                self._queue.task_done()

    async def _write_batch(self, batch: list[BaseEvent]) -> None:
        rows = [
            (event.id, event.session_id, event.timestamp, event.event_type, event.model_dump_json())
            for event in batch
        ]
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(_INSERT_SQL, rows)
        except Exception as exc:
            logger.error(
                "postgres_event_store_drain_failed",
                error=str(exc),
                batch_size=len(batch),
                session_ids=sorted({e.session_id for e in batch}),
            )

    async def flush(self) -> None:
        """Wait until every currently-queued event has been written (or
        its write attempted and logged on failure). Call this when a
        caller needs "durably written," not just "accepted," back —
        `append()` alone no longer implies that."""
        await self._queue.join()

    async def close(self) -> None:
        """Flush remaining queued events, stop the drain task, and close
        the connection pool. Idempotent — the pool being None (never
        initialized, or already closed) makes this a no-op."""
        if self._closed:
            return
        self._closed = True
        if self._drain_task is not None:
            await self.flush()
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            self._drain_task = None
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def append(self, event: BaseEvent) -> None:
        """Enqueue one event for background persistence. Auto-initializes
        (and starts the drain task) if not already initialized.

        Returns once the event is accepted onto the queue — NOT once it
        is durably written to Postgres. See the module and class
        docstrings for why, and `flush()` for how to wait for the
        stronger guarantee when a caller genuinely needs it."""
        if self._closed:
            raise EventStoreConnectionError("Cannot append: event store is closed.")
        await self._ensure_pool()
        try:
            await self._queue.put(event)
        except Exception as exc:
            raise EventStoreConnectionError(f"Failed to enqueue event: {exc}") from exc

    async def get_latest_session_id(self) -> str | None:
        """Return the session_id of the most recently appended event.

        Flushes the write queue first — a caller needs this to reflect
        every `append()` that happened-before this call, not whatever
        the background drain has gotten around to writing yet."""
        pool = await self._ensure_pool()
        await self.flush()

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT session_id FROM events ORDER BY seq DESC LIMIT 1")
        except Exception as exc:
            raise EventStoreConnectionError(
                f"Failed to query latest session from Postgres: {exc}"
            ) from exc
        return row["session_id"] if row is not None else None

    async def list_session_ids(self) -> list[str]:
        """Return every distinct session id, oldest-created first.

        Flushes the write queue first — see `get_latest_session_id`."""
        pool = await self._ensure_pool()
        await self.flush()

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT session_id FROM events GROUP BY session_id ORDER BY MIN(seq) ASC"
                )
        except Exception as exc:
            raise EventStoreConnectionError(
                f"Failed to list sessions from Postgres: {exc}"
            ) from exc
        return [row["session_id"] for row in rows]

    async def get_session_events(self, session_id: str) -> list[BaseEvent]:
        """Fetch all events for a session, in insertion (`seq`) order.

        Flushes the write queue first — see `get_latest_session_id`. This
        matters most here: `Agent._load_session_history` calls this to
        resume a session, and a resume that silently missed the last few
        appended events (still sitting in the queue) would replay a
        truncated, wrong trajectory."""
        pool = await self._ensure_pool()
        await self.flush()

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT event_type, payload FROM events WHERE session_id = $1 ORDER BY seq ASC",
                    session_id,
                )
        except Exception as exc:
            raise EventStoreConnectionError(
                f"Failed to fetch session events from Postgres: {exc}"
            ) from exc

        events: list[BaseEvent] = []
        for row in rows:
            cls = EVENT_CLASS_MAP.get(row["event_type"], BaseEvent)
            events.append(cls.model_validate_json(row["payload"]))
        return events


__all__ = ["EventStoreConnectionError", "PostgresEventStore"]
