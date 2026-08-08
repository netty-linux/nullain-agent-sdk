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
"""

from __future__ import annotations

import importlib
from typing import Any

from nullain.errors import NullainError
from nullain.events.store import EVENT_CLASS_MAP
from nullain.events.types import BaseEvent

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


class EventStoreConnectionError(NullainError):
    """Raised when PostgresEventStore cannot reach or query the database."""


class PostgresEventStore:
    """Postgres-backed `EventStorePort` adapter — Supabase or any Postgres.

    Args:
        dsn: A `postgresql://` connection string (Supabase's pooled
            connection string works directly — see docs/FUSION_PLAN.md's
            Supabase schema section for the surrounding `sessions`/`users`/
            `metadata`/`traces` tables this `events` table lives alongside;
            those are relational-side concerns this adapter does not own).
        pool_min_size: Minimum connections `asyncpg` keeps open.
        pool_max_size: Maximum connections `asyncpg` may open.

    Raises:
        ImportError: `asyncpg` is not installed
            (`pip install nullain-sdk[postgres]`).
    """

    def __init__(self, dsn: str, *, pool_min_size: int = 1, pool_max_size: int = 5) -> None:
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

    async def initialize(self) -> None:
        """Create the connection pool and schema. Idempotent."""
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

    async def _ensure_pool(self) -> Any:
        """Return the connection pool, initializing it first if needed.
        Centralizes the "auto-initialize on first use" behavior every
        method needs, and gives pyright one place to see `self._pool` is
        non-None afterward instead of re-deriving it at every call site."""
        if self._pool is None:
            await self.initialize()
        assert self._pool is not None  # initialize() always sets it or raises
        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def append(self, event: BaseEvent) -> None:
        """Persist one event. Auto-initializes if not already initialized."""
        pool = await self._ensure_pool()

        payload = event.model_dump_json()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO events (id, session_id, timestamp, event_type, payload) "
                    "VALUES ($1, $2, $3, $4, $5)",
                    event.id,
                    event.session_id,
                    event.timestamp,
                    event.event_type,
                    payload,
                )
        except Exception as exc:
            raise EventStoreConnectionError(f"Failed to append event to Postgres: {exc}") from exc

    async def get_latest_session_id(self) -> str | None:
        """Return the session_id of the most recently appended event."""
        pool = await self._ensure_pool()

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT session_id FROM events ORDER BY seq DESC LIMIT 1")
        except Exception as exc:
            raise EventStoreConnectionError(
                f"Failed to query latest session from Postgres: {exc}"
            ) from exc
        return row["session_id"] if row is not None else None

    async def list_session_ids(self) -> list[str]:
        """Return every distinct session id, oldest-created first."""
        pool = await self._ensure_pool()

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
        """Fetch all events for a session, in insertion (`seq`) order."""
        pool = await self._ensure_pool()

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
