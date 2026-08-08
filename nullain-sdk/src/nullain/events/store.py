"""Nullain Agent SDK — SQLite Async Event Store for Local Persistence."""

from pathlib import Path

import aiosqlite

from nullain.events.types import (
    BaseEvent,
    CompactionEvent,
    ErrorEvent,
    ModelResponseEvent,
    SessionRepairedEvent,
    SpecCreatedEvent,
    SpecVerifiedEvent,
    StreamDeltaEvent,
    TodoEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)

# NOTE: StreamDeltaEvent is intentionally included for round-trip fidelity, but
# streaming deltas are high-volume and typically not worth persisting. Callers
# that want to exclude them can filter before EventStore.append.
EVENT_CLASS_MAP: dict[str, type[BaseEvent]] = {
    "user_message": UserMessageEvent,
    "model_response": ModelResponseEvent,
    "tool_call": ToolCallEvent,
    "tool_result": ToolResultEvent,
    "compaction": CompactionEvent,
    "error": ErrorEvent,
    "spec_created": SpecCreatedEvent,
    "spec_verified": SpecVerifiedEvent,
    "session_repaired": SessionRepairedEvent,
    "stream_delta": StreamDeltaEvent,
    "todo": TodoEvent,
}


class EventStore:
    """Async SQLite persistence engine for conversation trajectory events."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database connection and schema.

        ``seq`` is an autoincrementing insertion counter and is the authoritative
        ordering key. ``timestamp`` alone is not: ``time.time()`` has ~15.6 ms
        resolution on Windows, so consecutively appended events routinely share
        one timestamp, and any tiebreaker on ``id`` (a random UUID) would order
        them arbitrarily — silently corrupting the trajectory the agent replays.

        For a file-backed store (anything other than ``":memory:"``), the
        parent directory is created if missing — aiosqlite does not do this
        itself, and callers (e.g. the ``Agent`` facade's default
        ``<workspace>/.nullain/sessions.db``) should not need to create it
        ahead of time.
        """
        if self._conn is None:
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(self.db_path)
            await self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_id ON events (session_id)"
            )
            await self._conn.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def append(self, event: BaseEvent) -> None:
        """Persist an event to SQLite."""
        if self._conn is None:
            await self.initialize()

        if self._conn is None:
            raise RuntimeError("Failed to initialize EventStore database connection")

        payload = event.model_dump_json()
        query = (
            "INSERT INTO events (id, session_id, timestamp, event_type, payload) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        await self._conn.execute(
            query,
            (event.id, event.session_id, event.timestamp, event.event_type, payload),
        )
        await self._conn.commit()

    async def get_latest_session_id(self) -> str | None:
        """Return the session_id of the most recently appended event, if any.

        Used to resume "the last conversation" (e.g. a CLI ``--continue``
        flag) without the caller having to track session ids itself. Latest
        is defined by insertion order (``seq``), consistent with
        :meth:`get_session_events`'s ordering — not by timestamp, for the
        same coarse-clock reason documented on :meth:`initialize`.
        """
        if self._conn is None:
            await self.initialize()

        if self._conn is None:
            raise RuntimeError("Failed to initialize EventStore database connection")

        query = "SELECT session_id FROM events ORDER BY seq DESC LIMIT 1"
        async with self._conn.execute(query) as cursor:
            row = await cursor.fetchone()
        return row[0] if row is not None else None

    async def list_session_ids(self) -> list[str]:
        """Return every distinct session id in the store, oldest first.

        Ordered by each session's earliest ``seq`` (not alphabetically by id,
        which would be meaningless for random UUIDs) so callers presenting a
        list — e.g. ``nullain doctor``'s session-integrity check — see
        sessions in the order they were created.
        """
        if self._conn is None:
            await self.initialize()

        if self._conn is None:
            raise RuntimeError("Failed to initialize EventStore database connection")

        query = "SELECT session_id FROM events GROUP BY session_id ORDER BY MIN(seq) ASC"
        async with self._conn.execute(query) as cursor:
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_session_events(self, session_id: str) -> list[BaseEvent]:
        """Fetch all events for a session in insertion order.

        Ordering is by ``seq`` (the insertion counter), not by ``timestamp``:
        the clock is too coarse on some platforms to separate consecutive
        appends, and event sourcing requires the exact order events were
        recorded in.
        """
        if self._conn is None:
            await self.initialize()

        if self._conn is None:
            raise RuntimeError("Failed to initialize EventStore database connection")

        query = "SELECT event_type, payload FROM events WHERE session_id = ? ORDER BY seq ASC"
        async with self._conn.execute(query, (session_id,)) as cursor:
            rows = await cursor.fetchall()

        events: list[BaseEvent] = []
        for event_type, payload in rows:
            cls = EVENT_CLASS_MAP.get(event_type, BaseEvent)
            event_obj = cls.model_validate_json(payload)
            events.append(event_obj)
        return events


__all__ = ["EVENT_CLASS_MAP", "EventStore"]
