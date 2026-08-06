"""Nullain Agent SDK — SQLite Async Event Store for Local Persistence."""

from pathlib import Path

import aiosqlite

from nullain.events.types import (
    BaseEvent,
    CompactionEvent,
    ErrorEvent,
    ModelResponseEvent,
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
    "stream_delta": StreamDeltaEvent,
    "todo": TodoEvent,
}


class EventStore:
    """Async SQLite persistence engine for conversation trajectory events."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database connection and schema."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path)
            await self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
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

    async def get_session_events(self, session_id: str) -> list[BaseEvent]:
        """Fetch all events for a given session sorted by timestamp."""
        if self._conn is None:
            await self.initialize()

        if self._conn is None:
            raise RuntimeError("Failed to initialize EventStore database connection")

        query = (
            "SELECT event_type, payload FROM events WHERE session_id = ? "
            "ORDER BY timestamp ASC, id ASC"
        )
        async with self._conn.execute(query, (session_id,)) as cursor:
            rows = await cursor.fetchall()

        events: list[BaseEvent] = []
        for event_type, payload in rows:
            cls = EVENT_CLASS_MAP.get(event_type, BaseEvent)
            event_obj = cls.model_validate_json(payload)
            events.append(event_obj)
        return events


__all__ = ["EVENT_CLASS_MAP", "EventStore"]
