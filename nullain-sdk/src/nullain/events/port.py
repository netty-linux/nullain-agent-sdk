"""Nullain Agent SDK — Event Store Port (hexagonal boundary).

`EventStorePort` is the persistence contract `Agent`/`AgentLoop` depend on
for conversation trajectory storage — mirrors `LLMProvider`/
`EmbeddingProvider`/`VectorStore`'s shape elsewhere in the SDK. Two
adapters implement it: `SQLiteEventStore` (the SDK's standalone/dev
default) and `PostgresEventStore` (docs/FUSION_PLAN.md ADR-2, for
Supabase-backed production deployments).

Both adapters must give identical resume/repair semantics — a session
persisted under one backend and its history read back must look the same
to `Agent._load_session_history` regardless of which store wrote it.
Repair (`nullain.events.repair.repair_session_events`) is applied by the
*caller* (the `Agent` facade), not by the store — so this is a property
each adapter gets for free by returning events in `seq` order, not
something either adapter implements itself.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nullain.events.types import BaseEvent


@runtime_checkable
class EventStorePort(Protocol):
    """Abstract protocol for conversation-trajectory persistence."""

    async def initialize(self) -> None:
        """Prepare the backing store (create schema/connection). Idempotent
        — safe to call more than once; later calls are no-ops."""
        ...

    async def close(self) -> None:
        """Release the backing connection. Idempotent — safe to call on an
        already-closed or never-opened store."""
        ...

    async def append(self, event: BaseEvent) -> None:
        """Persist one event. Auto-initializes if not already initialized."""
        ...

    async def get_session_events(self, session_id: str) -> list[BaseEvent]:
        """Fetch all events for a session, in insertion order (the
        authoritative order for event-sourcing replay — not by timestamp,
        which can be too coarse-grained to distinguish consecutive
        appends on some platforms)."""
        ...

    async def get_latest_session_id(self) -> str | None:
        """Return the most recently appended-to session's id, or None if
        the store is empty. "Most recent" is defined by insertion order,
        consistent with `get_session_events`."""
        ...

    async def list_session_ids(self) -> list[str]:
        """Return every distinct session id, ordered oldest-created first."""
        ...


__all__ = ["EventStorePort"]
