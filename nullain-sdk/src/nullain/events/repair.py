"""Nullain Agent SDK — repair pass for sessions corrupted by the pre-#24
compaction bug.

Before #24, ``ContextManager._collect_compacted`` used a naive positional
slice (``events[:-_RECENT_KEEP]``) that could compact away a
``ModelResponseEvent`` carrying ``tool_calls`` while the matching
``ToolResultEvent``(s) survived in the kept tail. ``Conversation.fold`` then
replays a `tool`-role message with a ``tool_call_id`` that has no preceding
`assistant` message declaring that call — invalid per the OpenAI
chat-completions schema, and the exact shape Ollama Cloud's compat shim
rejects with the opaque ``400 invalid message content type: <nil>``.

#24 stopped *new* corruption at the source; it explicitly left already-
persisted sessions broken, since the corruption is baked into the stored
``CompactionEvent.compacted_event_ids``. This module repairs those sessions
on load.

Repair strategy, in order of preference:
  (a) Re-pair — the offending ``ModelResponseEvent`` is still physically
      present in the stored event log (compaction never deletes rows, it
      only adds a ``CompactionEvent`` whose ``compacted_event_ids`` folds
      other events away). Shrinking that id back out of
      ``compacted_event_ids`` un-compacts the origin message, restoring the
      pairing exactly as #24's ``_compaction_boundary`` would have kept it.
  (b) Drop — if the origin event is missing from the log entirely (should
      not happen, but the repair must not assume the happy path), the
      orphaned tool result itself is excluded instead. The task it reported
      on is unrecoverable either way; dropping the dangling result is the
      only option that yields a valid message history.

Both strategies are auditable: the caller receives a ``SessionRepairedEvent``
describing exactly what was changed, alongside the repaired event list.
Uncorrupted sessions take a fast no-op path — the input list's identity is
returned unchanged (not merely equal) so a repair pass over a healthy session
does zero allocation beyond the detection scan.
"""

from __future__ import annotations

from collections.abc import Sequence

from nullain.events.types import (
    BaseEvent,
    CompactionEvent,
    ModelResponseEvent,
    SessionRepairedEvent,
    ToolResultEvent,
)


def _call_origins(events: Sequence[BaseEvent]) -> dict[str, str]:
    """Map every ``tool_calls[].id`` to the id of the ``ModelResponseEvent``
    that issued it, across the *entire* stored log (compacted or not)."""
    origins: dict[str, str] = {}
    for ev in events:
        if isinstance(ev, ModelResponseEvent) and ev.tool_calls:
            for tc in ev.tool_calls:
                origins[tc.id] = ev.id
    return origins


def find_orphaned_tool_results(events: Sequence[BaseEvent]) -> list[ToolResultEvent]:
    """Return every ``ToolResultEvent`` that would fold into an orphaned
    `tool`-role message: its ``call_id`` has no preceding, non-compacted
    ``ModelResponseEvent`` declaring that call.

    Mirrors ``Conversation.fold``'s own compaction bookkeeping exactly (same
    "last ``CompactionEvent`` wins for ``compacted_event_ids``" semantics) so
    detection agrees with what the real fold would produce.
    """
    compacted_ids: set[str] = set()
    for ev in events:
        if isinstance(ev, CompactionEvent):
            compacted_ids.update(ev.compacted_event_ids)

    origins = _call_origins(events)
    declared_calls: set[str] = set()
    orphans: list[ToolResultEvent] = []

    for ev in events:
        if ev.id in compacted_ids:
            continue
        if isinstance(ev, ModelResponseEvent) and ev.tool_calls:
            declared_calls.update(tc.id for tc in ev.tool_calls)
        elif isinstance(ev, ToolResultEvent) and ev.call_id not in declared_calls:
            # Not visible yet in the surviving history. Only a real orphan
            # if there's no way it *could* become visible — i.e. its origin
            # ModelResponseEvent doesn't exist at all, or exists but was
            # compacted away. A call_id with no origin anywhere is equally
            # unrecoverable and treated the same way by the caller (dropped,
            # not re-paired).
            origin_id = origins.get(ev.call_id)
            if origin_id is None or origin_id in compacted_ids:
                orphans.append(ev)
    return orphans


def repair_session_events(
    session_id: str, events: list[BaseEvent]
) -> tuple[list[BaseEvent], SessionRepairedEvent | None]:
    """Repair ``events`` so folding it never produces an orphaned tool
    message, returning ``(repaired_events, report)``.

    ``report`` is ``None`` on the no-op fast path (nothing was wrong), in
    which case the returned list is ``events`` itself, unchanged — callers
    can use identity (``is``) to skip re-persisting or re-logging. When a
    repair did happen, ``report`` is a ``SessionRepairedEvent`` describing
    what was re-paired vs. dropped, and the returned list is a new list
    (the input is never mutated in place, matching every other event type
    here being frozen/immutable).
    """
    orphans = find_orphaned_tool_results(events)
    if not orphans:
        return events, None

    origins = _call_origins(events)
    re_paired: list[str] = []
    dropped: list[str] = []

    # Which origin ids need to be un-compacted (strategy a), and which
    # orphaned ToolResultEvent ids have no recoverable origin and must be
    # dropped outright (strategy b).
    origins_to_restore: set[str] = set()
    drop_event_ids: set[str] = set()
    for orphan in orphans:
        origin_id = origins.get(orphan.call_id)
        if origin_id is not None:
            origins_to_restore.add(origin_id)
            re_paired.append(orphan.call_id)
        else:
            drop_event_ids.add(orphan.id)
            dropped.append(orphan.call_id)

    repaired: list[BaseEvent] = []
    for ev in events:
        if ev.id in drop_event_ids:
            continue
        if isinstance(ev, CompactionEvent) and origins_to_restore & set(ev.compacted_event_ids):
            new_compacted = tuple(
                eid for eid in ev.compacted_event_ids if eid not in origins_to_restore
            )
            ev = ev.model_copy(update={"compacted_event_ids": new_compacted})
        repaired.append(ev)

    report = SessionRepairedEvent(
        session_id=session_id,
        re_paired_call_ids=tuple(re_paired),
        dropped_call_ids=tuple(dropped),
    )
    return repaired, report


__all__ = ["find_orphaned_tool_results", "repair_session_events"]
