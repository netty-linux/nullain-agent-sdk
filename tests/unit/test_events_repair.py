"""Unit tests for nullain.events.repair — the pre-#24 orphaned-tool-result
session repair pass (issue #44).

Covers: detection, both repair strategies (re-pair vs. drop), the no-op fast
path for healthy sessions, a full fixture-DB integration test replaying a
real corrupted session through EventStore + repair + Conversation.fold end to
end, and a Hypothesis property test matching the repo's existing event-fold
property tests (test_events_property.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from nullain.events import (
    BaseEvent,
    CompactionEvent,
    Conversation,
    EventStore,
    ModelResponseEvent,
    SessionRepairedEvent,
    ToolResultEvent,
    UserMessageEvent,
    find_orphaned_tool_results,
    repair_session_events,
)
from nullain.llm import ToolCall


def _corrupted_events() -> list[BaseEvent]:
    """A minimal event log reproducing the exact pre-#24 corruption shape:
    a CompactionEvent whose compacted_event_ids includes the
    ModelResponseEvent that declared a tool call, but not the
    ToolResultEvent answering it — the naive events[:-_RECENT_KEEP] slice
    that #24 replaced could produce exactly this split.
    """
    return [
        UserMessageEvent(session_id="s1", id="u0", content="old prompt"),
        ModelResponseEvent(
            session_id="s1",
            id="m1",
            model="m",
            content=None,
            tool_calls=(ToolCall(id="call_1", name="write_file", arguments={}),),
        ),
        ToolResultEvent(
            session_id="s1", id="t1", call_id="call_1", tool_name="write_file", output="ok1"
        ),
        UserMessageEvent(session_id="s1", id="u1", content="verify fix"),
        # The bug: m1 is compacted away but t1 (its result) is not.
        CompactionEvent(
            session_id="s1",
            id="c1",
            summary="old prompt happened",
            compacted_event_ids=("u0", "m1"),
        ),
        UserMessageEvent(session_id="s1", id="u2", content="continue"),
    ]


class TestFindOrphanedToolResults:
    def test_detects_the_orphan(self) -> None:
        orphans = find_orphaned_tool_results(_corrupted_events())
        assert [ev.id for ev in orphans] == ["t1"]

    def test_healthy_session_has_no_orphans(self) -> None:
        events: list[BaseEvent] = [
            UserMessageEvent(session_id="s1", id="u0", content="prompt"),
            ModelResponseEvent(
                session_id="s1",
                id="m1",
                model="m",
                content=None,
                tool_calls=(ToolCall(id="call_1", name="write_file", arguments={}),),
            ),
            ToolResultEvent(
                session_id="s1", id="t1", call_id="call_1", tool_name="write_file", output="ok"
            ),
        ]
        assert find_orphaned_tool_results(events) == []

    def test_post_24_compaction_that_keeps_pairs_together_has_no_orphans(self) -> None:
        """A CompactionEvent that correctly keeps a tool-call turn and its
        result together (what #24's _compaction_boundary produces) must
        never be flagged — only a split pairing is corruption."""
        events: list[BaseEvent] = [
            UserMessageEvent(session_id="s1", id="u0", content="prompt"),
            ModelResponseEvent(
                session_id="s1",
                id="m1",
                model="m",
                content=None,
                tool_calls=(ToolCall(id="call_1", name="write_file", arguments={}),),
            ),
            ToolResultEvent(
                session_id="s1", id="t1", call_id="call_1", tool_name="write_file", output="ok"
            ),
            # Compacts BOTH m1 and t1 together — correct, not corruption.
            CompactionEvent(
                session_id="s1", id="c1", summary="done", compacted_event_ids=("u0", "m1", "t1")
            ),
            UserMessageEvent(session_id="s1", id="u1", content="next"),
        ]
        assert find_orphaned_tool_results(events) == []

    def test_missing_origin_entirely_is_still_an_orphan(self) -> None:
        """A tool result whose call_id has no ModelResponseEvent anywhere in
        the log at all (not even a compacted one) — the unrecoverable case
        that strategy (b) must catch."""
        events: list[BaseEvent] = [
            ToolResultEvent(
                session_id="s1", id="t1", call_id="ghost_call", tool_name="x", output="?"
            ),
        ]
        assert [ev.id for ev in find_orphaned_tool_results(events)] == ["t1"]


class TestRepairSessionEvents:
    def test_noop_fast_path_returns_same_list_identity(self) -> None:
        events: list[BaseEvent] = [UserMessageEvent(session_id="s1", content="hi")]
        repaired, report = repair_session_events("s1", events)
        assert repaired is events
        assert report is None

    def test_repairs_by_re_pairing_the_origin_message(self) -> None:
        repaired, report = repair_session_events("s1", _corrupted_events())
        assert report is not None
        assert report.re_paired_call_ids == ("call_1",)
        assert report.dropped_call_ids == ()

        # m1 must now be visible again (un-compacted) and t1 must still be
        # present — the pairing is restored, not destroyed.
        repaired_ids = {ev.id for ev in repaired}
        assert "m1" in repaired_ids
        assert "t1" in repaired_ids

        compaction = next(ev for ev in repaired if isinstance(ev, CompactionEvent))
        assert "m1" not in compaction.compacted_event_ids
        assert "u0" in compaction.compacted_event_ids  # unrelated compacted id untouched

    def test_repaired_history_folds_without_an_orphaned_tool_message(self) -> None:
        """End-to-end: after repair, Conversation.fold must never produce a
        `tool`-role message whose tool_call_id has no preceding assistant
        tool_calls entry — the exact shape that broke Ollama Cloud."""
        repaired, _ = repair_session_events("s1", _corrupted_events())
        state = Conversation.fold("s1", repaired)

        known_call_ids: set[str] = set()
        for msg in state.messages:
            d = msg.to_api_dict()
            if d["role"] == "tool":
                assert d["tool_call_id"] in known_call_ids
            for tc in msg.tool_calls or []:
                known_call_ids.add(tc.id)

    def test_drops_unrecoverable_orphan_with_no_origin(self) -> None:
        events: list[BaseEvent] = [
            UserMessageEvent(session_id="s1", id="u0", content="hi"),
            ToolResultEvent(
                session_id="s1", id="t1", call_id="ghost_call", tool_name="x", output="?"
            ),
        ]
        repaired, report = repair_session_events("s1", events)
        assert report is not None
        assert report.dropped_call_ids == ("ghost_call",)
        assert report.re_paired_call_ids == ()
        assert all(ev.id != "t1" for ev in repaired)

    def test_repair_does_not_mutate_the_input_list(self) -> None:
        original = _corrupted_events()
        original_ids = [ev.id for ev in original]
        repair_session_events("s1", original)
        assert [ev.id for ev in original] == original_ids

    def test_report_is_a_session_repaired_event(self) -> None:
        _, report = repair_session_events("s1", _corrupted_events())
        assert isinstance(report, SessionRepairedEvent)
        assert report.session_id == "s1"


@pytest.mark.asyncio
async def test_fixture_db_corrupted_session_repairs_and_round_trips(tmp_path: Path) -> None:
    """Integration test: a hand-built pre-#24 session DB (the exact shape
    EventStore.get_session_events would return from a real corrupted
    ``.nullain/sessions.db``) loads, repairs, and folds into a valid message
    history — no orphaned tool-role message anywhere.
    """
    db_path = tmp_path / "sessions.db"
    store = EventStore(db_path)
    await store.initialize()
    for ev in _corrupted_events():
        await store.append(ev)

    loaded = await store.get_session_events("s1")
    assert find_orphaned_tool_results(loaded), "fixture must reproduce real corruption"

    repaired, report = repair_session_events("s1", loaded)
    assert report is not None
    assert not find_orphaned_tool_results(repaired)

    state = Conversation.fold("s1", repaired)
    known_call_ids: set[str] = set()
    for msg in state.messages:
        d = msg.to_api_dict()
        if d["role"] == "tool":
            assert d["tool_call_id"] in known_call_ids
        for tc in msg.tool_calls or []:
            known_call_ids.add(tc.id)

    await store.close()


@st.composite
def _maybe_corrupted_sequence(draw: st.DrawFn) -> list[BaseEvent]:
    """Generate an arbitrary sequence of UserMessage / ModelResponse (with
    optional tool_calls) / ToolResult events, then a CompactionEvent that
    compacts an arbitrary subset of the *content* ids seen so far — which
    may or may not split a tool-call turn from its result, exercising both
    the corrupted and healthy paths.
    """
    session_id = "prop_session"
    n = draw(st.integers(min_value=0, max_value=8))
    events: list[BaseEvent] = []
    content_ids: list[str] = []
    call_ids: list[str] = []
    for i in range(n):
        kind = draw(st.sampled_from(["user", "model", "model_with_call", "tool"]))
        if kind == "user":
            ev: BaseEvent = UserMessageEvent(
                session_id=session_id, id=f"u{i}", content=draw(st.text(max_size=15))
            )
        elif kind == "model_with_call":
            call_id = f"call{i}"
            call_ids.append(call_id)
            ev = ModelResponseEvent(
                session_id=session_id,
                id=f"m{i}",
                model="m",
                content=None,
                tool_calls=(ToolCall(id=call_id, name="t", arguments={}),),
            )
        elif kind == "tool" and call_ids:
            call_id = draw(st.sampled_from(call_ids))
            ev = ToolResultEvent(
                session_id=session_id,
                id=f"t{i}",
                call_id=call_id,
                tool_name="t",
                output=draw(st.text(max_size=15)),
            )
        else:
            ev = ModelResponseEvent(
                session_id=session_id, id=f"m{i}", model="m", content=draw(st.text(max_size=15))
            )
        content_ids.append(ev.id)
        events.append(ev)

    if content_ids and draw(st.booleans()):
        compacted = draw(st.lists(st.sampled_from(content_ids), max_size=len(content_ids)))
        events.append(
            CompactionEvent(
                session_id=session_id,
                summary=draw(st.text(max_size=15)),
                compacted_event_ids=tuple(compacted),
            )
        )
    return events


@given(events=_maybe_corrupted_sequence())
def test_property_repaired_history_never_has_an_orphaned_tool_message(
    events: list[BaseEvent],
) -> None:
    """Property (issue #44): for ANY event sequence — corrupted or not —
    folding the output of repair_session_events never contains a `tool`-role
    message whose tool_call_id has no preceding assistant tool_calls entry.
    """
    repaired, _ = repair_session_events("prop_session", events)
    state = Conversation.fold("prop_session", repaired)

    known_call_ids: set[str] = set()
    for msg in state.messages:
        d = msg.to_api_dict()
        if d["role"] == "tool":
            assert d["tool_call_id"] in known_call_ids
        for tc in msg.tool_calls or []:
            known_call_ids.add(tc.id)


@given(events=_maybe_corrupted_sequence())
def test_property_repair_is_idempotent(events: list[BaseEvent]) -> None:
    """Repairing an already-repaired sequence must find nothing left to fix
    — repair_session_events should never leave residual corruption or flag
    its own output as still-corrupted."""
    repaired, _ = repair_session_events("prop_session", events)
    twice, second_report = repair_session_events("prop_session", repaired)
    assert second_report is None
    assert twice is repaired
