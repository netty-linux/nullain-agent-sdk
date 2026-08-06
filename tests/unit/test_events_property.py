"""Property-based tests and unit tests for Event Sourcing layer using Hypothesis."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from nullain.events import (
    BaseEvent,
    CompactionEvent,
    Conversation,
    EventBus,
    EventStore,
    IncrementalFold,
    ModelResponseEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from nullain.llm import TokenUsage


@pytest.mark.asyncio
async def test_event_bus_subscribe_publish() -> None:
    bus = EventBus()
    received: list[BaseEvent] = []

    async def handler(event: BaseEvent) -> None:
        received.append(event)

    bus.subscribe("user_message", handler)
    bus.subscribe("*", handler)

    ev = UserMessageEvent(session_id="s1", content="Hello bus")
    await bus.publish(ev)

    assert len(received) == 2  # 1 specific + 1 wildcard


@pytest.mark.asyncio
async def test_event_store_append_and_get() -> None:
    store = EventStore(":memory:")
    await store.initialize()

    ev1 = UserMessageEvent(session_id="sess_1", content="First message")
    ev2 = ModelResponseEvent(
        session_id="sess_1",
        model="qwen3",
        content="Response text",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    await store.append(ev1)
    await store.append(ev2)

    events = await store.get_session_events("sess_1")
    assert len(events) == 2
    assert events[0].id == ev1.id
    assert events[1].id == ev2.id

    state = Conversation.fold("sess_1", events)
    assert len(state.messages) == 2
    assert state.total_prompt_tokens == 10
    assert state.total_completion_tokens == 5

    await store.close()


@pytest.mark.asyncio
async def test_event_store_preserves_order_on_identical_timestamps() -> None:
    """Events sharing a timestamp come back in insertion order.

    Regression: ordering was ``timestamp ASC, id ASC``. ``time.time()`` has
    ~15.6 ms resolution on Windows, so consecutive appends routinely land on
    one timestamp and the tiebreaker on a random UUID reordered them —
    corrupting the trajectory. This pins the exact collision the CI hit,
    on every platform.
    """
    store = EventStore(":memory:")
    await store.initialize()

    stamp = 1_700_000_000.0
    appended = [
        UserMessageEvent(session_id="sess_ts", content=f"msg {i}", timestamp=stamp)
        for i in range(10)
    ]
    for ev in appended:
        await store.append(ev)

    events = await store.get_session_events("sess_ts")

    assert [e.id for e in events] == [e.id for e in appended]
    await store.close()


@given(
    user_texts=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10),
)
def test_hypothesis_event_fold_property(user_texts: list[str]) -> None:
    """Hypothesis property test: Folding events yields identical state after roundtrip."""
    session_id = "prop_test_session"
    original_events: list[BaseEvent] = []

    for text in user_texts:
        original_events.append(UserMessageEvent(session_id=session_id, content=text))
        original_events.append(
            ModelResponseEvent(
                session_id=session_id,
                model="qwen3",
                content=f"Echo: {text}",
                usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            )
        )

    original_state = Conversation.fold(session_id, original_events)

    # Roundtrip JSON serialization / deserialization
    serialized = [ev.model_dump_json() for ev in original_events]
    deserialized = [
        UserMessageEvent.model_validate_json(s)
        if "user_message" in s
        else ModelResponseEvent.model_validate_json(s)
        for s in serialized
    ]

    refolded_state = Conversation.fold(session_id, deserialized)

    assert len(original_state.messages) == len(refolded_state.messages)
    assert original_state.total_prompt_tokens == refolded_state.total_prompt_tokens
    assert original_state.total_completion_tokens == refolded_state.total_completion_tokens
    for m1, m2 in zip(original_state.messages, refolded_state.messages, strict=True):
        assert m1.role == m2.role
        assert m1.content == m2.content


@st.composite
def event_sequence(draw: st.DrawFn) -> list[BaseEvent]:
    """Generate an arbitrary sequence of fold-relevant events.

    Includes UserMessage / ModelResponse / ToolResult content events and
    Compaction events that compact a random subset of the content events seen so
    far, so the incremental fold's rebuild path is exercised.
    """
    session_id = "prop_session"
    n = draw(st.integers(min_value=0, max_value=10))
    events: list[BaseEvent] = []
    content_ids: list[str] = []
    for i in range(n):
        kind = draw(st.sampled_from(["user", "model", "tool", "compaction"]))
        if kind == "user":
            ev: BaseEvent = UserMessageEvent(
                session_id=session_id, content=draw(st.text(max_size=20))
            )
            content_ids.append(ev.id)
        elif kind == "model":
            ev = ModelResponseEvent(
                session_id=session_id,
                model="m",
                content=draw(st.text(max_size=20)),
                usage=draw(
                    st.one_of(
                        st.none(),
                        st.builds(
                            TokenUsage,
                            prompt_tokens=st.integers(0, 50),
                            completion_tokens=st.integers(0, 50),
                            total_tokens=st.integers(0, 100),
                        ),
                    )
                ),
            )
            content_ids.append(ev.id)
        elif kind == "tool":
            ev = ToolResultEvent(
                session_id=session_id,
                call_id=f"call{i}",
                tool_name="t",
                output=draw(st.text(max_size=20)),
            )
            content_ids.append(ev.id)
        else:
            compacted = (
                draw(st.lists(st.sampled_from(content_ids), max_size=len(content_ids)))
                if content_ids
                else []
            )
            ev = CompactionEvent(
                session_id=session_id,
                summary=draw(st.text(max_size=20)),
                compacted_event_ids=tuple(compacted),
            )
        events.append(ev)
    return events


@given(events=event_sequence())
def test_incremental_fold_matches_full_fold(events: list[BaseEvent]) -> None:
    """Property (M10 D3): incremental fold == full fold for any event sequence."""
    session_id = "prop_session"
    full = Conversation.fold(session_id, events)
    inc = IncrementalFold(session_id)
    inc.append(events)
    assert inc.state == full


@given(events=event_sequence())
def test_incremental_fold_one_by_one_matches_full_fold(events: list[BaseEvent]) -> None:
    """Property (M10 D3): appending events one at a time still equals the full fold.

    Exercises the incremental apply path (including compaction rebuilds) rather
    than a single bulk append.
    """
    session_id = "prop_session"
    full = Conversation.fold(session_id, events)
    inc = IncrementalFold(session_id)
    for ev in events:
        inc.append([ev])
    assert inc.state == full
