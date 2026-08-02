"""Property-based tests and unit tests for Event Sourcing layer using Hypothesis."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from nullain.events import (
    BaseEvent,
    Conversation,
    EventBus,
    EventStore,
    ModelResponseEvent,
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
