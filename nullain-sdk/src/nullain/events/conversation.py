"""Nullain Agent SDK — Conversation Event Folding and State Derivation."""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from nullain.events.types import (
    BaseEvent,
    CompactionEvent,
    ModelResponseEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from nullain.llm.types import ChatMessage


class ConversationState(BaseModel):
    """Derived conversation state aggregated by folding events in order."""

    session_id: str
    events: list[BaseEvent] = Field(default_factory=list[BaseEvent])
    messages: list[ChatMessage] = Field(default_factory=list[ChatMessage])
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    # Real prompt-token count of the most recent model response (M10 D5). This
    # is the provider-reported ``usage`` — the real context size — preferred over
    # the ``len//4`` estimate for compaction decisions. Reset to 0 on compaction
    # because the retained events' usage reflects the pre-compaction context.
    last_prompt_tokens: int = 0
    compaction_summary: str | None = None


class Conversation:
    """Conversation manager for event sourcing state derivation."""

    @staticmethod
    def fold(session_id: str, events: Sequence[BaseEvent]) -> ConversationState:
        """Derive conversation state by performing a pure functional fold over events."""
        state = ConversationState(session_id=session_id, events=list(events))

        compacted_ids: set[str] = set()
        # Collect compacted event IDs if a compaction event occurred
        for ev in events:
            if isinstance(ev, CompactionEvent):
                compacted_ids.update(ev.compacted_event_ids)
                state.compaction_summary = ev.summary

        # Build message history omitting compacted event IDs
        for ev in events:
            if ev.id in compacted_ids:
                continue

            if isinstance(ev, CompactionEvent):
                # The retained events' usage reflects the pre-compaction
                # context; the real size is re-measured by the next request.
                state.last_prompt_tokens = 0

            elif isinstance(ev, UserMessageEvent):
                state.messages.append(ChatMessage(role="user", content=ev.content))

            elif isinstance(ev, ModelResponseEvent):
                if ev.usage:
                    state.total_prompt_tokens += ev.usage.prompt_tokens
                    state.total_completion_tokens += ev.usage.completion_tokens
                    state.last_prompt_tokens = ev.usage.prompt_tokens

                tcs = list(ev.tool_calls) if ev.tool_calls else None
                state.messages.append(
                    ChatMessage(
                        role="assistant",
                        content=ev.content,
                        tool_calls=tcs,
                    )
                )

            elif isinstance(ev, ToolResultEvent):
                state.messages.append(
                    ChatMessage(
                        role="tool",
                        content=ev.output,
                        tool_call_id=ev.call_id,
                        name=ev.tool_name,
                    )
                )

        # Prepend compaction summary as a system message if present
        if state.compaction_summary:
            state.messages.insert(
                0,
                ChatMessage(
                    role="system",
                    content=f"[SUMMARY OF PRIOR CONVERSATION]\n{state.compaction_summary}",
                ),
            )

        return state


class IncrementalFold:
    """Cached, incremental conversation fold.

    Maintains a :class:`ConversationState` that is updated by applying only the
    newly-appended events, rather than re-folding the entire history on every
    step. ``Conversation.fold`` remains the pure reference implementation; this
    cache is a verifiable optimization — property-tested to equal the full fold
    for any event sequence.

    A :class:`CompactionEvent` invalidates the derived message/token state: the
    compacted events are dropped and the summary is prepended, so the cache is
    rebuilt from the retained events (compaction is rare, so the O(n) rebuild is
    amortized away by the O(1) incremental appends in between).
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._state = ConversationState(session_id=session_id)
        self._all_events: list[BaseEvent] = []
        self._compacted_ids: set[str] = set()

    @property
    def state(self) -> ConversationState:
        """The current derived conversation state."""
        return self._state

    def append(self, events: Sequence[BaseEvent]) -> ConversationState:
        """Apply new events incrementally and return the updated state."""
        for ev in events:
            self._all_events.append(ev)
            if isinstance(ev, CompactionEvent):
                self._compacted_ids.update(ev.compacted_event_ids)
                self._state.compaction_summary = ev.summary
                self._rebuild()
                # Retained events carry stale pre-compaction usage; the real
                # size is re-measured by the next model response (M10 D5).
                self._state.last_prompt_tokens = 0
            else:
                self._apply_message(ev)
        self._state.events = list(self._all_events)
        return self._state

    def _apply_message(self, ev: BaseEvent) -> None:
        """Fold a single non-compaction event into the cached state."""
        if ev.id in self._compacted_ids:
            return
        if isinstance(ev, UserMessageEvent):
            self._state.messages.append(ChatMessage(role="user", content=ev.content))
        elif isinstance(ev, ModelResponseEvent):
            if ev.usage:
                self._state.total_prompt_tokens += ev.usage.prompt_tokens
                self._state.total_completion_tokens += ev.usage.completion_tokens
                self._state.last_prompt_tokens = ev.usage.prompt_tokens
            tcs = list(ev.tool_calls) if ev.tool_calls else None
            self._state.messages.append(
                ChatMessage(role="assistant", content=ev.content, tool_calls=tcs)
            )
        elif isinstance(ev, ToolResultEvent):
            self._state.messages.append(
                ChatMessage(
                    role="tool",
                    content=ev.output,
                    tool_call_id=ev.call_id,
                    name=ev.tool_name,
                )
            )

    def _rebuild(self) -> None:
        """Rebuild the derived state from retained (non-compacted) events."""
        self._state.messages = []
        self._state.total_prompt_tokens = 0
        self._state.total_completion_tokens = 0
        for ev in self._all_events:
            if ev.id in self._compacted_ids:
                continue
            self._apply_message(ev)
        if self._state.compaction_summary:
            self._state.messages.insert(
                0,
                ChatMessage(
                    role="system",
                    content=f"[SUMMARY OF PRIOR CONVERSATION]\n{self._state.compaction_summary}",
                ),
            )


__all__ = ["Conversation", "ConversationState", "IncrementalFold"]
