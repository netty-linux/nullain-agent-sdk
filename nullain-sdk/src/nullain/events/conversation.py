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

            if isinstance(ev, UserMessageEvent):
                state.messages.append(ChatMessage(role="user", content=ev.content))

            elif isinstance(ev, ModelResponseEvent):
                if ev.usage:
                    state.total_prompt_tokens += ev.usage.prompt_tokens
                    state.total_completion_tokens += ev.usage.completion_tokens

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


__all__ = ["Conversation", "ConversationState"]
