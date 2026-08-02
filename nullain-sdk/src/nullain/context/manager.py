"""Nullain Agent SDK — ContextManager for Context Window Compaction and Defenses."""

from collections.abc import Sequence

from nullain.agent.spec import TaskSpec
from nullain.events import BaseEvent, CompactionEvent, UserMessageEvent
from nullain.llm.types import ChatMessage


class ContextManager:
    """Manages LLM context window, compaction, and instruction centrifugation."""

    def __init__(
        self,
        max_window_tokens: int = 32000,
        compaction_threshold: float = 0.75,
        reinject_every_steps: int = 5,
    ) -> None:
        self.max_window_tokens = max_window_tokens
        self.compaction_threshold = compaction_threshold
        self.reinject_every_steps = reinject_every_steps

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text snippet (approx 4 chars per token)."""
        return max(1, len(text) // 4)

    def estimate_context_tokens(self, messages: list[ChatMessage]) -> int:
        """Estimate total token count for current context window messages."""
        total = 0
        for msg in messages:
            if msg.content:
                total += self.estimate_tokens(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += self.estimate_tokens(str(tc.arguments))
        return total

    def should_compact(self, current_tokens: int) -> bool:
        """Check if current token usage exceeds the compaction threshold."""
        return current_tokens >= int(self.max_window_tokens * self.compaction_threshold)

    def compact(
        self,
        session_id: str,
        events: Sequence[BaseEvent],
        active_spec: TaskSpec | None = None,
    ) -> CompactionEvent:
        """Perform trajectory compaction, creating a CompactionEvent.

        Summarizes earlier events while preserving active spec and recent interactions.
        """
        compacted_ids: list[str] = []
        user_prompts: list[str] = []

        # Keep recent events (last 4) intact, compact earlier ones
        compact_candidates: Sequence[BaseEvent] = events[:-4] if len(events) > 4 else []
        for ev in compact_candidates:
            compacted_ids.append(ev.id)
            if isinstance(ev, UserMessageEvent):
                user_prompts.append(ev.content)

        summary_parts = [f"Compacted {len(compacted_ids)} trajectory events."]
        if active_spec:
            summary_parts.append(
                f"Active Spec Objective: {active_spec.objective} "
                f"(Steps: {', '.join(active_spec.steps)})"
            )
        if user_prompts:
            summary_parts.append(f"User Prompts Summary: {' | '.join(user_prompts)}")

        summary = "\n".join(summary_parts)

        return CompactionEvent(
            session_id=session_id,
            summary=summary,
            compacted_event_ids=tuple(compacted_ids),
        )

    def reinject_instructions(
        self, messages: list[ChatMessage], step_count: int
    ) -> list[ChatMessage]:
        """Re-inject critical operational rules near the end of context if step interval hit."""
        if step_count > 0 and step_count % self.reinject_every_steps == 0:
            reinject_msg = ChatMessage(
                role="system",
                content=(
                    "[SYSTEM INSTRUCTION RE-INJECTION]\n"
                    "Remember: Verify tool inputs, follow workspace permissions strictly, "
                    "and work towards completing acceptance criteria defined in TaskSpec."
                ),
            )
            return [*messages, reinject_msg]
        return messages


__all__ = ["ContextManager"]
