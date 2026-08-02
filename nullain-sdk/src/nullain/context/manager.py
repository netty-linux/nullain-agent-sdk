"""Nullain Agent SDK — ContextManager for Context Window Compaction and Defenses."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from nullain.agent.spec import TaskSpec
from nullain.events import (
    BaseEvent,
    CompactionEvent,
    ModelResponseEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from nullain.llm.provider import LLMProvider
from nullain.llm.types import ChatMessage, CompletionRequest
from nullain.telemetry import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Number of recent events kept verbatim (not summarized) during compaction.
_RECENT_KEEP = 4


class ContextManager:
    """Manages LLM context window, compaction, and instruction centrifugation.

    Compaction produces a ``CompactionEvent`` whose ``summary`` recapitulates
    the compacted trajectory. Two modes:

    - **Structural** (default, no provider): a bookkeeping summary that lists
      the active spec objective and the user prompts verbatim. Honest about
      being structural — it does NOT claim to semantically summarize.
    - **LLM summarization** (provider + model passed to ``compact``): asks the
      model to recap the compacted events concisely, preserving key decisions,
      file changes and outstanding work. Falls back to the structural summary
      if the provider call fails.
    """

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

    @staticmethod
    def _collect_compacted(events: Sequence[BaseEvent]) -> tuple[list[str], list[str], str]:
        """Split events into (compacted_ids, user_prompts, compacted_text)."""
        compacted_ids: list[str] = []
        user_prompts: list[str] = []
        text_parts: list[str] = []

        compact_candidates: Sequence[BaseEvent] = (
            events[:-_RECENT_KEEP] if len(events) > _RECENT_KEEP else []
        )
        for ev in compact_candidates:
            compacted_ids.append(ev.id)
            if isinstance(ev, UserMessageEvent):
                user_prompts.append(ev.content)
                text_parts.append(f"User: {ev.content}")
            elif isinstance(ev, ModelResponseEvent):
                if ev.content:
                    text_parts.append(f"Assistant: {ev.content}")
            elif isinstance(ev, ToolResultEvent):
                text_parts.append(f"Tool({ev.tool_name}): {ev.output}")
        return compacted_ids, user_prompts, "\n".join(text_parts)

    def _structural_summary(
        self,
        compacted_count: int,
        user_prompts: list[str],
        active_spec: TaskSpec | None,
    ) -> str:
        """Build the fallback bookkeeping summary (no LLM call)."""
        parts = [f"[Compacted {compacted_count} trajectory events (structural summary)]"]
        if active_spec:
            parts.append(
                f"Active Spec Objective: {active_spec.objective} "
                f"(Steps: {', '.join(active_spec.steps)})"
            )
        if user_prompts:
            parts.append(f"User Prompts: {' | '.join(user_prompts)}")
        return "\n".join(parts)

    async def _llm_summarize(
        self,
        provider: LLMProvider,
        model: str,
        compacted_text: str,
        active_spec: TaskSpec | None,
    ) -> str:
        """Ask the LLM to summarize the compacted trajectory text."""
        spec_context = ""
        if active_spec:
            spec_context = (
                f"\nActive task objective: {active_spec.objective}\n"
                f"Planned steps: {', '.join(active_spec.steps)}\n"
            )
        prompt = (
            "Summarize the following earlier conversation trajectory concisely. "
            "Preserve key decisions, file changes, errors encountered, and "
            "outstanding work so the agent can continue without losing context. "
            f"{spec_context}\n"
            f"Trajectory:\n{compacted_text}"
        )
        req = CompletionRequest(
            model=model,
            messages=[
                ChatMessage(
                    role="system",
                    content="You are a conversation summarizer. Be concise and factual.",
                ),
                ChatMessage(role="user", content=prompt),
            ],
            stream=False,
        )
        response = await provider.generate(req)
        text = (response.delta_text or "").strip()
        if not text:
            raise ValueError("empty summary from provider")
        return text

    async def compact(
        self,
        session_id: str,
        events: Sequence[BaseEvent],
        active_spec: TaskSpec | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ) -> CompactionEvent:
        """Perform trajectory compaction, creating a CompactionEvent.

        When ``provider`` and ``model`` are supplied, the compacted events are
        summarized by the LLM (real summarization). Otherwise, or on provider
        failure, a structural bookkeeping summary is produced.
        """
        compacted_ids, user_prompts, compacted_text = self._collect_compacted(events)
        structural = self._structural_summary(len(compacted_ids), user_prompts, active_spec)

        summary = structural
        if provider is not None and model and compacted_text:
            try:
                llm_summary = await self._llm_summarize(
                    provider, model, compacted_text, active_spec
                )
                summary = f"[Compacted {len(compacted_ids)} trajectory events]\n{llm_summary}"
            except Exception as err:
                logger.warning("context_llm_summarize_failed", error=str(err))
                summary = structural  # honest fallback

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
