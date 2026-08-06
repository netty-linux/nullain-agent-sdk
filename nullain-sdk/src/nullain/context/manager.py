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

# Characters per token assumed before any real ``usage`` has been observed.
# Deliberately the classic "~4 chars per token" heuristic; it is only the
# bootstrap value, replaced by the measured ratio after the first response.
_DEFAULT_CHARS_PER_TOKEN = 4.0

# Bounds on the calibrated chars-per-token ratio. A ratio outside this range
# means the sample was degenerate (e.g. a near-empty context whose prompt
# tokens are dominated by fixed provider overhead), so it is discarded rather
# than allowed to skew every later estimate.
_MIN_CHARS_PER_TOKEN = 1.0
_MAX_CHARS_PER_TOKEN = 12.0

# Minimum context size (in characters) a sample must have before it is used for
# calibration. Small contexts are dominated by per-request overhead and would
# calibrate the ratio far too low.
_MIN_CALIBRATION_CHARS = 200

# Weight of a new observation in the exponential moving average. Low enough
# that one anomalous response cannot swing the ratio, high enough to track a
# genuine model change within a few steps.
_CALIBRATION_ALPHA = 0.3


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
        # Calibrated characters-per-token ratio, refined by ``calibrate`` from
        # the provider's real ``usage``. Starts at the bootstrap heuristic.
        self._chars_per_token = _DEFAULT_CHARS_PER_TOKEN
        self._calibration_samples = 0

    @property
    def chars_per_token(self) -> float:
        """Current characters-per-token ratio used for estimation.

        Starts at the bootstrap heuristic (~4.0) and converges on the ratio
        actually observed from the provider's reported prompt tokens.
        """
        return self._chars_per_token

    @property
    def calibration_samples(self) -> int:
        """Number of provider responses that have refined the ratio."""
        return self._calibration_samples

    def calibrate(self, messages: list[ChatMessage], prompt_tokens: int) -> None:
        """Refine the chars-per-token ratio from a provider's real token count.

        The agent loop calls this after each model response, passing the exact
        messages that were sent and the ``prompt_tokens`` the provider reported
        for them. The measured ratio is folded into an exponential moving
        average, so estimation converges on the tokenizer the active model
        actually uses instead of a fixed ~4-chars-per-token guess.

        Degenerate samples are ignored: a non-positive token count, a context
        too small to be dominated by content rather than per-request overhead,
        or a ratio outside the plausible band.

        Args:
            messages: The exact messages sent in the request.
            prompt_tokens: Prompt tokens the provider reported for them.
        """
        if prompt_tokens <= 0:
            return
        chars = self._context_chars(messages)
        if chars < _MIN_CALIBRATION_CHARS:
            return
        observed = chars / prompt_tokens
        if not _MIN_CHARS_PER_TOKEN <= observed <= _MAX_CHARS_PER_TOKEN:
            logger.debug(
                "token_calibration_sample_rejected",
                observed_ratio=round(observed, 3),
                chars=chars,
                prompt_tokens=prompt_tokens,
            )
            return
        if self._calibration_samples == 0:
            self._chars_per_token = observed
        else:
            self._chars_per_token = (
                _CALIBRATION_ALPHA * observed + (1.0 - _CALIBRATION_ALPHA) * self._chars_per_token
            )
        self._calibration_samples += 1

    @staticmethod
    def _context_chars(messages: list[ChatMessage]) -> int:
        """Total characters the given messages contribute to the context.

        Mirrors exactly what :meth:`estimate_context_tokens` measures, so the
        calibrated ratio is derived from the same quantity it is applied to.
        """
        total = 0
        for msg in messages:
            if msg.content:
                total += len(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += len(str(tc.arguments))
        return total

    def estimate_tokens(self, text: str) -> int:
        """Estimate the token count of a text snippet.

        Uses the calibrated characters-per-token ratio (see :meth:`calibrate`),
        falling back to the ~4-chars-per-token heuristic until the first real
        ``usage`` has been observed.
        """
        return max(1, int(len(text) / self._chars_per_token))

    def estimate_context_tokens(self, messages: list[ChatMessage]) -> int:
        """Estimate the token count of the current context window messages."""
        return max(0, int(self._context_chars(messages) / self._chars_per_token))

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
