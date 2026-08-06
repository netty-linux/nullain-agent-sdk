"""Nullain Agent SDK — Task Intent and Complexity Classifier."""

import asyncio
import hashlib
from enum import StrEnum

from pydantic import BaseModel

from nullain.llm.provider import LLMProvider
from nullain.llm.types import ChatMessage, CompletionRequest
from nullain.telemetry import get_logger

logger = get_logger(__name__)


class Complexity(StrEnum):
    """Complexity classification levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class IntentResult(BaseModel):
    """Result of intent parsing and complexity evaluation."""

    intent_type: str
    complexity: Complexity
    suggested_tier: str


#: Suggested tier per complexity, shared by the heuristic and classifier paths.
_TIER_FOR_COMPLEXITY: dict[Complexity, str] = {
    Complexity.LOW: "fast",
    Complexity.MEDIUM: "balanced",
    Complexity.HIGH: "deep",
}

#: Short timeout for the classifier call — intent classification is a cheap,
#: low-stakes request; a slow classifier must not stall the session.
_CLASSIFIER_TIMEOUT = 5.0

#: Prompt asking the classifier model to emit intent + complexity on two lines.
_CLASSIFIER_PROMPT = (
    "Classify the following user task. Respond with exactly two lines:\n"
    "intent: <simple_edit|general_task|complex_architecture>\n"
    "complexity: <LOW|MEDIUM|HIGH>\n\nTask: {prompt}"
)


class IntentParser:
    """Classifies user tasks using deterministic heuristics, with an optional
    LLM classifier fallback (M11.3).

    The deterministic heuristics run first and are authoritative when they
    match a known keyword. When none matches with confidence, ``parse_async``
    asks the configured ``classifier_model`` (tier ``fast``) to classify the
    task, falling back to the heuristic default on any failure. Results are
    cached by prompt hash within the session.
    """

    def __init__(self, cache_size: int = 128) -> None:
        self._cache: dict[str, IntentResult] = {}
        self._cache_size = cache_size

    # -- Heuristic path ------------------------------------------------------

    def _heuristic(self, prompt: str) -> IntentResult | None:
        """Return a confident heuristic classification, or ``None`` if uncertain.

        ``None`` signals that no keyword matched with confidence, which is the
        trigger for the LLM classifier fallback in :meth:`parse_async`.
        """
        text = prompt.lower().strip()

        # Heuristic 1: Low complexity tasks (formatting, commit msg/message, simple queries)
        low_keywords = ("format", "lint", "commit msg", "commit message", "typo", "classify")
        if any(w in text for w in low_keywords):
            return IntentResult(
                intent_type="simple_edit",
                complexity=Complexity.LOW,
                suggested_tier=_TIER_FOR_COMPLEXITY[Complexity.LOW],
            )

        # Heuristic 2: High complexity tasks (architect, refactor multi-file, debug complex)
        high_keywords = ("architect", "refactor", "overhaul", "redesign", "security audit")
        if any(w in text for w in high_keywords):
            return IntentResult(
                intent_type="complex_architecture",
                complexity=Complexity.HIGH,
                suggested_tier=_TIER_FOR_COMPLEXITY[Complexity.HIGH],
            )

        return None

    def parse(self, prompt: str) -> IntentResult:
        """Heuristic-only classification (backward compatible).

        When no keyword matches, returns the medium/general default. The loop
        prefers :meth:`parse_async` so an uncertain prompt can be routed to the
        LLM classifier.
        """
        result = self._heuristic(prompt)
        if result is not None:
            return result
        return IntentResult(
            intent_type="general_task",
            complexity=Complexity.MEDIUM,
            suggested_tier=_TIER_FOR_COMPLEXITY[Complexity.MEDIUM],
        )

    # -- Classifier path -----------------------------------------------------

    async def parse_async(
        self,
        prompt: str,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ) -> IntentResult:
        """Classify a prompt, using the LLM classifier when the heuristic is
        uncertain.

        Deterministic heuristics run first; only when none matches with
        confidence (and a ``provider`` + ``model`` are available) is the
        classifier called. Any classifier failure falls back to the heuristic
        default. Results are cached by prompt hash within the session.

        Args:
            prompt: The user task to classify.
            provider: An :class:`~nullain.llm.LLMProvider` used for the
                classifier call. ``None`` keeps the parser heuristic-only.
            model: The classifier model id (typically ``router.classifier_model``).
        """
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = self._heuristic(prompt)
        if result is None and provider is not None and model:
            result = await self._classify(prompt, provider, model)
        if result is None:
            result = self.parse(prompt)

        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = result
        return result

    async def _classify(
        self, prompt: str, provider: LLMProvider, model: str
    ) -> IntentResult | None:
        """Ask the classifier model to classify intent + complexity.

        Returns ``None`` on any failure so the caller falls back to the
        heuristic default (fail-soft — a classifier outage must not derail the
        session).
        """
        try:
            request = CompletionRequest(
                model=model,
                messages=[
                    ChatMessage(
                        role="user",
                        content=_CLASSIFIER_PROMPT.format(prompt=prompt),
                    )
                ],
                temperature=0.0,
                max_tokens=32,
                stream=False,
            )
            chunk = await asyncio.wait_for(provider.generate(request), timeout=_CLASSIFIER_TIMEOUT)
        except Exception as err:
            logger.warning("intent_classifier_failed", model=model, error=str(err))
            return None
        return self._parse_classifier_output((chunk.delta_text or "").strip())

    def _parse_classifier_output(self, text: str) -> IntentResult | None:
        """Parse the classifier's two-line output into an :class:`IntentResult`.

        Returns ``None`` if the output does not contain a recognizable
        complexity and intent, so the caller falls back to the heuristic.
        """
        lowered = text.lower()
        if "complexity: high" in lowered:
            complexity = Complexity.HIGH
        elif "complexity: low" in lowered:
            complexity = Complexity.LOW
        elif "complexity: medium" in lowered:
            complexity = Complexity.MEDIUM
        else:
            return None

        if "intent: complex_architecture" in lowered:
            intent_type = "complex_architecture"
        elif "intent: simple_edit" in lowered:
            intent_type = "simple_edit"
        elif "intent: general_task" in lowered:
            intent_type = "general_task"
        else:
            return None

        return IntentResult(
            intent_type=intent_type,
            complexity=complexity,
            suggested_tier=_TIER_FOR_COMPLEXITY[complexity],
        )


__all__ = ["Complexity", "IntentParser", "IntentResult"]
