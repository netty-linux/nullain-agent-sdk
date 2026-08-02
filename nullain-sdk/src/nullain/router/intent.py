"""Nullain Agent SDK — Task Intent and Complexity Classifier."""

from enum import StrEnum

from pydantic import BaseModel


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


class IntentParser:
    """Classifies user tasks using deterministic heuristics."""

    def parse(self, prompt: str) -> IntentResult:
        """Parse user prompt and return classified intent and suggested tier."""
        text = prompt.lower().strip()

        # Heuristic 1: Low complexity tasks (formatting, commit msg/message, simple queries)
        low_keywords = ("format", "lint", "commit msg", "commit message", "typo", "classify")
        if any(w in text for w in low_keywords):
            return IntentResult(
                intent_type="simple_edit",
                complexity=Complexity.LOW,
                suggested_tier="fast",
            )

        # Heuristic 2: High complexity tasks (architect, refactor multi-file, debug complex)
        high_keywords = ("architect", "refactor", "overhaul", "redesign", "security audit")
        if any(w in text for w in high_keywords):
            return IntentResult(
                intent_type="complex_architecture",
                complexity=Complexity.HIGH,
                suggested_tier="deep",
            )

        # Heuristic 3: Medium complexity tasks (implement feature, write test, fix bug)
        return IntentResult(
            intent_type="general_task",
            complexity=Complexity.MEDIUM,
            suggested_tier="balanced",
        )


__all__ = ["Complexity", "IntentParser", "IntentResult"]
