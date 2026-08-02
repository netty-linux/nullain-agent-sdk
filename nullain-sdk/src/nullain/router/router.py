"""Nullain Agent SDK — Model Router and Fallback Circuit Breaker."""

import time

from nullain.config.settings import RouterConfig
from nullain.errors import NoModelAvailableError
from nullain.router.intent import IntentResult


class CircuitBreaker:
    """Circuit breaker tracking model failure rates."""

    def __init__(self, failure_threshold: int = 3, recovery_time: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failures: dict[str, int] = {}
        self.opened_at: dict[str, float] = {}

    def is_open(self, model: str) -> bool:
        """Check if circuit is open (model disabled due to failures)."""
        if model in self.opened_at:
            if (time.time() - self.opened_at[model]) > self.recovery_time:
                # Reset circuit after recovery time
                del self.opened_at[model]
                self.failures[model] = 0
                return False
            return True
        return False

    def record_failure(self, model: str) -> None:
        """Record a model failure and open circuit if threshold is reached."""
        count = self.failures.get(model, 0) + 1
        self.failures[model] = count
        if count >= self.failure_threshold:
            self.opened_at[model] = time.time()

    def record_success(self, model: str) -> None:
        """Record a successful model call and reset failure counters."""
        self.failures[model] = 0
        if model in self.opened_at:
            del self.opened_at[model]


class ModelRouter:
    """Model Router mapping tasks to tier models with circuit breaker fallbacks."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()
        self.circuit_breaker = CircuitBreaker()

    def select_model(self, tier: str) -> str:
        """Select best available model for a requested tier.

        Falls back across `fallback_chain` if primary tier models are open in circuit breaker.

        Raises:
            NoModelAvailableError if all candidates across all tiers are unavailable.
        """
        candidate_tiers = [tier] + [t for t in self.config.fallback_chain if t != tier]

        for t in candidate_tiers:
            tier_cfg = self.config.tiers.get(t)
            if not tier_cfg:
                continue
            for model in tier_cfg.models:
                if not self.circuit_breaker.is_open(model):
                    return model

        raise NoModelAvailableError(f"No available model found for tier '{tier}' or fallback chain")

    def escalate_tier(self, current_tier: str) -> str:
        """Escalate model tier (e.g. fast -> balanced -> deep) after failures."""
        escalation_map = {
            "fast": "balanced",
            "balanced": "deep",
            "deep": "deep",
        }
        return escalation_map.get(current_tier, "deep")

    def route_intent(self, intent: IntentResult) -> str:
        """Route an IntentResult to an active model name."""
        return self.select_model(intent.suggested_tier)


__all__ = ["CircuitBreaker", "ModelRouter"]
