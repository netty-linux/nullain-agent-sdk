"""Unit tests for ModelRouter and IntentParser."""

import pytest
from nullain.config import RouterConfig, TierConfig
from nullain.errors import NoModelAvailableError
from nullain.router import Complexity, IntentParser, ModelRouter


@pytest.mark.parametrize(
    ("prompt", "expected_tier", "expected_complexity"),
    [
        ("format code and fix lint issues", "fast", Complexity.LOW),
        ("generate git commit message", "fast", Complexity.LOW),
        ("implement user login authentication endpoint", "balanced", Complexity.MEDIUM),
        ("write unit test for event store", "balanced", Complexity.MEDIUM),
        ("architect multi-tenant database migration overhaul", "deep", Complexity.HIGH),
        ("refactor core event bus for parallel processing", "deep", Complexity.HIGH),
    ],
)
def test_intent_parser_classification(
    prompt: str, expected_tier: str, expected_complexity: Complexity
) -> None:
    parser = IntentParser()
    res = parser.parse(prompt)
    assert res.suggested_tier == expected_tier
    assert res.complexity == expected_complexity


def test_model_router_tier_selection() -> None:
    router = ModelRouter()
    assert router.select_model("fast") == "gpt-oss:20b"
    assert router.select_model("balanced") == "qwen3-coder:480b-cloud"
    assert router.select_model("deep") == "deepseek-v4-pro"


def test_circuit_breaker_and_fallback() -> None:
    cfg = RouterConfig(
        tiers={
            "fast": TierConfig(models=["m_fast1"]),
            "balanced": TierConfig(models=["m_bal1"]),
            "deep": TierConfig(models=["m_deep1"]),
        },
        fallback_chain=["balanced", "deep", "fast"],
    )
    router = ModelRouter(config=cfg)

    # Initially selects m_fast1 for fast tier
    assert router.select_model("fast") == "m_fast1"

    # Simulate 3 failures for m_fast1
    router.circuit_breaker.record_failure("m_fast1")
    router.circuit_breaker.record_failure("m_fast1")
    router.circuit_breaker.record_failure("m_fast1")

    # Fast model circuit opens -> falls back to next in fallback chain (m_bal1)
    assert router.select_model("fast") == "m_bal1"

    # Record failures for m_bal1 and m_deep1 as well
    for _ in range(3):
        router.circuit_breaker.record_failure("m_bal1")
        router.circuit_breaker.record_failure("m_deep1")

    # All circuits open -> raises NoModelAvailableError
    with pytest.raises(NoModelAvailableError):
        router.select_model("fast")
