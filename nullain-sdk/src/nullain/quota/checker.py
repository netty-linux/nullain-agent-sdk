"""Nullain Agent SDK — Quota Enforcement Port (ADR-4).

`QuotaChecker` is a small in-process Protocol `AgentLoop` consults before
each step (alongside the existing `max_tokens` budget check) — not a
`HookManager` hook. `HookManager`'s hooks spawn an external subprocess per
invocation (JSON over stdin/stdout, `docs/FUSION_PLAN.md`'s original ADR-4
sketch called this a "hook", but that subprocess-per-call model is the
wrong shape for a check that needs to run on every LLM/tool step at low
latency — this Protocol is the in-process equivalent, deliberately not
wired through `HookManager`.

Split of responsibility (ADR-4): `QuotaChecker` is *enforcement*
(fail-closed, in the critical path, denies the step outright) — separate
from *accounting* (recording what was actually spent), which stays the
caller's responsibility (e.g. `nullain-agent`'s `metering.py` subscribing
to the `EventBus`). `AgentLoop.max_tokens` remains the runaway-loop circuit
breaker for a single run; `QuotaChecker` is the per-tenant budget check
across runs (daily/monthly caps, tier-based limits) — the two compose, they
don't replace each other.

Optional by design: `AgentLoop(quota_checker=None)` (the default) means no
quota enforcement at all, identical to pre-ADR-4 behavior. A configured
checker's `check()` denial raises `QuotaExceededError`, structurally
distinct from `BudgetExceededError` (the per-run token ceiling) so a
caller can tell "this session ran too long" apart from "this tenant is out
of quota" without string-matching an error message.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nullain.errors import NullainError


class QuotaExceededError(NullainError):
    """Raised when a `QuotaChecker` denies a step — the tenant identified
    by `session_id` has exhausted its budget. Distinct from
    `BudgetExceededError` (AgentLoop.max_tokens, a per-run circuit breaker
    unrelated to any tenant/billing concept)."""


@runtime_checkable
class QuotaChecker(Protocol):
    """Abstract protocol for pre-step quota enforcement."""

    async def check(self, session_id: str) -> None:
        """Raise `QuotaExceededError` if `session_id` has no budget left
        to take another step. Returning normally means "allowed" — this
        is a permission gate, not a usage query (see a separate accounting
        mechanism, e.g. an EventBus subscriber, for that)."""
        ...


__all__ = ["QuotaChecker", "QuotaExceededError"]
