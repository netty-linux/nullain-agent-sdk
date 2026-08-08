"""Nullain Agent SDK — Quota Enforcement Module (ADR-4).

See `nullain.quota.checker` for the full design rationale — in short:
`QuotaChecker` is an in-process, per-step enforcement Protocol `AgentLoop`
consults before each step, kept separate from `HookManager` (which spawns
external subprocesses and is too slow for a check that runs this often)
and from accounting (recording actual spend, the caller's job).
"""

from nullain.quota.checker import QuotaChecker, QuotaExceededError

__all__ = ["QuotaChecker", "QuotaExceededError"]
