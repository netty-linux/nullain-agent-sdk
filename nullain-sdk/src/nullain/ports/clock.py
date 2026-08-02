"""Nullain Agent SDK — Clock Port for Dependency Injection."""

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Abstract clock for deterministic testing."""

    def now(self) -> float:
        """Return current timestamp as float (epoch seconds)."""
        ...


class SystemClock:
    """Default system clock using time.time()."""

    def now(self) -> float:
        """Return current system time."""
        return time.time()


__all__ = ["Clock", "SystemClock"]
