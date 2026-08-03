"""Nullain Agent SDK — Sandbox selector.

Picks the :class:`~nullain.tools.sandbox.port.Sandbox` adapter for the current
platform from the configured :class:`~nullain.config.settings.SandboxConfig`.

Selection rules:
- ``enabled == False`` -> :class:`NoSandbox` (explicit opt-out, never required).
- Otherwise prefer a real platform adapter when one is shipped for this OS.
  Until a real adapter reports ``available()``, the runner would fail-closed on
  a ``required`` adapter; to keep the transition regression-free, a platform
  with no available real adapter falls back to NoSandbox (permissive). Real
  adapters (landlock / seatbelt / windows_job) are wired in follow-up commits
  and replace this fallback on their respective platforms.
"""

from __future__ import annotations

import sys

from nullain.config.settings import SandboxConfig
from nullain.tools.sandbox.adapters.none import NoSandbox
from nullain.tools.sandbox.port import Sandbox


def _platform_adapter(cfg: SandboxConfig) -> Sandbox | None:
    """Return the real adapter for this platform, or None if none is available.

    Real adapters (landlock on Linux, seatbelt on macOS, windows_job on Windows)
    are added in follow-up commits and inserted here. A returned adapter carries
    ``required`` from config so the runner fail-closes when it is unavailable.
    """
    # Foundation: no real adapter wired yet. Platform gating kept explicit so
    # follow-up commits plug in cleanly without re-architecting the selector.
    _ = (sys.platform, cfg)
    return None


def select_sandbox(cfg: SandboxConfig) -> Sandbox:
    """Select the sandbox adapter for the current platform and config.

    Args:
        cfg: Sandbox configuration (enabled / required / allow_paths / deny_network).

    Returns:
        A Sandbox adapter. NoSandbox when disabled or no real adapter is
        available yet; a real platform adapter otherwise.
    """
    if not cfg.enabled:
        return NoSandbox(required=False)
    adapter = _platform_adapter(cfg)
    if adapter is None:
        return NoSandbox(required=False)
    return adapter


__all__ = ["select_sandbox"]
