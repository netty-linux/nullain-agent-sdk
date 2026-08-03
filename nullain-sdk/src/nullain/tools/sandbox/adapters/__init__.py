"""Nullain Agent SDK — Sandbox adapters (platform-specific isolation).

Each adapter implements the :class:`~nullain.tools.sandbox.port.Sandbox` port.
The selector (:mod:`nullain.tools.sandbox.selector`) picks the matching adapter
for the current platform; the runner fail-closes when a required adapter reports
unavailable.
"""

from nullain.tools.sandbox.adapters.none import NoSandbox

__all__ = ["NoSandbox"]
