"""Nullain Agent SDK — Subagent Authority and Capability Intersection.

The authority-intersection law (P4.24): a child subagent's effective authority
is the meet (greatest-lower-bound / intersection) of four inputs::

    effective = parent_authority ∧ delegation ∧ child_def ∧ policy

A capability is granted to the child only if **all four** grant it. Any single
denial removes the capability — a subagent can never hold more authority than
the narrowest of: what its parent holds, what the parent explicitly delegated,
what the child is defined to do, and what the ``PermissionPolicy`` permits. No
competitor harness proves this bound; it is the trust invariant of the
multi-agent tree.

``Authority`` is a frozen, hashable value object: :meth:`Authority.meet`
intersects capabilities and allowed-tools (unioning deny patterns, AND-ing
``can_spawn``). The root agent holds ``None`` (unrestricted — it is the trust
root); only spawned subagents carry a materialised ``Authority`` that the
``ToolRegistry`` enforces at the execution gate, on top of the existing
permission policy. The law is therefore enforced at a single chokepoint, not
scattered across tools.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid an import-time cycle: nullain.tools.__init__ imports this module,
    # so it must not import nullain.tools.* at module load. The PermissionPolicy
    # type is only needed for the from_policy projection, resolved lazily.
    from nullain.tools.permissions import PermissionPolicy


class Capability(StrEnum):
    """A coarse authority dimension a tool may require and a subagent may hold.

    Capabilities are intersected (not unioned) across delegation layers: a
    child holds a capability only if every layer in the chain grants it.
    """

    READ = "read"  # inspect state: read_file, grep, glob, git_status
    WRITE = "write"  # mutate workspace state: write_file, edit_file, git_commit
    EXEC = "exec"  # run an arbitrary subprocess: bash
    NETWORK = "network"  # reach an external endpoint: web_fetch
    SPAWN = "spawn"  # dispatch a further sub-agent


_ALL_CAPABILITIES: frozenset[Capability] = frozenset(Capability)


def _meet_tools(a: frozenset[str] | None, b: frozenset[str] | None) -> frozenset[str] | None:
    """Intersect two allowed-tool sets, treating ``None`` as the universe.

    ``None`` (any tool) is the identity element of the meet: intersecting the
    universe with a set yields the set, so an unrestricted factor never widens
    a restricted one.
    """
    if a is None:
        return b
    if b is None:
        return a
    return a & b


@dataclass(frozen=True)
class Authority:
    """A frozen authority bundle enforced for a sub-agent.

    Attributes:
        capabilities: Capabilities the holder may exercise. The meet of two
            authorities intersects these sets (a capability survives only if
            both grant it).
        allowed_tools: Names of tools the holder may call. ``None`` denotes the
            universe (any registered tool) and is the identity for meet; a set
            restricts the holder to those names.
        deny_patterns: Regex deny patterns (commands/paths). The meet UNIONS
            these — more denies is strictly tighter, never looser.
        can_spawn: Whether the holder may itself spawn sub-agents. The meet
            ANDs these (both must permit spawning).
    """

    capabilities: frozenset[Capability]
    allowed_tools: frozenset[str] | None
    deny_patterns: frozenset[str]
    can_spawn: bool

    @classmethod
    def unrestricted(cls) -> Authority:
        """The maximal authority held by the trust root before any narrowing."""
        return cls(
            capabilities=_ALL_CAPABILITIES,
            allowed_tools=None,
            deny_patterns=frozenset(),
            can_spawn=True,
        )

    @classmethod
    def only(
        cls,
        capabilities: Iterable[Capability],
        *,
        allowed_tools: frozenset[str] | None = None,
        deny_patterns: frozenset[str] | None = None,
        can_spawn: bool = False,
    ) -> Authority:
        """Convenience constructor for a deliberately narrowed authority.

        Defaults to ``can_spawn=False``: a narrowed subagent may not spawn
        further children unless the caller explicitly opts in. This matches the
        depth-cap discipline competitors apply (e.g. Grok's depth cap of 1).
        """
        return cls(
            capabilities=frozenset(capabilities),
            allowed_tools=allowed_tools,
            deny_patterns=frozenset(deny_patterns or ()),
            can_spawn=can_spawn,
        )

    def meet(self, other: Authority) -> Authority:
        """Intersect two authorities (greatest lower bound).

        ``capabilities`` and ``allowed_tools`` are intersected (a grant survives
        only if both grant it); ``deny_patterns`` are unioned (a deny from
        either tightens the result); ``can_spawn`` is the logical AND.
        """
        return Authority(
            capabilities=self.capabilities & other.capabilities,
            allowed_tools=_meet_tools(self.allowed_tools, other.allowed_tools),
            deny_patterns=self.deny_patterns | other.deny_patterns,
            can_spawn=self.can_spawn and other.can_spawn,
        )

    def permits(self, tool_name: str, requires: frozenset[Capability]) -> bool:
        """Whether this authority permits calling ``tool_name``.

        False if the tool is outside the allowed set, or if the tool requires
        any capability the holder lacks. Tools with empty ``requires`` (e.g.
        ``ask_user``) are governed solely by the allowed-tool set.
        """
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return False
        return requires <= self.capabilities

    @classmethod
    def from_policy(cls, policy: PermissionPolicy) -> Authority:
        """Project a ``PermissionPolicy`` into an authority factor for the meet.

        READ is always permitted; WRITE/EXEC are granted only when the policy
        does not default-deny them. The policy's deny patterns tighten the
        result via union at meet time. This is the ``policy`` operand of the
        intersection law — it folds the existing, tested permission engine into
        the capability lattice without duplicating its logic.
        """
        # Lazy import keeps this module free of a tools-package dependency at
        # import time (breaks the nullain.tools <-> nullain.authority cycle).
        from nullain.tools.permissions import PermissionLevel

        caps: set[Capability] = {Capability.READ}
        if policy.default_write_level != PermissionLevel.DENY:
            caps.add(Capability.WRITE)
        if policy.default_exec_level != PermissionLevel.DENY:
            caps.add(Capability.EXEC)
        return cls(
            capabilities=frozenset(caps),
            allowed_tools=None,
            deny_patterns=frozenset(policy.deny_patterns),
            can_spawn=True,
        )


__all__ = ["Authority", "Capability"]
