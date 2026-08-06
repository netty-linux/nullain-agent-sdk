"""P4.24 — Subagent authority-intersection law.

Proves that a child subagent's effective authority is the meet (intersection) of

    effective = parent_authority ∧ delegation ∧ child_def ∧ policy

and that the ``ToolRegistry`` enforces it at the execution gate. The law is the
trust invariant no competitor harness proves: a subagent can never hold more
authority than the narrowest of the four factors, and a single denial removes a
capability outright (no ASK escape).
"""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from nullain import Authority, Capability, NullainError, ToolPermissionError
from nullain.agent import AgentLoop
from nullain.events import EventBus
from nullain.llm import CompletionChunk, CompletionRequest, LLMProvider, ToolCall
from nullain.tools import PermissionLevel, PermissionPolicy, ToolRegistry
from nullain_tools import register_default_tools


class _FakeProvider(LLMProvider):
    """Fake provider yielding a scripted response sequence."""

    def __init__(self, responses: list[CompletionChunk]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        if self.call_count < len(self.responses):
            chunk = self.responses[self.call_count]
            self.call_count += 1
            return chunk
        return CompletionChunk(delta_text="done")

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        yield await self.generate(request)

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Authority value object: meet is a true intersection
# ---------------------------------------------------------------------------


def test_meet_intersects_capabilities() -> None:
    a = Authority.only({Capability.READ, Capability.WRITE})
    b = Authority.only({Capability.READ, Capability.EXEC})
    met = a.meet(b)
    assert met.capabilities == {Capability.READ}
    # Idempotent / commutative greatest-lower-bound.
    assert a.meet(b) == b.meet(a)


def test_meet_universe_is_identity() -> None:
    narrowed = Authority.only({Capability.READ})
    assert narrowed.meet(Authority.unrestricted()) == narrowed
    assert Authority.unrestricted().meet(narrowed) == narrowed


def test_meet_intersects_allowed_tools_with_none_as_universe() -> None:
    a = Authority.only({Capability.READ}, allowed_tools=frozenset({"read_file", "write_file"}))
    b = Authority.only({Capability.READ}, allowed_tools=frozenset({"read_file"}))
    assert a.meet(b).allowed_tools == frozenset({"read_file"})
    # None (universe) does not widen a restricted set.
    universe = Authority.only({Capability.READ}, allowed_tools=None)
    assert a.meet(universe).allowed_tools == frozenset({"read_file", "write_file"})


def test_meet_unions_deny_patterns() -> None:
    a = Authority.only({Capability.READ}, deny_patterns=frozenset({r"rm\s+-rf"}))
    b = Authority.only({Capability.READ}, deny_patterns=frozenset({r"\.env"}))
    met = a.meet(b)
    assert met.deny_patterns == {r"rm\s+-rf", r"\.env"}


def test_meet_ands_can_spawn() -> None:
    a = Authority.only({Capability.SPAWN}, can_spawn=True)
    b = Authority.only({Capability.SPAWN}, can_spawn=False)
    assert a.meet(b).can_spawn is False
    assert a.meet(a).can_spawn is True


def test_permits_requires_capability_subset_and_allowed_tool() -> None:
    auth = Authority.only({Capability.READ}, allowed_tools=frozenset({"read_file"}))
    assert auth.permits("read_file", frozenset({Capability.READ}))
    # Missing capability.
    assert not auth.permits("write_file", frozenset({Capability.WRITE}))
    # Capability present but tool outside allowed set.
    assert not auth.permits("grep", frozenset({Capability.READ}))
    # Empty requires: governed solely by allowed set.
    assert auth.permits("read_file", frozenset())


def test_from_policy_projects_levels_and_denies() -> None:
    policy = PermissionPolicy(
        workspace_root=".",
        default_read_level=PermissionLevel.ALLOW,
        default_write_level=PermissionLevel.DENY,
        default_exec_level=PermissionLevel.ASK,
        deny_patterns=[r"rm\s+-rf"],
    )
    auth = Authority.from_policy(policy)
    assert Capability.READ in auth.capabilities
    assert Capability.EXEC in auth.capabilities
    assert Capability.WRITE not in auth.capabilities  # default-deny drops WRITE
    assert r"rm\s+-rf" in auth.deny_patterns


# ---------------------------------------------------------------------------
# ToolRegistry: capability surface + authority gate
# ---------------------------------------------------------------------------


def _registry(workspace: Path) -> ToolRegistry:
    reg = ToolRegistry(permission_policy=PermissionPolicy(workspace_root=str(workspace)))
    register_default_tools(reg, workspace)
    return reg


def test_capability_surface_unions_tool_requires(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    surface = reg.capability_surface()
    # Built-ins declare READ (read_file/grep/glob/...), WRITE (write_file/...),
    # EXEC (bash), NETWORK (web_fetch) — the surface is their union.
    assert Capability.READ in surface.capabilities
    assert Capability.WRITE in surface.capabilities
    assert Capability.EXEC in surface.capabilities
    assert Capability.NETWORK in surface.capabilities
    # allowed_tools is exactly the registered tool names (via public list_specs).
    names = {spec.function.name for spec in reg.list_specs()}
    assert surface.allowed_tools is not None
    assert surface.allowed_tools == names
    assert "write_file" in surface.allowed_tools


def test_scoped_shares_tools_but_sets_authority(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    auth = Authority.only({Capability.READ}, allowed_tools=frozenset({"read_file"}))
    child = reg.scoped(authority=auth, permission_policy=reg.permission_policy)
    assert child.authority == auth
    # Tools are shared (same specs), not narrowed by the scoped authority.
    assert child.list_specs() == reg.list_specs()
    # Parent is untouched.
    assert reg.authority is None


@pytest.mark.asyncio
async def test_registry_gate_denies_missing_capability(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    auth = Authority.only({Capability.READ})  # no WRITE
    child = reg.scoped(authority=auth, permission_policy=reg.permission_policy)
    with pytest.raises(ToolPermissionError, match="denied by authority") as exc_info:
        await child.execute("write_file", {"path": "x.txt", "content": "hi"})
    assert "write" in str(exc_info.value).lower()
    assert not (tmp_path / "x.txt").exists()


@pytest.mark.asyncio
async def test_registry_gate_denies_tool_outside_allowed_set(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    auth = Authority.only(
        {Capability.READ, Capability.EXEC}, allowed_tools=frozenset({"read_file"})
    )
    child = reg.scoped(authority=auth, permission_policy=reg.permission_policy)
    # bash requires EXEC (held) but is not in the allowed set -> denied.
    with pytest.raises(ToolPermissionError, match="denied by authority"):
        await child.execute("bash", {"command_args": ["echo", "hi"]})


@pytest.mark.asyncio
async def test_registry_gate_allows_permitted_call(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    (tmp_path / "readme.txt").write_text("hello world")
    auth = Authority.only({Capability.READ}, allowed_tools=frozenset({"read_file"}))
    child = reg.scoped(authority=auth, permission_policy=reg.permission_policy)
    out = await child.execute("read_file", {"path": "readme.txt"})
    # read_file (M8) returns numbered lines; the gate allows the call through.
    assert "hello world" in out


@pytest.mark.asyncio
async def test_root_registry_has_no_authority_gate(tmp_path: Path) -> None:
    """A registry with authority=None (the root) never applies the gate —
    backward compatible with pre-P4.24 callers."""
    reg = _registry(tmp_path)
    assert reg.authority is None
    (tmp_path / "readme.txt").write_text("ok")
    # write_file would be denied by an authority lacking WRITE; the root has no
    # authority so the call proceeds to the normal policy path (ASK -> denied
    # fail-closed without a callback, but NOT an authority denial).
    with pytest.raises(ToolPermissionError) as exc_info:
        await reg.execute("write_file", {"path": "readme.txt", "content": "x"})
    assert "authority" not in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# AgentLoop.spawn: the four-way intersection, end-to-end
# ---------------------------------------------------------------------------


def _spec_chunk() -> CompletionChunk:
    return CompletionChunk(
        delta_text=(
            '{"objective": "Subtask", "steps": ["reply"], '
            '"target_files": [], "acceptance_criteria": []}'
        )
    )


@pytest.mark.asyncio
async def test_spawn_read_only_delegation_blocks_write(tmp_path: Path) -> None:
    """The law, end-to-end: parent delegates READ-only authority; the child is
    given the full registry (which contains write_file, requiring WRITE); the
    child requests write_file; the intersection drops WRITE so the file is
    NEVER created — even though the child-def and policy both permit writes."""
    reg = _registry(tmp_path)
    write_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(
                id="w1", name="write_file", arguments={"path": "secret.txt", "content": "pwned"}
            )
        ]
    )
    provider = _FakeProvider([_spec_chunk(), write_chunk, CompletionChunk(delta_text="blocked")])

    parent = AgentLoop(provider=provider, tools=reg, event_bus=EventBus(), model="sub-model")
    read_only = Authority.only(
        {Capability.READ},
        allowed_tools=None,  # any tool by name, but capability-gated
        can_spawn=True,
    )

    text = await parent.spawn("Do the subtask", model="sub-model", authority=read_only)
    # The subagent ran to completion without raising...
    assert isinstance(text, str)
    # ...but the write was refused by the authority gate: no file materialised.
    assert not (tmp_path / "secret.txt").exists()


@pytest.mark.asyncio
async def test_spawn_clamps_delegation_to_parent_authority(tmp_path: Path) -> None:
    """A parent cannot delegate authority it does not hold: delegation is
    clamped to the parent's authority in the meet. Here the parent holds only
    READ (no WRITE) but tries to delegate {READ, WRITE}; the child's write is
    still blocked."""
    reg = _registry(tmp_path)
    write_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(id="w1", name="write_file", arguments={"path": "clamped.txt", "content": "x"})
        ]
    )
    provider = _FakeProvider([_spec_chunk(), write_chunk, CompletionChunk(delta_text="blocked")])

    parent = AgentLoop(
        provider=provider,
        tools=reg,
        event_bus=EventBus(),
        model="sub-model",
        authority=Authority.only({Capability.READ}, can_spawn=True),  # parent lacks WRITE
    )
    over_delegation = Authority.only({Capability.READ, Capability.WRITE}, can_spawn=True)

    await parent.spawn("Do the subtask", model="sub-model", authority=over_delegation)
    assert not (tmp_path / "clamped.txt").exists()


@pytest.mark.asyncio
async def test_spawn_refuses_without_spawn_capability(tmp_path: Path) -> None:
    """A subagent whose authority lacks the SPAWN capability cannot spawn a
    child — delegation depth is capped at the authority gate, not by convention."""
    reg = _registry(tmp_path)
    provider = _FakeProvider([])
    parent = AgentLoop(
        provider=provider,
        tools=reg,
        event_bus=EventBus(),
        model="sub-model",
        authority=Authority.only({Capability.READ}, can_spawn=False),
    )
    with pytest.raises(NullainError, match="SPAWN"):
        await parent.spawn("Do the subtask", authority=Authority.only({Capability.READ}))


@pytest.mark.asyncio
async def test_spawn_without_authority_is_backward_compatible(tmp_path: Path) -> None:
    """spawn with authority=None keeps the pre-P4.24 behavior: the child inherits
    an unrestricted view, no authority gate is applied, and a write succeeds
    (subject only to the normal policy — here an ALLOW-all policy)."""
    reg = ToolRegistry(
        permission_policy=PermissionPolicy(
            workspace_root=str(tmp_path),
            default_read_level=PermissionLevel.ALLOW,
            default_write_level=PermissionLevel.ALLOW,
            default_exec_level=PermissionLevel.ALLOW,
            deny_patterns=[],
        )
    )
    register_default_tools(reg, tmp_path)
    write_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(id="w1", name="write_file", arguments={"path": "free.txt", "content": "ok"})
        ]
    )
    provider = _FakeProvider([_spec_chunk(), write_chunk, CompletionChunk(delta_text="done")])
    parent = AgentLoop(provider=provider, tools=reg, event_bus=EventBus(), model="sub-model")

    await parent.spawn("Do the subtask", model="sub-model")  # authority=None
    assert (tmp_path / "free.txt").read_text() == "ok"


@pytest.mark.asyncio
async def test_spawn_policy_factor_drops_write_when_policy_denies(tmp_path: Path) -> None:
    """The ``policy`` operand: a policy that default-denies WRITE removes
    WRITE from the effective authority even when delegation grants it. Proves
    the fourth factor of the intersection is enforced, not just the first three."""
    reg = ToolRegistry(
        permission_policy=PermissionPolicy(
            workspace_root=str(tmp_path),
            default_read_level=PermissionLevel.ALLOW,
            default_write_level=PermissionLevel.DENY,  # policy denies WRITE
            default_exec_level=PermissionLevel.ASK,
            deny_patterns=[],
        )
    )
    register_default_tools(reg, tmp_path)
    write_chunk = CompletionChunk(
        tool_calls=[
            ToolCall(id="w1", name="write_file", arguments={"path": "policy.txt", "content": "x"})
        ]
    )
    provider = _FakeProvider([_spec_chunk(), write_chunk, CompletionChunk(delta_text="blocked")])
    parent = AgentLoop(provider=provider, tools=reg, event_bus=EventBus(), model="sub-model")
    # Delegation grants WRITE, but the policy factor must drop it.
    delegation = Authority.only({Capability.READ, Capability.WRITE}, can_spawn=True)

    await parent.spawn("Do the subtask", model="sub-model", authority=delegation)
    assert not (tmp_path / "policy.txt").exists()
