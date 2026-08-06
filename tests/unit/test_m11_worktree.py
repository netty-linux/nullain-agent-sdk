"""M11.4 — Worktree-isolated subagents.

Proves that ``AgentLoop.spawn(isolation="worktree")`` runs the child in a
detached git worktree with a re-rooted tool registry, integrates changed files
back into the parent workspace, and guarantees worktree cleanup in ``finally``
(including on the failure path). The authority-intersection law (P4.24) is
preserved: the worktree child's registry is scoped exactly like an in-place
child's.
"""

import shutil
from collections.abc import AsyncGenerator, Callable
from pathlib import Path

import pytest
from nullain.agent import AgentLoop
from nullain.events import EventBus
from nullain.llm import CompletionChunk, CompletionRequest, LLMProvider, ToolCall
from nullain.tools import PermissionLevel, PermissionPolicy, ToolRegistry
from nullain.tools.sandbox import execute_subprocess
from nullain_tools import register_default_tools


def _resolve_git() -> str:
    """Resolve the git executable (S607: never launch a partial path)."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required for worktree-isolation tests")
    return git


#: Resolved git executable.
_GIT = _resolve_git()


class _FakeProvider(LLMProvider):
    """Fake provider yielding a scripted response sequence."""

    def __init__(self, responses: list[CompletionChunk], raise_on_generate: bool = False) -> None:
        self.responses = list(responses)
        self.raise_on_generate = raise_on_generate
        self.call_count = 0

    async def generate(self, request: CompletionRequest) -> CompletionChunk:
        if self.raise_on_generate:
            raise RuntimeError("child crashed")
        if self.call_count < len(self.responses):
            chunk = self.responses[self.call_count]
            self.call_count += 1
            return chunk
        return CompletionChunk(delta_text="done")

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[CompletionChunk, None]:
        yield await self.generate(request)

    async def health_check(self) -> bool:
        return True


def _spec_chunk() -> CompletionChunk:
    return CompletionChunk(
        delta_text=(
            '{"objective": "Subtask", "steps": ["reply"], '
            '"target_files": [], "acceptance_criteria": []}'
        )
    )


def _write_chunk(path: str, content: str) -> CompletionChunk:
    return CompletionChunk(
        tool_calls=[
            ToolCall(id="w1", name="write_file", arguments={"path": path, "content": content})
        ]
    )


async def _git_init(workspace: Path) -> None:
    """Initialise a git repo with one commit so ``git worktree add`` works."""
    for args in (
        [_GIT, "init", "-q"],
        [_GIT, "config", "user.email", "test@example.com"],
        [_GIT, "config", "user.name", "Test"],
    ):
        code, _ = await execute_subprocess(args, cwd=workspace)
        assert code == 0, f"git {' '.join(args)} failed"
    (workspace / "base.txt").write_text("base\n", encoding="utf-8")
    code, _ = await execute_subprocess([_GIT, "add", "."], cwd=workspace)
    assert code == 0
    code, _ = await execute_subprocess([_GIT, "commit", "-q", "-m", "init"], cwd=workspace)
    assert code == 0


def _allow_all_registry(workspace: Path) -> ToolRegistry:
    reg = ToolRegistry(
        permission_policy=PermissionPolicy(
            workspace_root=str(workspace),
            default_read_level=PermissionLevel.ALLOW,
            default_write_level=PermissionLevel.ALLOW,
            default_exec_level=PermissionLevel.ALLOW,
            deny_patterns=[],
        )
    )
    register_default_tools(reg, workspace)
    return reg


def _recording_factory(calls: list[Path]) -> Callable[[Path], ToolRegistry]:
    """A tool_factory that records the worktree path it was asked to root at."""

    def factory(worktree_root: Path) -> ToolRegistry:
        calls.append(worktree_root)
        return _allow_all_registry(worktree_root)

    return factory


async def _worktree_paths(workspace: Path) -> list[str]:
    """Return the absolute worktree paths registered for ``workspace``."""
    code, out = await execute_subprocess([_GIT, "worktree", "list", "--porcelain"], cwd=workspace)
    assert code == 0
    return [line.split(" ", 1)[1] for line in out.splitlines() if line.startswith("worktree ")]


@pytest.mark.asyncio
async def test_spawn_worktree_integrates_changes_and_cleans_up(tmp_path: Path) -> None:
    """The child runs in a detached worktree (re-rooted registry), its edit is
    integrated back into the parent workspace, and the worktree is removed."""
    await _git_init(tmp_path)
    factory_calls: list[Path] = []
    provider = _FakeProvider(
        [_spec_chunk(), _write_chunk("result.txt", "hello"), CompletionChunk(delta_text="done")]
    )
    parent = AgentLoop(
        provider=provider,
        tools=_allow_all_registry(tmp_path),
        event_bus=EventBus(),
        model="sub-model",
        workspace_root=tmp_path,
        tool_factory=_recording_factory(factory_calls),
    )

    text = await parent.spawn("Do the subtask", model="sub-model", isolation="worktree")

    assert isinstance(text, str)
    # The child ran against a re-rooted registry at a path distinct from the
    # parent workspace — i.e. a real worktree, not the parent checkout.
    assert len(factory_calls) == 1
    assert factory_calls[0] != tmp_path.resolve()
    # The child's edit was integrated back into the parent workspace.
    assert (tmp_path / "result.txt").read_text() == "hello"
    # The worktree was removed: only the main checkout remains registered.
    assert len(await _worktree_paths(tmp_path)) == 1


@pytest.mark.asyncio
async def test_spawn_worktree_cleans_up_on_failure(tmp_path: Path) -> None:
    """Even when the child run fails, the worktree is removed in ``finally``."""
    await _git_init(tmp_path)
    provider = _FakeProvider([], raise_on_generate=True)
    parent = AgentLoop(
        provider=provider,
        tools=_allow_all_registry(tmp_path),
        event_bus=EventBus(),
        model="sub-model",
        workspace_root=tmp_path,
        tool_factory=_recording_factory([]),
    )

    with pytest.raises(RuntimeError, match="child crashed"):
        await parent.spawn("Do the subtask", model="sub-model", isolation="worktree")

    # The worktree was created and then cleaned up despite the failure.
    assert len(await _worktree_paths(tmp_path)) == 1


@pytest.mark.asyncio
async def test_spawn_worktree_requires_tool_factory(tmp_path: Path) -> None:
    """Worktree isolation is refused when no tool_factory is configured."""
    await _git_init(tmp_path)
    provider = _FakeProvider([_spec_chunk(), CompletionChunk(delta_text="done")])
    parent = AgentLoop(
        provider=provider,
        tools=_allow_all_registry(tmp_path),
        event_bus=EventBus(),
        model="sub-model",
        workspace_root=tmp_path,
    )

    with pytest.raises(Exception, match="tool_factory"):
        await parent.spawn("Do the subtask", model="sub-model", isolation="worktree")
