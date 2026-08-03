"""Nullain Tools — Controlled Git workspace operations."""

from pathlib import Path

from nullain.authority import Capability
from nullain.tools import RegisteredTool, execute_subprocess, tool
from nullain.tools.sandbox import Sandbox, SandboxOptions


def create_git_tools(
    workspace_root: str | Path,
    sandbox: Sandbox | None = None,
    sandbox_opts: SandboxOptions | None = None,
) -> list[RegisteredTool]:
    root = Path(workspace_root).resolve()

    @tool(
        name="git_status",
        description="Get current git working directory status.",
        read_only=True,
        requires=frozenset({Capability.READ}),
    )
    async def git_status() -> str:
        _, output = await execute_subprocess(
            ["git", "status"], cwd=root, sandbox=sandbox, sandbox_opts=sandbox_opts
        )
        return output

    @tool(
        name="git_diff",
        description="Get current git uncommitted changes diff.",
        read_only=True,
        requires=frozenset({Capability.READ}),
    )
    async def git_diff() -> str:
        _, output = await execute_subprocess(
            ["git", "diff"], cwd=root, sandbox=sandbox, sandbox_opts=sandbox_opts
        )
        return output or "No uncommitted changes."

    @tool(
        name="git_commit",
        description="Stage specified or all changes and create a git commit.",
        requires=frozenset({Capability.WRITE}),
    )
    async def git_commit(message: str, files: list[str] | None = None) -> str:
        add_args = ["git", "add"] + (files if files else ["."])
        await execute_subprocess(add_args, cwd=root, sandbox=sandbox, sandbox_opts=sandbox_opts)
        ret, commit_output = await execute_subprocess(
            ["git", "commit", "-m", message], cwd=root, sandbox=sandbox, sandbox_opts=sandbox_opts
        )
        if ret != 0:
            return f"Git commit failed: {commit_output}"
        return f"Git commit successful:\n{commit_output}"

    return [git_status, git_diff, git_commit]


__all__ = ["create_git_tools"]
