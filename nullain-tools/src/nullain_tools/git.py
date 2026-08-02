"""Nullain Tools — Controlled Git workspace operations."""

from pathlib import Path

from nullain.tools import RegisteredTool, execute_subprocess, tool


def create_git_tools(workspace_root: str | Path) -> list[RegisteredTool]:
    root = Path(workspace_root).resolve()

    @tool(name="git_status", description="Get current git working directory status.")
    async def git_status() -> str:
        _, output = await execute_subprocess(["git", "status"], cwd=root)
        return output

    @tool(name="git_diff", description="Get current git uncommitted changes diff.")
    async def git_diff() -> str:
        _, output = await execute_subprocess(["git", "diff"], cwd=root)
        return output or "No uncommitted changes."

    @tool(name="git_commit", description="Stage changes and create a git commit.")
    async def git_commit(message: str) -> str:
        await execute_subprocess(["git", "add", "."], cwd=root)
        ret, commit_output = await execute_subprocess(["git", "commit", "-m", message], cwd=root)
        if ret != 0:
            return f"Git commit failed: {commit_output}"
        return f"Git commit successful:\n{commit_output}"

    return [git_status, git_diff, git_commit]


__all__ = ["create_git_tools"]
