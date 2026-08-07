"""Nullain Tools — Bash command execution tool."""

from pathlib import Path

from nullain.authority import Capability
from nullain.tools import RegisteredTool, execute_subprocess, tool
from nullain.tools.result import ToolResult
from nullain.tools.sandbox import Sandbox, SandboxOptions


def create_bash_tool(
    workspace_root: str | Path,
    sandbox: Sandbox | None = None,
    sandbox_opts: SandboxOptions | None = None,
    timeout: float = 300.0,
) -> RegisteredTool:
    root = Path(workspace_root).resolve()

    @tool(
        name="bash",
        description=(
            "Execute a shell command specified as a list of argument strings in the workspace."
        ),
        requires=frozenset({Capability.EXEC}),
    )
    async def bash(command_args: list[str]) -> str | ToolResult:
        returncode, output = await execute_subprocess(
            cmd_args=command_args,
            cwd=root,
            timeout=timeout,
            sandbox=sandbox,
            sandbox_opts=sandbox_opts,
        )
        if returncode != 0:
            return ToolResult(
                output=f"Command exit code: {returncode}\nOutput:\n{output}",
                is_error=True,
                error_type="ToolError",
            )
        return output or "Command completed successfully with no output."

    return bash


__all__ = ["create_bash_tool"]
