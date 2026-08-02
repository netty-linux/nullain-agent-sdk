"""Nullain Agent SDK — Tool Registry and Execution Manager."""

from collections.abc import Awaitable, Callable
from typing import Any, cast

from nullain.errors import ToolNotFoundError, ToolPermissionError
from nullain.llm.types import ToolSpec
from nullain.tools.decorator import RegisteredTool
from nullain.tools.permissions import PermissionLevel, PermissionPolicy

# Async callback invoked when a tool action resolves to ASK. Receives the tool
# name and a human-readable description of the action; resolves True to grant.
PermissionCallback = Callable[[str, str], Awaitable[bool]]


class ToolRegistry:
    """Registry managing available tools, validation, and permissions."""

    def __init__(
        self,
        permission_policy: PermissionPolicy | None = None,
        permission_callback: PermissionCallback | None = None,
    ) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self.permission_policy = permission_policy
        # When ASK has no callback, execution is DENIED (fail-closed). This
        # prevents the previous behavior where ASK silently became ALLOW.
        self.permission_callback = permission_callback

    def register(self, tool_obj: RegisteredTool) -> None:
        """Register a RegisteredTool instance."""
        self._tools[tool_obj.name] = tool_obj

    def is_read_only(self, name: str) -> bool:
        """Whether a registered tool is marked side-effect-free (read-only).

        Unknown tools are treated as NOT read-only (fail-safe: a write tool
        must never accidentally be dispatched concurrently).
        """
        tool = self._tools.get(name)
        return bool(tool and tool.read_only)

    def get_tool(self, name: str) -> RegisteredTool:
        """Retrieve tool by name.

        Raises:
            ToolNotFoundError if tool is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' is not registered in ToolRegistry")
        return self._tools[name]

    def list_specs(self) -> list[ToolSpec]:
        """Get ToolSpec list for all registered tools."""
        return [t.spec for t in self._tools.values()]

    def _evaluate_permission(self, name: str, arguments: dict[str, Any]) -> PermissionLevel:
        """Resolve the permission level for a tool action under the current policy.

        Tools without an explicit policy rule (or with no policy configured)
        resolve to ALLOW. This preserves existing permissive behavior for
        unconfigured registries while keeping deny-pattern enforcement.
        """
        if self.permission_policy is None:
            return PermissionLevel.ALLOW

        if name == "bash" and "command_args" in arguments:
            raw_args = arguments["command_args"]
            if isinstance(raw_args, list):
                cmd_args = [str(x) for x in cast(list[Any], raw_args)]
                return self.permission_policy.evaluate_command(cmd_args)
            return PermissionLevel.ALLOW

        if name in ("write_file", "edit_file"):
            target_path = str(
                arguments.get("file_path")
                or arguments.get("target_file")
                or arguments.get("path")
                or ""
            )
            if target_path:
                return self.permission_policy.evaluate_file_access(target_path, is_write=True)
            return PermissionLevel.ALLOW

        if name == "read_file":
            target_path = str(arguments.get("file_path") or arguments.get("path") or "")
            if target_path:
                return self.permission_policy.evaluate_file_access(target_path, is_write=False)
            return PermissionLevel.ALLOW

        return PermissionLevel.ALLOW

    def _describe_request(self, name: str, arguments: dict[str, Any]) -> str:
        """Build a human-readable description of a tool action for ASK prompts."""
        if name == "bash":
            raw_args = arguments.get("command_args", [])
            if isinstance(raw_args, list):
                rendered = " ".join(str(x) for x in cast(list[Any], raw_args))
            else:
                rendered = str(raw_args)
            return f"Run shell command: {rendered}"
        if name in ("write_file", "edit_file"):
            target = (
                arguments.get("file_path")
                or arguments.get("target_file")
                or arguments.get("path")
                or ""
            )
            verb = "Write" if name == "write_file" else "Edit"
            return f"{verb} file: {target}"
        if name == "read_file":
            target = arguments.get("file_path") or arguments.get("path") or ""
            return f"Read file: {target}"
        return f"Execute tool: {name}"

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute registered tool by name with arguments dict.

        Permission resolution:
        - ALLOW: proceed.
        - DENY: raise ToolPermissionError.
        - ASK: invoke permission_callback; if none is configured, DENY
          (fail-closed) instead of silently allowing.

        Returns:
            String output of execution.
        """
        registered = self.get_tool(name)
        # A tool may carry a fixed permission level (e.g. MCP-backed tools)
        # that overrides the argument-based policy heuristics.
        if registered.permission_level is not None:
            level = registered.permission_level
        else:
            level = self._evaluate_permission(name, arguments)

        if level == PermissionLevel.DENY:
            raise ToolPermissionError(f"Tool '{name}' denied by permission policy")

        if level == PermissionLevel.ASK:
            if self.permission_callback is None:
                raise ToolPermissionError(
                    f"Tool '{name}' requires permission (ASK) but no permission callback is "
                    "configured — denied (fail-closed)"
                )
            description = self._describe_request(name, arguments)
            granted = await self.permission_callback(name, description)
            if not granted:
                raise ToolPermissionError(f"Tool '{name}' denied by user")

        res = await registered.execute(arguments)
        return str(res)


__all__ = ["PermissionCallback", "ToolRegistry"]
