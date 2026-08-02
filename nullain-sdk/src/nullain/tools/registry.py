"""Nullain Agent SDK — Tool Registry and Execution Manager."""

from typing import Any, cast

from nullain.errors import ToolNotFoundError, ToolPermissionError
from nullain.llm.types import ToolSpec
from nullain.tools.decorator import RegisteredTool
from nullain.tools.permissions import PermissionLevel, PermissionPolicy


class ToolRegistry:
    """Registry managing available tools, validation, and permissions."""

    def __init__(self, permission_policy: PermissionPolicy | None = None) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self.permission_policy = permission_policy

    def register(self, tool_obj: RegisteredTool) -> None:
        """Register a RegisteredTool instance."""
        self._tools[tool_obj.name] = tool_obj

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

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute registered tool by name with arguments dict.

        Returns:
            String output of execution.
        """
        registered = self.get_tool(name)

        if self.permission_policy is not None and name == "bash" and "command_args" in arguments:
            raw_args = arguments["command_args"]
            if isinstance(raw_args, list):
                cmd_args = [str(x) for x in cast(list[Any], raw_args)]
                level = self.permission_policy.evaluate_command(cmd_args)
                if level == PermissionLevel.DENY:
                    raise ToolPermissionError(
                        f"Execution of command args {cmd_args} denied by policy"
                    )

        res = await registered.execute(arguments)
        return str(res)


__all__ = ["ToolRegistry"]
