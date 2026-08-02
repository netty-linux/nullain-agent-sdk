"""Nullain Agent SDK — @tool Decorator for Auto Schema Extraction."""

import inspect
from collections.abc import Callable
from typing import Any

from nullain.llm.types import FunctionSpec, ToolSpec
from nullain.tools.permissions import PermissionLevel


class RegisteredTool:
    """Wrapper holding tool execution function and its extracted ToolSpec."""

    def __init__(
        self,
        name: str,
        description: str,
        spec: ToolSpec,
        func: Callable[..., Any],
        read_only: bool = False,
        permission_level: PermissionLevel | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.spec = spec
        self.func = func
        self.is_async = inspect.iscoroutinefunction(func)
        # Read-only tools (no side effects: read_file, grep, glob, ...) can be
        # dispatched concurrently when the model requests several at once. The
        # Act loop uses this to run a batch of read-only calls via gather.
        self.read_only = read_only
        # Optional fixed permission level that overrides the registry's
        # heuristic policy resolution. Used by MCP-backed tools, whose
        # side-effects cannot be introspected from arguments: a server is
        # either trusted (ALLOW) or gated behind the human approval loop (ASK).
        # None means "use the registry's per-tool policy logic".
        self.permission_level = permission_level

    async def execute(self, kwargs: dict[str, Any]) -> Any:
        """Execute tool function handling both sync and async functions."""
        if self.is_async:
            return await self.func(**kwargs)
        return self.func(**kwargs)


def tool(
    name: str | None = None,
    description: str | None = None,
    read_only: bool = False,
    permission_level: PermissionLevel | None = None,
) -> Callable[[Callable[..., Any]], RegisteredTool]:
    """Decorator to register a python function as an agent Tool with JSON Schema auto-derivation.

    Args:
        name: Override tool name (defaults to the function name).
        description: Override tool description (defaults to the docstring).
        read_only: Mark the tool as side-effect-free so the Act loop may
            dispatch it concurrently with other read-only tool calls.
        permission_level: Optional fixed permission level overriding the
            registry's policy heuristics (used for MCP-backed tools).
    """

    def decorator(func: Callable[..., Any]) -> RegisteredTool:
        tool_name = name or func.__name__
        tool_doc = description or (func.__doc__ or "").strip() or f"Tool {tool_name}"

        sig = inspect.signature(func)
        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            default_val = ... if param.default == inspect.Parameter.empty else param.default
            fields[param_name] = (param_type, default_val)

        from pydantic import create_model

        model_cls = create_model(f"{tool_name}_Args", **fields)
        schema = model_cls.model_json_schema()

        # Clean JSON Schema title/description keys
        schema.pop("title", None)

        tool_spec = ToolSpec(
            type="function",
            function=FunctionSpec(
                name=tool_name,
                description=tool_doc,
                parameters=schema,
            ),
        )

        return RegisteredTool(
            name=tool_name,
            description=tool_doc,
            spec=tool_spec,
            func=func,
            read_only=read_only,
            permission_level=permission_level,
        )

    return decorator


__all__ = ["RegisteredTool", "tool"]
