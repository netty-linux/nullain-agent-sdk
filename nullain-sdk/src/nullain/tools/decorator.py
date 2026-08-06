"""Nullain Agent SDK — @tool Decorator for Auto Schema Extraction."""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from nullain.authority import Capability
from nullain.llm.types import FunctionSpec, ToolSpec
from nullain.tools.permissions import PermissionLevel

#: Loads a tool's full JSON input schema on demand (P4.26 deferred schemas).
#: Returns the ``parameters`` dict for the tool's ``FunctionSpec``.
SchemaLoader = Callable[[], Awaitable[dict[str, Any]]]


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
        requires: frozenset[Capability] = frozenset(),
        schema_loader: SchemaLoader | None = None,
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
        # Capabilities the tool exercises (P4.24 authority-intersection law).
        # The registry's authority gate refuses a call when the calling
        # subagent's authority does not include every required capability.
        # Empty for capability-neutral tools (e.g. ask_user), which are then
        # governed solely by the allowed-tool set.
        self.requires = frozenset(requires)
        # Deferred schema (P4.26): when set, the full input schema is not
        # eagerly present — ``spec.function.parameters`` starts empty and is
        # populated on demand by :meth:`hydrate_schema`. A deferred tool is
        # hidden from the LLM's tool list until hydrated (see
        # ``ToolRegistry.list_specs``), so the model only ever sees schemas for
        # tools it has chosen to use. None = eager schema (the default).
        self._schema_loader = schema_loader
        self._hydrated = schema_loader is None

    @property
    def is_deferred(self) -> bool:
        """Whether this tool's schema is loaded lazily (P4.26)."""
        return self._schema_loader is not None

    @property
    def is_hydrated(self) -> bool:
        """Whether this tool's schema has been loaded (eager tools are always hydrated)."""
        return self._hydrated

    async def hydrate_schema(self) -> None:
        """Load the deferred schema into ``spec.function.parameters`` (idempotent).

        No-op for non-deferred tools and for deferred tools already hydrated.
        """
        if self._hydrated or self._schema_loader is None:
            return
        self.spec.function.parameters = await self._schema_loader()
        self._hydrated = True

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
    requires: frozenset[Capability] = frozenset(),
) -> Callable[[Callable[..., Any]], RegisteredTool]:
    """Decorator to register a python function as an agent Tool with JSON Schema auto-derivation.

    Args:
        name: Override tool name (defaults to the function name).
        description: Override tool description (defaults to the docstring).
        read_only: Mark the tool as side-effect-free so the Act loop may
            dispatch it concurrently with other read-only tool calls.
        permission_level: Optional fixed permission level overriding the
            registry's policy heuristics (used for MCP-backed tools).
        requires: Capabilities the tool exercises, enforced by the registry's
            authority gate against a subagent's effective authority (P4.24).
            Defaults to empty (capability-neutral).
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
            requires=requires,
        )

    return decorator


__all__ = ["RegisteredTool", "tool"]
