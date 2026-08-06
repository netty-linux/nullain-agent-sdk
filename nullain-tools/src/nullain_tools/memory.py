"""Nullain Tools — persistent memory tools (save_memory / read_memory).

Backed by :class:`nullain.memory.PersistentMemory`. The agent uses these to
record durable facts (user preferences, project constraints, feedback) that
are re-injected into the system prompt and survive context compaction.
"""

from nullain.authority import Capability
from nullain.memory import MemoryEntry, MemoryType, PersistentMemory
from nullain.tools import RegisteredTool, tool
from nullain.tools.result import ToolResult


def create_memory_tools(persistent_memory: PersistentMemory) -> list[RegisteredTool]:
    """Build save_memory + read_memory tools bound to a PersistentMemory."""
    pm = persistent_memory

    @tool(
        name="save_memory",
        description=(
            "Persist a durable fact to memory. Use for user preferences, project "
            "constraints, or feedback worth recalling in future runs. `name` is a "
            "lowercase kebab-case slug; `type` is user|feedback|project|reference."
        ),
        read_only=False,
        requires=frozenset({Capability.WRITE}),
    )
    async def save_memory(
        name: str,
        description: str,
        body: str,
        memory_type: str = "reference",
    ) -> str | ToolResult:
        try:
            mtype = MemoryType(memory_type)
        except ValueError:
            return (
                f"Error: invalid memory type '{memory_type}'. "
                "Use one of: user, feedback, project, reference."
            )
        try:
            pm.write(MemoryEntry(name=name, description=description, body=body, type=mtype))
        except ValueError as err:
            return ToolResult(
                output=f"Error: {err}",
                is_error=True,
                error_type="ToolError",
            )
        return f"Saved memory '{name}' ({mtype.value})."

    @tool(
        name="read_memory",
        description=(
            "Read the body of a persisted memory entry by its slug name, or list "
            "all stored memory slugs when name is omitted."
        ),
        read_only=True,
        requires=frozenset({Capability.READ}),
    )
    async def read_memory(name: str = "") -> str | ToolResult:
        if not name:
            slugs = pm.list_entries()
            if not slugs:
                return "No memories stored."
            return "Stored memories:\n" + "\n".join(f"- {s}" for s in slugs)
        body = pm.read(name)
        if body is None:
            return ToolResult(
                output=f"Error: no memory named '{name}'.",
                is_error=True,
                error_type="ToolError",
            )
        return body

    return [save_memory, read_memory]


__all__ = ["create_memory_tools"]
