"""Example 02 — Register a custom tool.

Shows how to define a tool with the ``@tool`` decorator and register it into a
registry that the Agent executes against.

Run:  uv run python examples/02_custom_tool.py
"""

import asyncio

from nullain import Agent, tool
from nullain.tools import ToolRegistry


@tool(
    name="shout",
    description="Uppercase the given text.",
    read_only=True,
)
def shout(text: str) -> str:
    """Return ``text`` uppercased."""
    return text.upper()


async def main() -> None:
    """Build an Agent with a custom tool and run a prompt."""
    registry = ToolRegistry()
    registry.register(shout)
    agent = Agent(workspace_root=".", tools=registry)
    result = await agent.run("use the shout tool on 'hello world'")
    print(f"status: {result.status}")
    print(f"output: {result.final_text}")


if __name__ == "__main__":
    asyncio.run(main())
