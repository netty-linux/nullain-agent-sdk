"""Example 01 — Basic agent with the Agent facade.

Run:  uv run python examples/01_basic_agent.py
"""

import asyncio

from nullain import Agent


async def main() -> None:
    """Run a single prompt through the Agent facade and print the result."""
    agent = Agent(workspace_root=".")
    result = await agent.run("list the python files in this workspace")
    print(f"status: {result.status}")
    print(f"steps:  {result.steps}")
    print(f"output: {result.final_text}")


if __name__ == "__main__":
    asyncio.run(main())
