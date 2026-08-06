"""Example 03 — Subagent authority-intersection law.

A child subagent's effective authority is the meet of parent authority,
delegation, child definition, and policy. Here the parent delegates READ-only
authority, so a child that tries to write is refused by the authority gate.

Run:  uv run python examples/03_subagent_authority.py
"""

import asyncio
from pathlib import Path

from nullain import AgentLoop, Authority, Capability
from nullain.events import EventBus
from nullain.llm import OllamaCloudProvider
from nullain.tools import ToolRegistry
from nullain_tools import register_default_tools


async def main() -> None:
    """Spawn a READ-only subagent and show that writes are blocked."""
    registry = ToolRegistry()
    register_default_tools(registry, workspace_root=".")
    provider = OllamaCloudProvider()
    parent = AgentLoop(provider=provider, tools=registry, event_bus=EventBus())

    read_only = Authority.only({Capability.READ}, can_spawn=True)
    # The subagent is given the full registry (which contains write_file), but
    # the intersection drops WRITE, so any write attempt is refused.
    text = await parent.spawn("create a file called secret.txt", authority=read_only)
    print(f"subagent output: {text}")
    print("secret.txt created:", Path("secret.txt").exists())


if __name__ == "__main__":
    asyncio.run(main())
