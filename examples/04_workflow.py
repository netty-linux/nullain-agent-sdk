"""Example 04 — Deterministic workflow orchestrator.

A workflow is a Python function that orchestrates subagents deterministically:
which subagents run, in what order, and with what fan-out is fixed by the
script, never decided by an LLM.

Run:  uv run python examples/04_workflow.py
"""

import asyncio

from nullain import AgentLoop
from nullain.events import EventBus
from nullain.llm import OllamaCloudProvider
from nullain.tools import ToolRegistry
from nullain.workflow import Workflow, WorkflowContext, loop_spawner
from nullain_tools import register_default_tools


async def research_and_summarize(ctx: WorkflowContext) -> str:
    """Run two subagents in parallel, then a third that summarizes."""
    await ctx.phase("research")
    results = await ctx.parallel(
        [
            lambda: ctx.agent("list the python files in this workspace", label="scout"),
            lambda: ctx.agent("count the lines of code in the SDK", label="counter"),
        ]
    )
    await ctx.phase("summarize")
    summary = await ctx.agent(f"summarize these findings: {results}", label="summarizer")
    return summary


workflow = Workflow(
    name="research",
    description="Scout the workspace and summarize findings.",
    fn=research_and_summarize,
)


async def main() -> None:
    """Run the workflow against a real agent loop."""
    registry = ToolRegistry()
    register_default_tools(registry, workspace_root=".")
    provider = OllamaCloudProvider()
    loop = AgentLoop(provider=provider, tools=registry, event_bus=EventBus())
    result = await workflow.run(loop_spawner(loop), event_bus=EventBus())
    print(f"workflow result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
