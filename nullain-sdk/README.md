# nullain-sdk

Core engine of the [Nullain Agent SDK](https://github.com/netty-linux/nullain-agent-sdk) —
a production-grade autonomous coding agent for Python, built on Hexagonal
Architecture and immutable Event Sourcing.

This package provides the agent loop, LLM provider adapters, model routing,
context management, memory, MCP/LSP clients, the permission/sandbox
security layer, and the plugin system. Built-in tools (`read_file`,
`bash`, `grep`, ...) live in the separate
[`nullain-tools`](https://pypi.org/project/nullain-tools/) package, which
`nullain-sdk` depends on.

## Install

```bash
pip install nullain-sdk
```

## Quick start

```python
import asyncio
from nullain import Agent


async def main() -> None:
    agent = Agent(workspace_root=".")
    result = await agent.run("list the python files in this workspace")
    print(result.final_text)


asyncio.run(main())
```

A synchronous facade is also available for scripts:

```python
from nullain import Agent

result = Agent(workspace_root=".").run_sync("say hello")
print(result.final_text)
```

`Agent` assembles safe defaults — an Ollama Cloud provider, the built-in
tool registry, fail-closed permission policy, and the platform sandbox — so
most callers never need to construct the lower-level `AgentLoop` directly.

## What's inside

- **`AgentLoop`** — the Plan/Act/Verify ReAct loop, with self-correction,
  loop detection, and adaptive step/token budgets.
- **`Conversation`** — immutable, append-only event sourcing with a
  deterministic fold; conversation state is always derivable from history.
- **`ModelRouter`** — task classification into `fast`/`balanced`/`deep`
  tiers with circuit breakers and fallback chains.
- **`ContextManager`** — automatic compaction at 75% context-window
  capacity, preserving active task state and instruction re-injection.
- **`PermissionPolicy` + `Sandbox`** — 3-tier command/file permissions
  (`allow`/`ask`/`deny`) plus an OS-level, fail-closed subprocess sandbox
  (Landlock on Linux, Seatbelt on macOS, Job Object on Windows).
- **`Authority`** — subagent capability intersection: a spawned subagent's
  effective authority is the meet of parent, delegation, definition, and
  policy — never wider than any one of them.
- **`PluginLoader`** — signed, capability-manifested, SBOM'd plugin bundles
  (Ed25519 verification via the optional `signing` extra).

## Documentation

The full docs, architecture diagrams, configuration reference, and CLI
usage live in the
[monorepo README and `docs/`](https://github.com/netty-linux/nullain-agent-sdk) —
this package is one piece of that whole; the CLI (`nullain`) and built-in
tools are what most users interact with day to day.

## License

MIT — see [LICENSE](https://github.com/netty-linux/nullain-agent-sdk/blob/master/nullain-sdk/LICENSE).
