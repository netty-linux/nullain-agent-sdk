# Quickstart

From zero to a running agent in a few minutes.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- An Ollama Cloud API key (set `OLLAMA_API_KEY`)

## Install

```bash
git clone https://github.com/netty-linux/nullain-agent-sdk.git
cd nullain-agent-sdk
uv sync
```

## The one-liner

The fastest way to run an agent is the CLI:

```bash
uv run nullain run "list the python files in this workspace"
```

This builds an `Agent` with safe defaults (Ollama provider, built-in tools,
fail-closed permissions, platform sandbox) and prints the result.

For structured output you can pipe:

```bash
uv run nullain run "list the python files" --json | jq '.type'
```

## Interactive chat

```bash
uv run nullain chat
```

Multi-turn session with streaming and TTY permission approval for `ASK`-level
tool calls.

## From Python

The `Agent` facade is the primary entry point:

```python
import asyncio
from nullain import Agent


async def main() -> None:
    agent = Agent(workspace_root=".")
    result = await agent.run("list the python files in this workspace")
    print(result.final_text)


if __name__ == "__main__":
    asyncio.run(main())
```

Synchronous scripts can use `run_sync`:

```python
from nullain import Agent

result = Agent(workspace_root=".").run_sync("say hello")
print(result.final_text)
```

## Streaming events

```python
import asyncio
from nullain import Agent, RunResult


async def main() -> None:
    agent = Agent(workspace_root=".")
    async for item in agent.stream("refactor the config loader"):
        if isinstance(item, RunResult):
            print(f"\nstatus={item.status} steps={item.steps}")
        else:
            print(f"[{item.event_type}]")


if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

By default the `Agent` loads `nullain.toml` from the current directory (or the
path in `NULLAIN_CONFIG`). See [configuration.md](configuration.md) for the full
reference. To point at a specific file:

```python
from nullain import Agent

agent = Agent.from_config("path/to/nullain.toml")
```

## Next steps

- [configuration.md](configuration.md) — every `nullain.toml` section.
- [tools.md](tools.md) — the built-in tools and their capabilities.
- [architecture.md](architecture.md) — how the pipeline fits together.
- [api-stability.md](api-stability.md) — what is public API under SemVer.
- `examples/` — runnable examples: `01_basic_agent.py` … `05_mcp_server.py`.
