<div align="center">

<img src="AI-AGENT-SDK-CHIBI-TRANSPARENTE.png" alt="Nullain Agent SDK Logo" width="240" />

# Nullain Agent SDK

**The Production-Grade Autonomous Agent Engine & SDK for Python**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Hexagonal](https://img.shields.io/badge/architecture-Hexagonal-purple.svg)](https://github.com/netty-linux/nullain-agent-sdk)
[![Type Checker: Pyright Strict](https://img.shields.io/badge/pyright-strict-green.svg)](https://github.com/microsoft/pyright)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*Built for high-reliability software engineering workflows. Consumed by the Go CLI via zero-drift NDJSON protocol over stdio.*

[Overview](#-overview) • [Core Features](#-core-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Protocol & Daemon](#-protocol--daemon) • [Development](#-development)

---

</div>

## 📌 Overview

**Nullain Agent SDK** (`nullain`) is a production-grade agentic engine designed to reason, execute code, self-correct, and learn from experience across software development tasks.

Built on **Hexagonal Architecture** and **Event Sourcing**, `nullain` enforces deterministic execution, strict boundaries against prompt injection, and intelligent LLM routing across Ollama Cloud models.

---

## ✨ Core Features

- 🧠 **Tiered Model Routing (`ModelRouter`)**
  Intelligent task classification routing requests to `fast` (`gpt-oss:20b`), `balanced` (`qwen3-coder:480b-cloud`), or `deep` (`deepseek-v4-pro`) tiers with automated circuit breakers and fallback chains.
- 📜 **Immutable Event Sourcing (`Conversation`)**
  All interaction history is stored as frozen, append-only Pydantic event sequences. Conversation state is derived via deterministic fold—enabling replayability, zero-drift persistence, and trajectory audits.
- 📐 **Plan/Act Hybrid Loop (`AgentLoop`)**
  Structured execution pipeline featuring task spec generation, strict validation (`SpecValidator`), human approval gates, and a verification phase (`VERIFY`) with self-correction on test or linter failures.
- 🛡️ **Context Compaction & Instruction Centrifugation (`ContextManager`)**
  Prevents context rot and degradation. Automatically compacts history at 75% window capacity while preserving active specs, key decisions, and diagnostics, paired with instruction re-injection and progressive tool disclosure.
- 🧬 **Episodic Memory & Learning Loop (`EpisodicMemory`)**
  SQLite-backed trajectory engine that records execution attempts, success metrics, and repository fingerprints—injecting relevant few-shot examples into future tasks.
- 🔒 **Defense-in-Depth Security & Sandboxing (`PermissionPolicy` + `Sandbox`)**
  Subprocess execution strictly via explicit argument lists (no `shell=True`), strict path resolution (`resolve()` + `is_relative_to`), and 3-tier action permissions (`allow`, `ask`, `deny`). On top of that, an **OS-level fail-closed sandbox** (`Landlock` on Linux ≥5.13, `Seatbelt` on macOS, Job Object on Windows) isolates filesystem + network for subprocess tools — if a sandbox is required and unavailable, execution is **refused**, never run unsandboxed.
- 🪪 **Subagent Authority-Intersection Law (`Authority`)**
  A child subagent's effective authority is the **meet** (intersection) of four factors — parent authority ∧ delegation ∧ child definition ∧ policy. A capability is granted only if all four grant it; any single denial removes it outright (no ASK escape). `AgentLoop.spawn` materialises the bound onto a scoped child registry.
- 📦 **Signed Plugins + SBOM + Capability Manifests (`PluginLoader`)**
  Plugins (v1: MCP server bundles) are signed, capability-manifested bundles. A signature covers identity + transport command + capabilities + tool declarations + a content-hashed SBOM, so drift in any of them invalidates the signature. The loader verifies, intersects declared capabilities with the operator's grant, and registers tools — **fail-closed at every branch** (Ed25519 via the optional `signing` extra).
- 🔎 **Tool Search & Deferred MCP Schemas**
  MCP tools register with minimal metadata (name + description); full input schemas are **deferred** and hydrated on demand. The agent discovers tools via a `search_tools` tool, and only the schemas of the tools it actually selects are loaded into the LLM context — keeping the prompt small as the tool surface scales.
- 🧩 **Deterministic Workflow Orchestrator (`Workflow`)**
  A workflow is a Python function that orchestrates subagents deterministically — which subagents run, in what order, with what fan-out and pipeline stages is fixed by the script, never decided by an LLM. `agent()`, `parallel()` (barrier), and `pipeline()` (no barrier) compose into auditable, resumable, testable multi-agent tasks.
- ⚡ **Zero-Drift Stdio NDJSON Daemon (`nullain-agentd`)**
  Exposes a typed stdio NDJSON protocol with automated JSON Schema generation (`make schema`), allowing Go, Rust, or CLI wrappers to consume the SDK natively.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│  PUBLIC API                                                │
│  AgentLoop · Conversation · OllamaCloudProvider · Workflow │
├────────────────────────────────────────────────────────────┤
│  ORCHESTRATION & HARNESS                                   │
│  AgentLoop (ReAct + Plan/Act) · Workflow (subagent DSL)    │
│  IntentParser · SpecValidator · Reflection/Self-correction │
├────────────────────────────────────────────────────────────┤
│  CONTEXT & MEMORY                                          │
│  ContextManager (compaction, instruction centrifuging)     │
│  EpisodicMemory · PersistentMemory · TrajectoryRecord      │
├────────────────────────────────────────────────────────────┤
│  MODEL ROUTING                                             │
│  ModelRouter (task → tier → model) · CircuitBreaker        │
│  LLMProvider (Port) ← OllamaCloudProvider (Adapter)        │
├────────────────────────────────────────────────────────────┤
│  TOOLS & EXECUTION                                         │
│  ToolRegistry · Tool Search (deferred schemas) · MCPClient │
│  PermissionPolicy · Sandbox (fail-closed) · Authority gate │
├────────────────────────────────────────────────────────────┤
│  PLUGINS & TRUST                                           │
│  PluginLoader · PluginManifest · SignatureVerifier · SBOM  │
├────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE & TRANSPORT                                │
│  EventBus · Telemetry (structlog) · Stdio NDJSON Protocol  │
└────────────────────────────────────────────────────────────┘
```

---

## 📦 Workspace Package Structure

The repository is organized as a high-efficiency `uv` monorepo:

- **`nullain-sdk/`**: Core SDK engine — LLM adapters, event bus, router, context manager, memory, MCP client, authority, plugins, sandbox, and the workflow orchestrator.
- **`nullain-tools/`**: Built-in developer tools (`read_file`, `write_file`, `edit_file`, `bash`, `grep`, `git`, `web_fetch`, `ask_user`, `search_tools`, memory tools).
- **`nullain-agentd/`**: High-performance daemon reading/writing NDJSON over stdio for CLI integration.
- **`schema/`**: Exported JSON Schema contracts for cross-language interoperability.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- [**uv**](https://github.com/astral-sh/uv) (fast Python package installer)

### Setup

```bash
# Clone the repository
git clone https://github.com/netty-linux/nullain-agent-sdk.git
cd nullain-agent-sdk

# Sync virtual environment & dependencies
uv sync
```

### CLI

```bash
# Run a single prompt
uv run nullain run "list the python files in this workspace"

# Structured NDJSON output for piping
uv run nullain run "list the python files" --json

# Interactive multi-turn chat with TTY permission approval
uv run nullain chat

# Environment health checks
uv run nullain doctor

# Manage MCP servers declared in nullain.toml
uv run nullain mcp list
```

### Python SDK Usage Example

The `Agent` facade is the primary entry point — it assembles the provider,
tools, permission policy, router, and sandbox with safe defaults:

```python
import asyncio
from nullain import Agent


async def main():
    agent = Agent(workspace_root=".")
    result = await agent.run("Audit pyproject.toml and ensure all dependencies are up to date")
    print("Agent Execution Result:", result.final_text)


if __name__ == "__main__":
    asyncio.run(main())
```

For scripts, `run_sync` is a thin synchronous facade:

```python
from nullain import Agent

result = Agent(workspace_root=".").run_sync("say hello")
print(result.final_text)
```

### Documentation & Examples

- [docs/quickstart.md](docs/quickstart.md) — from zero to a running agent.
- [docs/configuration.md](docs/configuration.md) — every `nullain.toml` section.
- [docs/tools.md](docs/tools.md) — the built-in tools and their capabilities.
- [docs/architecture.md](docs/architecture.md) — the pipeline and layers.
- [docs/api-stability.md](docs/api-stability.md) — what is public API under SemVer.
- `examples/` — runnable examples: `01_basic_agent.py` … `05_mcp_server.py`.

---

## 🔄 Protocol & Daemon

`nullain-agentd` serves as the stdio IPC bridge between the SDK and host processes (such as the Go CLI).

### Running the Daemon

```bash
uv run python -m nullain_agentd.main
```

### Protocol Envelope Structure

```json
{
  "v": 1,
  "type": "user.message",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "session_id": "sess_100",
    "prompt": "Create a test suite for the memory module"
  }
}
```

To update the JSON Schema exported to `schema/protocol_v1.json`:

```bash
make schema
```

---

## 🧪 Development & Quality Assurance

Quality is enforced through strict static analysis, complete type safety, and automated test coverage.

```bash
# Run all quality checks (lint + typecheck + tests)
make check

# Run test suite
make test

# Run linter checks
make lint

# Run Pyright strict type check
make typecheck

# Format code automatically
make format

# Audit dependencies for known vulnerabilities
make audit

# Regenerate the exported JSON Schema (schema/protocol_v1.json)
make schema
```

> **Plugin signing (optional).** Ed25519 signature verification requires the
> `signing` extra: `uv sync --extra signing` (installs `cryptography`). Without
> it, the SDK installs cleanly and a signed plugin is refused fail-closed rather
> than loaded on trust.

---

## 🔒 Security & Threat Model

1. **Subprocess Isolation**: Zero reliance on `shell=True`. Commands run via argument array execution with timeouts and output truncation.
2. **Fail-Closed OS Sandbox**: Subprocess tools run inside an OS-level sandbox (`Landlock` on Linux ≥5.13, `Seatbelt` on macOS, Job Object on Windows) isolating filesystem + network. If a sandbox is required and unavailable, execution is **refused** — never run unsandboxed.
3. **Workspace Containment**: File operations enforce absolute path resolution verified against the `workspace_root`. Symlinks are resolved prior to authorization checks.
4. **Subagent Authority-Intersection**: A child subagent's effective authority is the meet of parent ∧ delegation ∧ child definition ∧ policy — a capability is granted only if all four grant it, with no ASK escape from the bound.
5. **Signed Plugins & SBOM**: Plugins are signed, capability-manifested bundles; the signature covers identity + transport + capabilities + tool declarations + a content-hashed SBOM, and the loader is fail-closed at every branch.
6. **Secret Redaction**: Automatic redaction patterns prevent API keys and credentials from entering logs or LLM context prompts.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
