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
- 🔒 **Defense-in-Depth Security & Sandboxing (`PermissionPolicy`)**
  Subprocess execution strictly via explicit argument lists (no `shell=True`), strict path resolution (`resolve()` + `is_relative_to`), and 3-tier action permissions (`allow`, `ask`, `deny`).
- ⚡ **Zero-Drift Stdio NDJSON Daemon (`nullain-agentd`)**
  Exposes a typed stdio NDJSON protocol with automated JSON Schema generation (`make schema`), allowing Go, Rust, or CLI wrappers to consume the SDK natively.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│  PUBLIC API                                                │
│  AgentLoop · Conversation · OllamaCloudProvider            │
├────────────────────────────────────────────────────────────┤
│  ORCHESTRATION & HARNESS                                   │
│  AgentLoop (ReAct + Plan/Act) · IntentParser               │
│  SpecValidator · Reflection/Self-correction                │
├────────────────────────────────────────────────────────────┤
│  CONTEXT & MEMORY                                          │
│  ContextManager (compaction, instruction centrifuging)     │
│  EpisodicMemory · TrajectoryRecord (SQLite)                │
├────────────────────────────────────────────────────────────┤
│  MODEL ROUTING                                             │
│  ModelRouter (task → tier → model) · CircuitBreaker        │
│  LLMProvider (Port) ← OllamaCloudProvider (Adapter)        │
├────────────────────────────────────────────────────────────┤
│  TOOLS & EXECUTION                                         │
│  ToolRegistry · PermissionPolicy · Sandboxed Execution     │
├────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE & TRANSPORT                                │
│  EventBus · Telemetry (structlog) · Stdio NDJSON Protocol  │
└────────────────────────────────────────────────────────────┘
```

---

## 📦 Workspace Package Structure

The repository is organized as a high-efficiency `uv` monorepo:

- **`nullain-sdk/`**: Core SDK engine containing LLM adapters, event bus, router, context manager, and memory storage.
- **`nullain-tools/`**: Built-in developer tools (`read_file`, `write_file`, `edit_file`, `bash`, `grep`, `git`).
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

### Python SDK Usage Example

```python
import asyncio
from nullain.agent import AgentLoop
from nullain.llm import OllamaCloudProvider
from nullain.tools import ToolRegistry
from nullain_tools import register_default_tools

async def main():
    # Initialize tools & provider
    registry = ToolRegistry()
    register_default_tools(registry, workspace_root=".")
    provider = OllamaCloudProvider()

    # Create agent loop instance
    agent = AgentLoop(provider=provider, tools=registry)

    # Run agent task
    output = await agent.run(
        prompt="Audit pyproject.toml and ensure all dependencies are up to date",
        session_id="session-001"
    )
    print("Agent Execution Result:", output)

if __name__ == "__main__":
    asyncio.run(main())
```

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
```

---

## 🔒 Security & Threat Model

1. **Subprocess Isolation**: Zero reliance on `shell=True`. Commands run via argument array execution with timeouts and output truncation.
2. **Workspace Containment**: File operations enforce absolute path resolution verified against the `workspace_root`. Symlinks are resolved prior to authorization checks.
3. **Secret Redaction**: Automatic redaction patterns prevent API keys and credentials from entering logs or LLM context prompts.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
