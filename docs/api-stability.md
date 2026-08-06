# API Stability Policy

This document defines what is under SemVer in the Nullain Agent SDK and what is
internal. It exists because the top-level `__all__` previously exported 106
symbols, freezing the entire internal surface under SemVer (AGENTS.md rule 11:
"minimal public API").

## The rule

**Only the symbols listed in `nullain/__init__.py::__all__` are public API.**
They are covered by SemVer: a breaking change to any of them requires a major
version bump.

**Everything else is internal.** It remains importable by its full module path
(e.g. `from nullain.plugins import PluginLoader`, `from nullain.mcp import
MCPClient`) for advanced users, but it is **not** part of the stable surface
and may change without notice.

## What is public (the `__all__` surface)

The public surface is deliberately small (~38 symbols) and grouped:

- **Facade + loop:** `Agent`, `AgentLoop`, `RunResult`, `RunStatus`
- **Conversation + events:** `Conversation`, `EventBus`, `BaseEvent`,
  `UserMessageEvent`, `ToolCallEvent`, `ToolResultEvent`, `ModelResponseEvent`,
  `StreamDeltaEvent`, `ErrorEvent`, `CompactionEvent`, `SpecCreatedEvent`,
  `SpecVerifiedEvent`
- **Tools:** `tool`, `ToolRegistry`
- **Providers:** `LLMProvider`, `OllamaCloudProvider`
- **Config:** `NullainSettings`, `load_settings`
- **Error hierarchy (top level):** `NullainError`, `ToolError`,
  `ToolPermissionError`, `ToolNotFoundError`, `ToolExecutionError`,
  `ProviderError`, `ProviderTimeoutError`, `ProviderRateLimitError`,
  `ProviderAuthenticationError`, `NoModelAvailableError`, `ContextError`,
  `BudgetExceededError`, `RouterError`, `MCPError`, `SpecValidationError`,
  `PluginError`

## What is internal (importable, not stable)

Everything else that was previously in `__all__` — the MCP client classes,
plugin loader, workflow orchestrator, router internals, memory stores, protocol
payloads, telemetry helpers, and the remaining error subclasses — moved out of
the top-level surface. They are still reachable by their full module path.

## How to add to the public API

Adding a symbol to `__all__` is a deliberate act. Before doing so, ask:

1. Is this something a consumer of the SDK needs to name directly?
2. Is its signature stable enough to freeze under SemVer?
3. Does it have a docstring and pass `pyright` strict?

If the answer to any is "no", keep it internal and importable by full path
instead.

## How to remove from the public API

Removing a symbol from `__all__` is a **non-breaking** change: the import
statement stays in `__init__.py`, so `from nullain import X` still works; only
`from nullain import *` and the documented surface stop including it. This is
how the surface was reduced from 106 to ~38 without breaking existing imports.
