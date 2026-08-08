# Architecture

Nullain is built on **Hexagonal Architecture** (ports & adapters) and **Event
Sourcing** (frozen, append-only Pydantic events; state derived by fold).

## Layer diagram

```
┌────────────────────────────────────────────────────────────┐
│  PUBLIC API                                                │
│  Agent (facade) · AgentLoop · RunResult · EventBus         │
├────────────────────────────────────────────────────────────┤
│  ORCHESTRATION & HARNESS                                   │
│  AgentLoop (Intent → Route → Plan → Act → Verify → Memory) │
│  Workflow (subagent DSL) · SpecValidator · self-correction │
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

## The run pipeline

`AgentLoop.run_result` drives a single task through six phases:

1. **Intent** — `IntentParser` classifies the prompt into an intent type and
   complexity (LOW / MEDIUM / HIGH). Deterministic heuristics run first and are
   authoritative when they match; when none matches with confidence and a
   `classifier_model` is configured, the parser asks that model to classify the
   task, falling back to the heuristic default on any failure (M11.3).
2. **Route** — `ModelRouter` maps the intent to a tier and model, with a
   fallback chain and circuit breakers.
3. **Plan** — for MEDIUM/HIGH tasks, the model generates a `TaskSpec`
   (objective, steps, target files, acceptance criteria), validated by
   `SpecValidator`.
4. **Act** — a ReAct loop: the model emits tool calls, the registry executes
   them (enforcing permission policy, sandbox, and the subagent authority
   gate), and results feed back. Read-only calls may dispatch concurrently.
5. **Verify** — when the loop produces a final answer, acceptance criteria are
   checked; failures trigger self-correction (up to `self_correction_max`).
6. **Memory** — the trajectory is recorded to `EpisodicMemory` for future
   few-shot retrieval; durable facts go to `PersistentMemory`.

Every step publishes frozen events to the `EventBus` and appends them to the
`EventStore`. `Conversation.fold` derives the conversation state from the event
sequence, so history is replayable and auditable.

## The `Agent` facade

`Agent` (in `nullain/agent/facade.py`) is the primary entry point. It assembles
the collaborators — provider, tool registry, permission policy, router, sandbox,
memory — with safe defaults, then delegates to `AgentLoop`. It never
reimplements the loop. `run` returns a `RunResult`; `stream` yields events then
the final `RunResult`; `run_sync` is a thin synchronous facade.

## Ports & adapters

- **`LLMProvider`** (port) ← `OllamaCloudProvider` (adapter). Swap in a fake for
  tests or a different backend for production.
- **`Sandbox`** (port) ← platform adapters: `landlock` (Linux ≥5.13),
  `seatbelt` (macOS), `windows_job` (Windows). Fail-closed when required.
- **`Clock`** (port) ← `SystemClock` (adapter), injectable for deterministic
  tests.

## Security model

- Subprocesses run by explicit argv, never `shell=True`.
- Paths are resolved (`resolve()`) and checked with `is_relative_to` before
  authorization; symlinks are resolved first.
- A subagent's authority is the **meet** of parent ∧ delegation ∧ child ∧
  policy — a single denial removes a capability outright (no ASK escape).
- Plugins are signed, SBOM'd, capability-manifested bundles, fail-closed at
  every branch.

## Daemon (`nullain-agentd`)

`nullain-agentd` exposes the SDK to non-Python hosts over a stdio NDJSON
protocol (`nullain/protocol/`), rather than embedding this Python process
into a client written in another language. `run_agentd` owns one process's
worth of shared, expensive-to-create state — the `LLMProvider`, the MCP/LSP
server subprocesses, prepared plugins, the OS sandbox adapter — prepared
once at startup and reused across every session (P4.23/P4.25).

Per-session state (the tool registry, permission policy, workspace root,
and persistent memory) is kept in a `dict[session_id, _SessionState]`, never
as a single mutable variable — two concurrent `session_id`s cannot see or
clobber each other's workspace or tools (issue #43). A `user.message` for an
unrecognized `session_id` is rejected with a structured error rather than
silently falling back to whichever session most recently started.

Session history is durable, not process-local: every event lands in
`<workspace>/.nullain/sessions.db` (the same `EventStore` `Agent` uses — see
[`api-stability.md`](api-stability.md)'s M11 notes), so a `session.start` on
a `session_id` that already has events resumes that conversation — including
across a full daemon restart, and including the same orphaned-tool-result
repair pass `Agent._load_session_history` applies (issue #44) for a session
persisted before the #24 compaction fix.

`ASK`-level permission checks and `ask_user` tool calls have no TTY to read
from inside the daemon, so both are proxied over the same NDJSON channel as
`permission.request`/`permission.response` and
`ask_user.request`/`ask_user.response` pairs; the daemon blocks the affected
tool call until a matching response arrives, and treats a closed stream (EOF)
as a denial (fail-closed), never a hang.

## Subagents & worktree isolation

`AgentLoop.spawn` runs a sub-agent with fresh context and an isolated event
bus, returning only its final text. When `isolation="worktree"` (M11.4), the
child runs in a detached `git worktree` (`git worktree add --detach`) with a
tool registry re-rooted at the worktree via a `tool_factory` — tools bind to
their workspace root at creation time, so the child edits an isolated checkout
rather than the parent's. Changed files are integrated back into the parent
workspace afterwards, and the worktree is removed in `finally` (including on
failure/cancellation). The authority-intersection law is preserved: isolation
is additional to the capability gate, never a substitute.
