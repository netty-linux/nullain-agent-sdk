# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/) —
see [`docs/api-stability.md`](docs/api-stability.md) for exactly what's
covered by that guarantee at this pre-1.0 stage.

## [Unreleased]

## [0.3.0]

### Added
- `nullain.config.WebFetchConfig`: new `[web_fetch]` `nullain.toml` section
  (`user_agent`, `accept`, `accept_language`) letting an operator override
  the HTTP headers `web_fetch` sends on every request. Default stays an
  honest bot-identifying `User-Agent` (`Nullain-Agent-SDK/0.1
  (+web_fetch)`) — sites enforcing anti-scraping policy against it is
  expected behavior, not something to route around with a spoofed
  browser User-Agent by default.

### Changed
- `web_fetch` now appends a clear hint to HTTP 401/403/429 responses
  ("this site is likely blocking automated requests ... use a different
  source instead") so the agent stops retrying a blocked URL and looks
  elsewhere. Left unchanged for other statuses (404, 400, ...), which are
  usually a genuine broken-URL/bad-request error rather than a block.

## [0.2.0]

### Added
- `nullain.rag`: Cuckoo Filter + RAG Tree + `EmbeddingProvider`
  (`FastEmbedProvider`, in-process ONNX, no PyTorch/Ollama) +
  `VectorStore` (`QdrantStore`), all tenant-scoped by construction —
  `VectorStore.scoped_search` is the only search entry point, always
  filtered by `tenant_id`. Optional `[rag]` extra
  (`fastembed`, `qdrant-client`).
- `nullain.events.PostgresEventStore`: production `EventStorePort`
  adapter backed by Postgres/Supabase, for deployments running more than
  one agent-process replica against shared session history. Same
  append-only schema and `seq`-ordered resume semantics as
  `SQLiteEventStore` (renamed from `EventStore`, kept as a backward
  -compatible alias). `EventStorePort` extracted as the shared Protocol
  both adapters implement. Optional `[postgres]` extra (`asyncpg`).
- `nullain.quota.QuotaChecker`: an in-process Protocol consulted by
  `AgentLoop` before each step, alongside the existing `max_tokens`
  budget check — for tenant/billing-level quota enforcement, deliberately
  separate from `HookManager` (subprocess-based, no pre-LLM-call
  lifecycle point). A denial raises `QuotaExceededError` and surfaces as
  `RunResult.status == "quota_exceeded"`, distinct from `"budget"` (the
  per-run token ceiling).
- Arrow-key Yes/No/Always permission menu for `ASK`-level tool calls in
  the interactive chat, replacing the typed `y/N` prompt.
- Up/Down command history in the interactive chat, scoped to the current
  session (never persisted to disk).
- First-run setup wizard: running `nullain` with no prior configuration
  walks through provider setup before opening chat.
- Session persistence (SQLite-backed) with `--session`/`--continue`.
- Checkpoints and `undo` for file-write tools.
- Real-time CLI status lines for tool calls (Claude-Code-style colored
  status dots), collapsing repeated calls to the same tool into one line.
- `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue/PR
  templates.
- `docs/tui.md` documenting the terminal UI's visual design.

### Changed
- Default agent budget raised from 25 steps / 100k tokens to 100 steps /
  2M tokens, with both configurable via `[agent]` in `nullain.toml`.
- `read_file` cache hits now return a short pointer instead of re-sending
  the full file content, since the model already has it in context —
  saves real tokens on repeat reads within a session.
- Secret redaction extended from `bash` output to `read_file`, `grep`,
  `edit_file`, and `multi_edit` output.
- Permission deny-list expanded beyond the original 6 patterns (destructive
  disk commands, `curl | sh`-style remote code execution, common
  credential file locations); fixed a false positive where `.env.example`
  was denied along with real `.env` files.

### Fixed
- `ChatMessage.to_api_dict()` now sends an explicit `content: null` for
  tool-call-only assistant turns instead of omitting the key — fixes an
  intermittent `400 invalid message content type: <nil>` from Ollama
  Cloud's compat shim.
- Context compaction no longer splits a tool-call turn from its results —
  a prior bug could compact away the assistant message that issued a tool
  call while keeping its result, producing an invalid message history.
- `nullain-sdk` now declares `nullain-tools` and (on Windows) `colorama`
  as real dependencies, instead of only working by accident inside the
  monorepo workspace.
- Configurable `bash_timeout` (default 300s, up from a hardcoded 120s)
  so long-running commands aren't killed mid-execution.

## [0.1.0]

Initial pre-release version.
