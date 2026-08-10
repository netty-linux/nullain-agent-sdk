# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/) —
see [`docs/api-stability.md`](docs/api-stability.md) for exactly what's
covered by that guarantee at this pre-1.0 stage.

## [Unreleased]

## [0.6.0] (nullain-sdk)

### Fixed
- The Plan phase no longer demands `target_files` from tasks that
  produce no files. `_generate_spec` asked every MEDIUM/HIGH task for
  "any files you expect to create or modify" unconditionally — but
  `target_files` is load-bearing past the plan itself: `SpecValidator.
  verify` fails the run when a listed file is absent from disk, and only
  writes to a listed file count as progress for the adaptive step
  budget. A file-less task (a payment charge, a web search, a plain
  answer) was therefore pushed to invent filenames, failed verification
  for not creating them, and entered a `[VERIFY-CORRECTION]` round
  chasing a file it had made up. Both the JSON schema and the
  instruction now state the field is optional, belongs empty in that
  case, and explain why. Observed downstream as an agent answering
  "generate a PIX charge" with `write_file` + `python_sandbox` +
  `read_file` + `list_directory` and an error reading a nonexistent
  `output/test_results.txt`, never calling the dedicated payment tool.

### Added
- `AgentConfig.plan_complexity_threshold` (`"medium"` | `"high"` |
  `"never"`, default `"medium"` — unchanged behavior) sets the lowest
  complexity that still runs the Plan phase. `IntentParser` falls back
  to MEDIUM whenever no keyword heuristic matches and no
  `router.classifier_model` is configured, so a general-purpose
  deployment plans on effectively every turn; `"high"` restricts
  planning to explicitly complex work. An unrecognized value falls back
  to the default rather than silently disabling the phase.

## [0.5.0] (nullain-sdk) / [0.4.0] (nullain-tools)

### Added
- `web_search` now tries a self-hosted SearXNG instance first (via
  `WebSearchConfig`-equivalent `searxng_base_url` / `register_default_tools(searxng_base_url=...)`),
  falling back automatically to the existing DuckDuckGo scrape on any
  failure (timeout, non-2xx, malformed JSON, empty results). Unconfigured,
  behavior is unchanged — DuckDuckGo only.
- `web_fetch` can now render pages with Crawl4AI (a real headless browser)
  before falling back, via `WebFetchConfig.use_crawl4ai` /
  `register_default_tools(web_fetch_use_crawl4ai=True)`. Solves JS-heavy
  pages plain `httpx` can't render. The existing Wayback Machine fallback
  is unchanged and still triggers on a bot-block response from either
  path. Requires the `nullain-tools[crawl]` extra; degrades to a clean
  failure (never raises) if it's missing.
- `PostgresEventStore.append()` now enqueues onto a bounded in-process
  queue and returns immediately instead of awaiting the INSERT inline,
  removing per-event Postgres round-trip latency from the agent loop's
  critical path. A single background task drains the queue in batches via
  `executemany`. New `flush()` method waits for the queue to fully drain;
  `get_session_events`/`list_session_ids`/`get_latest_session_id` all call
  it automatically before reading, so resume/replay never observes a
  truncated trajectory. `close()` flushes before closing the pool. See the
  module docstring for the full design and the "accepted for write" vs
  "durably written" semantics this introduces (`SQLiteEventStore` is
  unaffected — it remains synchronous).
- `configure_tracing(exporter="otlp")`: ships spans to a real OTLP/HTTP
  backend (Jaeger, Tempo, Honeycomb, ...) via the new `otlp` extra,
  honoring the standard `OTEL_EXPORTER_OTLP_*` env vars. Previously, any
  `exporter` value other than `"console"` silently installed no span
  processor at all — that's now a `ValueError` instead of a silent no-op.
- `ContextManager`'s LLM-based compaction now asks the model for a
  structured `StateSummary` (key decisions, files changed, errors
  encountered, outstanding work) via a forced tool call, rendered to the
  same prose shape `CompactionEvent.summary` always had. Falls back to
  free-text (then to the structural summary) if the model ignores the
  tool or returns malformed arguments.

## [0.4.0] (nullain-sdk) / [0.3.0] (nullain-tools)

### Added
- `web_search` tool: scrapes DuckDuckGo's public HTML endpoint (no API
  key, no JavaScript) and returns real result URLs with titles and
  snippets. Fills a real gap — `web_fetch` requires already knowing a
  URL, so without a search tool a model has no way to find one besides
  guessing from (often stale or wrong) training knowledge, which was
  producing repeated 404s in production.

### Changed
- `web_fetch` now automatically falls back to the Wayback Machine on a
  bot-block response (401/403/429) instead of just reporting the error.
  A snapshot older than 7 days triggers a fresh capture first
  (best-effort — a failed capture falls back to whatever snapshot
  already exists). The result always states the snapshot's date, so
  stale content is never presented as if it were live. 404 and other
  non-bot-block statuses are unaffected.

### Fixed
- `nullain.__version__`/`nullain_tools.__version__` were hardcoded and
  two releases stale (`"0.1.0"` while `0.3.2`/`0.2.0` were live on
  PyPI). Both (and `nullain_agentd.__version__`) now derive from
  `importlib.metadata` at import time instead of a hand-copied string.

## [0.3.2]

### Fixed
- `ToolRegistry.execute` now aliases the `bash` tool's `command: str`
  argument to its real `command_args: list[str]` parameter (via
  `shlex.split`, still a plain argv — no shell interpretation). Models
  across providers reliably guess `command: str` (the shape most other
  agent frameworks use) instead of reading the actual schema, previously
  crashing with a raw `TypeError`. The normalization happens before the
  permission policy's deny-pattern check runs, so a `command` string
  can't bypass `evaluate_command`'s destructive-command denial the way
  an unrecognized non-list value would.

## [0.3.1]

### Fixed
- `nullain-sdk`'s dependency on `nullain-tools` tightened from `>=0.1.0`
  to `>=0.2.0`. `nullain.agent.facade.Agent.__init__` always passes
  `web_fetch_headers` to `register_default_tools()` (added in
  `nullain-tools` 0.2.0, alongside 0.3.0's web_fetch changes) — a plain
  `>=0.1.0` let pip resolve the older `nullain-tools` release (no
  matching keyword argument) next to the newer `nullain-sdk`, which broke
  every `Agent()` construction with `TypeError: register_default_tools()
  got an unexpected keyword argument 'web_fetch_headers'`. Confirmed in
  production before this fix; PyPI installs of `nullain-sdk` before
  0.3.1 need `pip install --upgrade nullain-tools` alongside upgrading
  the SDK.

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
