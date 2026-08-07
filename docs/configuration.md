# Configuration

The SDK reads a `nullain.toml` file. It is located at the path in the
`NULLAIN_CONFIG` environment variable, or `./nullain.toml` in the current
directory. A complete example lives at `nullain.toml.example`.

## Top-level keys

```toml
ollama_base_url = "https://ollama.com"
```

- `ollama_base_url` — base URL of the Ollama Cloud endpoint.
- `ollama_api_key` — API key (usually set via the `OLLAMA_API_KEY` env var
  instead, so it never lands in a committed file — `NULLAIN_OLLAMA_API_KEY`
  also works and takes precedence if both are set).

## `[router]` — model routing

```toml
[router]
fallback_chain = ["deep", "balanced", "fast"]
# Optional model used to classify a task's intent/complexity when the
# deterministic heuristics are not confident (M11.3). When unset, the parser
# stays heuristic-only.
# classifier_model = "deepseek-v4-flash:0731-cloud"

[router.tiers.fast]
models = ["deepseek-v4-flash:0731-cloud"]
max_context = 64000

[router.tiers.balanced]
models = ["glm-5.2:cloud"]
max_context = 128000

[router.tiers.deep]
models = ["gemma4:31b-cloud"]
max_context = 128000
```

The `ModelRouter` classifies each task into an intent and complexity, picks a
tier, and falls back along `fallback_chain` when a model is unavailable or the
circuit breaker trips.

`IntentParser` runs deterministic heuristics first and is authoritative when
they match a known keyword. When none matches with confidence and
`classifier_model` is set, the parser asks that model (tier `fast`) to classify
the task, falling back to the heuristic default on any failure (M11.3). Results
are cached by prompt hash within the session.

## `[sandbox]` — OS-level subprocess isolation

```toml
[sandbox]
enabled = true
required = true
allow_paths = []
deny_network = true
```

- `enabled` — turn isolation on/off.
- `required` — when `true` and the platform adapter is unavailable, `bash`/`git`
  **refuse** to run (fail-closed) rather than executing unconfined.
- `allow_paths` — extra writable paths granted to the sandboxed child.
- `deny_network` — block network access where the adapter supports it.

## `[mcp.servers.<name>]` — MCP servers

```toml
[mcp.servers.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
auto_approve = false
enabled = true
```

Each server is spawned over stdio with an explicit argv list (never a shell).
Tools are registered as `mcp__<server>__<tool>`. `auto_approve = false` (the
default) gates every MCP tool call through the permission loop.

Manage these from the CLI:

```bash
nullain mcp list
nullain mcp add filesystem --command npx --args -y @modelcontextprotocol/server-filesystem /workspace
nullain mcp remove filesystem
```

## `[lsp.servers.<language>]` — LSP servers

```toml
[lsp.servers.python]
command = "pyright-langserver"
args = ["--stdio"]
enabled = true

[lsp.servers.typescript]
command = "typescript-language-server"
args = ["--stdio"]
enabled = true
```

Each server is spawned over stdio with an explicit argv list (never a shell).
The four read-only tools (`lsp_diagnostics`, `lsp_goto_definition`,
`lsp_find_references`, `lsp_hover`) route a file to its server by extension →
language → this map. A server that fails to initialize is logged and skipped
(fail-soft); an unsupported file type or unavailable server surfaces as a
`ToolResult` error (M11.2).

## `[plugins]` — signed plugin bundles

```toml
[plugins]
enabled = true
require_signature = true
trusted_keys = { alpha = "base64-ed25519-pubkey..." }
allowed_capabilities = ["read", "write", "network"]

[plugins.entries.search]
manifest = "./plugins/search/plugin.json"
auto_approve = false
allowed_capabilities = ["read"]
enabled = true
```

Plugins are signed, SBOM'd, capability-manifested MCP bundles. The loader is
fail-closed at every branch: an unsigned or unverified plugin is refused when
`require_signature = true`.

## `[[hooks.*]]` — lifecycle hooks

```toml
[[hooks.pre_tool]]
command = ["./hooks/pre_tool.sh"]
timeout = 30.0
```

Each hook runs a command (argv list, no shell) and pipes the event payload as
JSON on stdin. Exit codes: `0` success, `2` block, other = non-blocking
failure.
