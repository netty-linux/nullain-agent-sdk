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
  instead, so it never lands in a committed file).

## `[router]` — model routing

```toml
[router]
fallback_chain = ["deep", "balanced", "fast"]

[router.tiers.fast]
models = ["gpt-oss:20b"]
max_context = 32000

[router.tiers.balanced]
models = ["qwen3-coder:480b-cloud", "gpt-oss:120b"]
max_context = 128000

[router.tiers.deep]
models = ["deepseek-v4-pro"]
max_context = 128000
```

The `ModelRouter` classifies each task into an intent and complexity, picks a
tier, and falls back along `fallback_chain` when a model is unavailable or the
circuit breaker trips.

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
