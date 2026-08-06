# Built-in Tools

The built-in tools are registered by `register_default_tools(registry,
workspace_root)`. Each tool declares the capabilities it exercises, enforced
by the registry's authority gate for subagents (P4.24).

## Filesystem

| Tool | Capability | Read-only | Description |
|------|-----------|-----------|-------------|
| `read_file` | READ | yes | Numbered `cat -n` lines; `offset`/`limit` paging; truncates absurdly long lines. |
| `write_file` | WRITE | no | Write text content to a file. |
| `edit_file` | WRITE | no | Exact replacement; rejects ambiguous matches and no-ops; requires a prior read. |
| `multi_edit` | WRITE | no | Atomic chained edits; writes only if every hunk succeeds. |
| `grep` | READ | yes | ripgrep-backed search with a pure-Python fallback; content/files/count modes. |
| `glob` | READ | yes | Find files matching a glob pattern. |
| `list_directory` | READ | yes | List directory entries (`name/` for dirs). |
| `todo_write` | — | no | Update the todo list; at most one `in_progress`; emits a `TodoEvent`. |
| `undo` | WRITE | no | Restore the most recent pre-write checkpoint of a file (M11.1). |

### `read_file`

```text
read_file(path, offset=0, limit=2000) -> str
```

Returns numbered lines (`1\t...`). When the file exceeds the page, the footer
announces how many lines remain and the `offset` to continue from.

### `edit_file`

```text
edit_file(path, old_str, new_str, replace_all=False) -> str
```

Fails if `old_str` appears more than once (unless `replace_all`), if it is a
no-op (`old_str == new_str`), or if the file was not read first in the session.

### `multi_edit`

```text
multi_edit(path, edits: [{old_str, new_str, replace_all?}]) -> str
```

Applies all edits in sequence over in-memory content; writes to disk only if
every edit succeeds. Each edit operates on the result of the previous one.

### `grep`

```text
grep(pattern, relative_dir=".", output_mode="content", context_lines=0,
     glob_filter=None, case_insensitive=False, head_limit=200) -> str
```

Uses ripgrep when available (explicit argv, never shell); otherwise a pure-Python
fallback with the same output format. Truncation is always announced.

## Execution

| Tool | Capability | Read-only | Description |
|------|-----------|-----------|-------------|
| `bash` | EXEC | no | Run a command by explicit argv list (never `shell=True`), sandboxed. |
| `git` | EXEC | no | Git operations, sandboxed. |

## Network

| Tool | Capability | Read-only | Description |
|------|-----------|-----------|-------------|
| `web_fetch` | NETWORK | yes | Fetch a URL and convert HTML to text. |

## Interaction

| Tool | Capability | Read-only | Description |
|------|-----------|-----------|-------------|
| `ask_user` | — | no | Ask the user a question; returns an error when no callback is wired. |

## Discovery

| Tool | Capability | Read-only | Description |
|------|-----------|-----------|-------------|
| `search_tools` | READ | yes | Search for and hydrate deferred-schema MCP tools (P4.26). |

## Memory

| Tool | Capability | Read-only | Description |
|------|-----------|-----------|-------------|
| `save_memory` | WRITE | no | Persist a durable fact (registered when a `PersistentMemory` is provided). |
| `read_memory` | READ | yes | Read persisted facts (registered when a `PersistentMemory` is provided). |

## Language Server (LSP) — read-only code intelligence

Registered by `register_lsp_tools(registry, clients)` when LSP servers are
configured (M11.2). Each tool routes a file to its server by extension →
language → server map. A server that fails to initialize is logged and skipped
(fail-soft); an unsupported file type or unavailable server surfaces as a
`ToolResult` error, never a session crash.

| Tool | Capability | Read-only | Description |
|------|-----------|-----------|-------------|
| `lsp_diagnostics` | READ | yes | Pull diagnostics for a file. |
| `lsp_goto_definition` | READ | yes | Jump to a symbol's definition. |
| `lsp_find_references` | READ | yes | Find all references to a symbol. |
| `lsp_hover` | READ | yes | Hover documentation for a symbol. |

## Capabilities

The `Capability` enum values are `READ`, `WRITE`, `EXEC`, `NETWORK`, `SPAWN`.
A subagent's effective authority is the intersection of parent authority,
delegation, child definition, and policy — a tool call is refused if it
requires a capability the subagent lacks or a tool outside its allowed set.
