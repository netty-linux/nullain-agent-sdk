# nullain-tools

Built-in coding tools for the
[Nullain Agent SDK](https://github.com/netty-linux/nullain-agent-sdk) —
paged file reads, safe edits, ripgrep-backed search, checkpoints/undo, git
operations, and sandboxed shell execution.

This package requires [`nullain-sdk`](https://pypi.org/project/nullain-sdk/)
(installed automatically as a dependency) and is registered into an `Agent`
automatically — most users never import it directly.

## Install

```bash
pip install nullain-tools
```

`nullain-sdk` pulls this in as a dependency, so a plain
`pip install nullain-sdk` already gets you the full built-in toolset.

## What's inside

| Tool | Description |
|------|-------------|
| `read_file` | Numbered, paged reads (`cat -n` style); truncates absurdly long lines. |
| `write_file` / `edit_file` / `multi_edit` | Safe writes — exact-match edits reject ambiguous matches and no-ops; `multi_edit` applies a batch of hunks atomically. |
| `grep` / `glob` / `list_directory` | ripgrep-backed search with a pure-Python fallback; glob matching; directory listing. |
| `bash` | Shell execution via explicit argv (never `shell=True`), inside the SDK's OS-level sandbox when available. |
| `git_status` / `git_diff` / `git_commit` | Controlled git operations scoped to the workspace. |
| `web_fetch` | Fetch a URL and return HTML converted to plain text. |
| `todo_write` / `undo` | Session todo tracking; restore the most recent pre-write checkpoint of a file. |

Every tool that reads file content (`read_file`, `grep`) or reports what
changed (`edit_file`, `multi_edit`) redacts common secret patterns (API
keys, tokens) before the output reaches the model or a log.

See [`docs/tools.md`](https://github.com/netty-linux/nullain-agent-sdk/blob/master/docs/tools.md)
in the main repository for the full reference, including each tool's
capability requirements and permission level.

## License

MIT — see [LICENSE](https://github.com/netty-linux/nullain-agent-sdk/blob/master/nullain-tools/LICENSE).
