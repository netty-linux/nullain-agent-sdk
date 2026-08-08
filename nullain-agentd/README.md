# nullain-agentd

NDJSON-over-stdio daemon exposing the
[Nullain Agent SDK](https://github.com/netty-linux/nullain-agent-sdk) to
non-Python clients — editors, CLI wrappers, or a host process written in
Go, Rust, or any other language that can speak newline-delimited JSON over
a pipe.

This package requires [`nullain-sdk`](https://pypi.org/project/nullain-sdk/)
and [`nullain-tools`](https://pypi.org/project/nullain-tools/) (installed
automatically as dependencies).

## Install

```bash
pip install nullain-agentd
```

## Running

```bash
nullain-agentd
```

The daemon reads one JSON object per line on stdin and writes one JSON
object per line on stdout — no framing beyond newlines, no partial-message
buffering required on the client side.

## Protocol

Every message is a versioned envelope:

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

The full JSON Schema contract is published at
[`schema/protocol_v1.json`](https://github.com/netty-linux/nullain-agent-sdk/blob/master/schema/protocol_v1.json)
in the main repository — generate it locally from the source of truth with
`make schema`. Golden fixture files for every message type live under
[`schema/golden/`](https://github.com/netty-linux/nullain-agent-sdk/tree/master/schema/golden).

This is intended as an integration point for building a native client
(a Go or Rust CLI, an editor extension) on top of the SDK without embedding
a Python runtime in that client — the daemon is the only process that needs
Python.

## Sessions

Each session is identified by the `session_id` in its `session.start` and
`user.message` payloads:

```json
{"v": 1, "type": "session.start", "id": "1", "payload": {"session_id": "sess_100", "workspace_root": "/path/to/project"}}
{"v": 1, "type": "user.message", "id": "2", "payload": {"session_id": "sess_100", "prompt": "list the python files"}}
```

- **Concurrent sessions are isolated.** Each `session_id` gets its own tool
  registry, permission policy, and workspace root — a `user.message` for one
  session never touches another session's files, even if both were started
  on the same daemon connection. Shared, expensive-to-create collaborators
  (the LLM provider, MCP/LSP server subprocesses, prepared plugins, the OS
  sandbox) are prepared once and reused across every session.
- **Sessions resume after a restart.** Every event is persisted to
  `<workspace>/.nullain/sessions.db` as it happens. A `session.start` for a
  `session_id` with prior history in that store picks the conversation back
  up — send the same `session_id` again after the daemon process restarts
  and the agent sees the full prior trajectory, not a blank slate. A session
  that predates a Landlock/compaction fix gets its history transparently
  repaired on load (never a silent mutation — a `SessionRepairedEvent`
  records exactly what changed) rather than failing the request.
- **Permission prompts are proxied to the client.** The daemon has no TTY of
  its own, so an `ASK`-level tool call emits a `permission.request` over the
  same channel and blocks until a matching `permission.response` arrives (or
  the client closes the stream, which is treated as a denial — fail-closed).
  `ask_user` tool calls round-trip the same way via `ask_user.request` /
  `ask_user.response`.
- A `user.message` for a `session_id` that never had a `session.start`
  returns a `session.end` with `status: "error"` rather than guessing which
  workspace to use.

## License

MIT — see [LICENSE](https://github.com/netty-linux/nullain-agent-sdk/blob/master/nullain-agentd/LICENSE).
