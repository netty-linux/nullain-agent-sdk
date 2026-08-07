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

## License

MIT — see [LICENSE](https://github.com/netty-linux/nullain-agent-sdk/blob/master/nullain-agentd/LICENSE).
