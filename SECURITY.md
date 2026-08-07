# Security Policy

## Reporting a vulnerability

If you find a security vulnerability in the Nullain Agent SDK, please
report it privately — do not open a public GitHub issue.

Use GitHub's private vulnerability reporting for this repository:
[https://github.com/netty-linux/nullain-agent-sdk/security/advisories/new](https://github.com/netty-linux/nullain-agent-sdk/security/advisories/new)

Include, as far as you're able:

- A description of the vulnerability and its impact.
- Steps to reproduce (a minimal repro is enormously helpful).
- The affected version(s) — `nullain-sdk`, `nullain-tools`, and
  `nullain-agentd` are versioned together.
- Any suggested fix or mitigation, if you have one.

We'll acknowledge your report as soon as we can and keep you updated as we
work on a fix. We ask that you give us a reasonable window to address the
issue before any public disclosure.

## Supported versions

This project is pre-1.0 (`0.x`). Only the latest published release is
supported with security fixes; there is no backport policy across minor
versions yet. See [`docs/api-stability.md`](docs/api-stability.md) for
what's covered by compatibility guarantees at this stage.

## Scope

In scope:

- The `nullain-sdk`, `nullain-tools`, and `nullain-agentd` packages as
  published to PyPI, and this repository's source.
- The CLI (`nullain`) and daemon (`nullain-agentd`) entry points.
- The permission/sandbox security model (`PermissionPolicy`, the OS-level
  sandbox adapters, the subagent authority-intersection gate) and the
  signed-plugin verification path.

Out of scope:

- Vulnerabilities in third-party dependencies — please report those
  upstream (though we're glad to hear about them too, so we can update).
- The Ollama Cloud service itself, or any other LLM provider.
- Issues that require an attacker to already have arbitrary code execution
  on the host running the agent (the sandbox's threat model is about
  containing what the *agent itself* does, not about defending an already
  fully-compromised machine).

## What we consider a security issue here

Given this SDK executes shell commands, reads/writes files, and calls out
to an LLM that can itself be adversarially prompted, examples of real
security issues include:

- A way to escape the workspace root via path resolution (symlinks,
  traversal, etc.) that the permission policy or sandbox should have
  caught.
- A way to bypass the OS-level sandbox's fail-closed behavior (i.e.
  unsandboxed execution proceeding silently when the sandbox was required
  and unavailable).
- A prompt-injection path that lets untrusted tool output (a fetched web
  page, a file the agent reads) escalate to executing a command or writing
  a file the permission policy should have denied or asked about.
- A flaw in the signed-plugin verification (`Ed25519Verifier`) that lets an
  unsigned or tampered plugin manifest be accepted as valid.
- Secrets (API keys, credentials) leaking into logs, telemetry, or the LLM
  context despite the redaction layer.
