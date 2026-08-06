"""Nullain Agent SDK — Exceptions Hierarchy.

All domain exceptions inherit from NullainError.
"""

from typing import Any


class NullainError(Exception):
    """Base exception for all Nullain SDK errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ProviderError(NullainError):
    """Base exception for LLM provider errors."""

    pass


class ProviderTimeoutError(ProviderError):
    """Raised when an LLM provider call times out."""

    pass


class ProviderRateLimitError(ProviderError):
    """Raised when hitting LLM provider rate limits (e.g. HTTP 429)."""

    pass


class ProviderAuthenticationError(ProviderError):
    """Raised when provider authentication fails (e.g. invalid API key)."""

    pass


class ToolError(NullainError):
    """Base exception for tool execution or validation errors."""

    pass


class ToolPermissionError(ToolError):
    """Raised when tool execution is denied by PermissionPolicy."""

    pass


class ToolExecutionError(ToolError):
    """Raised when a tool execution fails inside sandbox."""

    pass


class ToolNotFoundError(ToolError):
    """Raised when attempting to execute an unregistered tool."""

    pass


class RouterError(NullainError):
    """Base exception for model routing errors."""

    pass


class NoModelAvailableError(RouterError):
    """Raised when no suitable model is available in the requested tier."""

    pass


class BudgetExceededError(NullainError):
    """Raised when task token or cost budget is exceeded."""

    pass


class ContextError(NullainError):
    """Raised when context assembly or compaction fails."""

    pass


class ContextWindowExhaustedError(ContextError):
    """Raised when compaction cannot bring the context window under threshold.

    Indicates thrashing: repeated compaction does not free enough tokens, so
    continuing would loop indefinitely. Fail loudly instead of degrading.
    """

    pass


class SpecValidationError(NullainError):
    """Raised when Plan/Act task specification fails validation."""

    pass


class MCPError(NullainError):
    """Base exception for Model Context Protocol (MCP) client errors."""

    pass


class MCPProtocolError(MCPError):
    """Raised when an MCP server returns a malformed or error JSON-RPC response."""

    pass


class MCPTransportError(MCPError):
    """Raised when an MCP transport fails (spawn, EOF, timeout, I/O)."""


class LSPError(NullainError):
    """Base class for LSP client errors (M11.2)."""


class LSPProtocolError(LSPError):
    """Raised when an LSP server returns an error or malformed response."""


class LSPTransportError(LSPError):
    """Raised when an LSP transport fails (spawn, EOF, timeout, I/O)."""

    """Raised when the MCP transport layer fails (spawn, I/O, EOF)."""

    pass


class SandboxError(NullainError):
    """Base exception for OS-level sandbox failures."""

    pass


class SandboxUnavailableError(SandboxError):
    """Raised when a sandbox is required but no usable adapter is available.

    Fail-closed: rather than executing a subprocess without isolation, the
    runner refuses and raises. This is the security differentiator — the agent
    is the sandboxed-by-default harness, not the fail-open one.
    """

    pass


class PluginError(NullainError):
    """Base exception for plugin loading/verification failures.

    Plugins are external, signed, capability-manifested tool bundles (P4.25).
    Every load failure is fail-closed: an unverified or over-capable plugin is
    refused rather than loaded with reduced guarantees.
    """

    pass


class PluginSignatureError(PluginError):
    """Raised when a plugin's signature is missing, unverifiable, or invalid.

    Covers: unsigned plugin when signatures are required; signed plugin with no
    verifier backend available (fail-closed); signature that does not verify
    against the trusted key set; unknown key id.
    """

    pass


class PluginManifestError(PluginError):
    """Raised when a plugin manifest is structurally invalid or inconsistent.

    Covers: schema mismatch, a tool whose required capabilities exceed the
    plugin's declared capabilities, an unsupported signature algorithm, or a
    runtime-version precondition not met.
    """

    pass


class PluginSbomError(PluginError):
    """Raised when a plugin's SBOM digest does not match the computed digest.

    The signature covers the canonicalized manifest + SBOM digest, so any
    dependency drift that escapes the digest invalidates the signature too;
    this error is the explicit SBOM-integrity check.
    """

    pass


class PluginCapabilityError(PluginError):
    """Raised when a plugin declares no tool usable under the granted capabilities.

    Tools whose required capabilities exceed the operator-granted set are
    dropped (with a structured log); if every tool is dropped the plugin has
    nothing usable to register and the load is refused rather than registering
    an empty shell.
    """

    pass


__all__ = [
    "BudgetExceededError",
    "ContextError",
    "ContextWindowExhaustedError",
    "LSPError",
    "LSPProtocolError",
    "LSPTransportError",
    "MCPError",
    "MCPProtocolError",
    "MCPTransportError",
    "NoModelAvailableError",
    "NullainError",
    "PluginCapabilityError",
    "PluginError",
    "PluginManifestError",
    "PluginSbomError",
    "PluginSignatureError",
    "ProviderAuthenticationError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "RouterError",
    "SandboxError",
    "SandboxUnavailableError",
    "SpecValidationError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolPermissionError",
]
