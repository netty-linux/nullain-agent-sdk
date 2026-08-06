"""Nullain Agent SDK — Plugins: signed, SBOM'd, capability-manifested tool bundles (P4.25).

A plugin is an MCP server whose identity, launch command, capabilities, tool
declarations, and SBOM are pinned in a signed manifest. The loader verifies the
signature, intersects the plugin's declared capabilities with the operator's
grant, and registers the tools into a :class:`~nullain.tools.registry.ToolRegistry`
with per-tool ``requires``/``permission_level`` taken from the manifest. The
P4.23 sandbox confines the plugin subprocess; the P4.24 authority gate confines
what a delegating subagent may reach. This module composes the two.

Fail-closed by default: an unverified or over-capable plugin is refused rather
than loaded with reduced guarantees. Unsigned plugins load only under an
explicit ``require_signature=false`` opt-in (trusted-local).
"""

from nullain.plugins.loader import PluginLoader, PreparedPlugin
from nullain.plugins.manifest import (
    MANIFEST_SCHEMA_VERSION,
    SBOM,
    SBOM_SCHEMA_VERSION,
    PluginManifest,
    PluginSignature,
    PluginToolDecl,
    PluginTransportDecl,
    SBOMDependency,
)
from nullain.plugins.signing import (
    Ed25519Verifier,
    NoVerifier,
    SignatureVerifier,
    select_verifier,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "SBOM",
    "SBOM_SCHEMA_VERSION",
    "Ed25519Verifier",
    "NoVerifier",
    "PluginLoader",
    "PluginManifest",
    "PluginSignature",
    "PluginToolDecl",
    "PluginTransportDecl",
    "PreparedPlugin",
    "SBOMDependency",
    "SignatureVerifier",
    "select_verifier",
]
