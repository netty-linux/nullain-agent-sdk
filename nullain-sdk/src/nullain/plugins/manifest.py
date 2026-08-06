"""Nullain Agent SDK — Plugin Manifest, Tool Declarations, and SBOM (P4.25).

A plugin is a signed, capability-manifested bundle of tools (v1: an MCP server
whose command, capabilities, tool declarations, and SBOM are pinned in a signed
manifest). This module defines the data model only — verification, capability
enforcement, and registration live in :mod:`nullain.plugins.loader`.

The manifest is the trust anchor: the signature covers a canonical
serialization of the manifest (with the signature field excluded and the SBOM
collapsed to its pinned digest), so any drift in identity, transport command,
capabilities, tool declarations, or dependencies invalidates the signature.
"""

import hashlib
import json

from pydantic import BaseModel, Field

from nullain.authority import Capability
from nullain.tools.permissions import PermissionLevel

MANIFEST_SCHEMA_VERSION = "nullain-plugin/0.1"
SBOM_SCHEMA_VERSION = "nullain-sbom/0.1"


class SBOMDependency(BaseModel):
    """One entry in a plugin's software bill of materials."""

    name: str
    version: str
    # Content hash of the package artifact, e.g. "sha256:...". Empty when the
    # publisher did not pin an artifact hash (recorded honestly, not silently).
    digest: str = ""


class SBOM(BaseModel):
    """Minimal software bill of materials: a hashed dependency manifest.

    v1 is a structured, content-hashed dependency list — not a full CycloneDX/
    SPDX document. The digest is canonical (dependencies sorted by name+version,
    JSON with sorted keys) so signature coverage is deterministic. Emitting
    CycloneDX/SPDX via the standard toolchain is a documented follow-up.
    """

    schema_version: str = SBOM_SCHEMA_VERSION
    dependencies: list[SBOMDependency] = Field(default_factory=list[SBOMDependency])

    def digest(self) -> str:
        """Deterministic SHA-256 over the canonical (sorted) dependency list."""
        payload = json.dumps(
            [
                d.model_dump(mode="json")
                for d in sorted(self.dependencies, key=lambda d: (d.name, d.version))
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PluginToolDecl(BaseModel):
    """A tool a plugin declares it exposes.

    ``name`` is the tool name as the plugin's MCP server reports it (un-
    namespaced); the loader matches it to the registered ``mcp__<plugin>__<tool>``
    name. ``requires`` MUST be a subset of the plugin's declared capabilities —
    a tool claiming a capability the plugin did not declare is a manifest error
    (fail-closed at load). ``permission_level`` optionally overrides the registry
    heuristic (used for MCP tools whose side-effects are not introspectable).
    """

    name: str
    requires: frozenset[Capability] = frozenset()
    permission_level: PermissionLevel | None = None
    description: str = ""


class PluginTransportDecl(BaseModel):
    """The MCP server command, signed inside the manifest.

    Putting the command in the signed manifest (not in the operator config)
    prevents command substitution: an attacker who tampers with the launch
    command breaks the signature. The operator config only references the
    manifest path and optionally narrows granted capabilities.
    """

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class PluginSignature(BaseModel):
    """A detached signature over the manifest's canonical signed message.

    ``key_id`` identifies which trusted key signed (matched against the
    operator's ``trusted_keys`` map). ``value`` is the base64 signature.
    """

    algorithm: str = "ed25519"
    key_id: str
    value: str


class PluginManifest(BaseModel):
    """A signed plugin manifest — the trust anchor for a plugin bundle."""

    schema_version: str = MANIFEST_SCHEMA_VERSION
    name: str
    version: str
    publisher: str
    description: str = ""
    min_runtime_version: str | None = None
    transport: PluginTransportDecl
    # Capabilities the plugin declares it may exercise. Every tool's `requires`
    # must be a subset of this set; the operator's granted capabilities are
    # intersected with it at load (the P4.24 meet, applied to plugins).
    capabilities: frozenset[Capability]
    tools: list[PluginToolDecl] = Field(default_factory=list[PluginToolDecl])
    sbom: SBOM
    # Pinned digest of `sbom`, carried explicitly so signature coverage of the
    # dependency list is visible without re-serializing the SBOM.
    sbom_digest: str
    signature: PluginSignature | None = None

    def signed_message(self) -> bytes:
        """Canonical bytes covered by the signature.

        The manifest is serialized with the signature field excluded and the
        SBOM collapsed to its pinned digest, using sorted-key compact JSON for
        determinism. The verifier recomputes exactly this message.
        """
        data = self.model_dump(mode="json", exclude={"signature"})
        data["sbom"] = self.sbom_digest
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "SBOM",
    "SBOM_SCHEMA_VERSION",
    "PluginManifest",
    "PluginSignature",
    "PluginToolDecl",
    "PluginTransportDecl",
    "SBOMDependency",
]
