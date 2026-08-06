"""Nullain Agent SDK — Plugin Loader (P4.25).

Loads a signed, capability-manifested plugin into a :class:`ToolRegistry`. The
load pipeline is fail-closed and runs in order:

1. **validate** — schema version, SBOM digest integrity, and that every tool's
   required capabilities are a subset of the plugin's declared capabilities.
2. **verify** — the manifest signature. Signed plugin + no verifier backend =>
   refuse; signature mismatch / unknown key => refuse; unsigned +
   ``require_signature`` => refuse. Unsigned + ``require_signature=False`` =>
   load with a structured warning (trusted-local opt-in).
3. **capability-intersect** — effective = declared ∩ operator-granted. Tools
   whose ``requires`` exceed the effective set are dropped with a structured
   log (no silent truncation); if every tool is dropped, the plugin is refused.
4. **register** — the plugin's MCP server is launched and its tools registered
   (namespaced ``mcp__<plugin>__<tool>``), then each tool's ``requires`` and
   ``permission_level`` are overridden from the manifest.

The capability intersection (step 3) is the P4.24 meet applied to plugins:
``effective = plugin.capabilities ∧ operator_grant``. At runtime the subagent
authority gate intersects again, so a plugin tool is reachable only through
both the operator grant AND the delegating agent's authority.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

from nullain.authority import Capability
from nullain.errors import (
    PluginCapabilityError,
    PluginManifestError,
    PluginSbomError,
    PluginSignatureError,
)
from nullain.mcp import MCPClient, StdioTransport, register_mcp_tools
from nullain.plugins.manifest import MANIFEST_SCHEMA_VERSION, PluginManifest, PluginToolDecl
from nullain.plugins.signing import SignatureVerifier
from nullain.telemetry import get_logger
from nullain.tools import ToolRegistry

logger = get_logger(__name__)


@dataclass
class PreparedPlugin:
    """A verified, initialized plugin ready to register into session registries.

    Holds the live MCP client (the server subprocess persists across sessions)
    and the capability/declaration metadata needed for per-session registration.
    """

    manifest: PluginManifest
    client: MCPClient
    effective_capabilities: frozenset[Capability]
    declared: dict[str, PluginToolDecl]
    auto_approve: bool


class PluginLoader:
    """Validates, verifies, and registers signed plugin manifests."""

    def __init__(
        self,
        verifier: SignatureVerifier,
        *,
        require_signature: bool = True,
        trusted_keys: dict[str, str] | None = None,
        allowed_capabilities: frozenset[Capability] | None = None,
        auto_approve_default: bool = False,
    ) -> None:
        self._verifier = verifier
        self.require_signature = require_signature
        self.trusted_keys: dict[str, str] = dict(trusted_keys or ())
        self.allowed_capabilities: frozenset[Capability] = frozenset(allowed_capabilities or ())
        self.auto_approve_default = auto_approve_default

    # -- public API ---------------------------------------------------------

    async def load(self, manifest: PluginManifest, registry: ToolRegistry) -> list[str]:
        """Convenience: prepare + register in one call (used by tests/one-shot loaders)."""
        prepared = await self.prepare(manifest)
        try:
            return await self.register(prepared, registry)
        except Exception:
            with contextlib.suppress(Exception):
                await prepared.client.close()
            raise

    async def load_from_file(self, manifest_path: str | Path, registry: ToolRegistry) -> list[str]:
        """Load and register a plugin from a manifest JSON file."""
        manifest = self.parse_manifest_file(manifest_path)
        return await self.load(manifest, registry)

    async def prepare(self, manifest: PluginManifest) -> PreparedPlugin:
        """Validate + verify + initialize the plugin's MCP server (shared across sessions)."""
        self._validate(manifest)
        self._verify_signature(manifest)
        effective = self._effective_capabilities(manifest)
        transport = StdioTransport(
            command=manifest.transport.command,
            args=manifest.transport.args,
            env=manifest.transport.env,
        )
        client = MCPClient(transport=transport, name=manifest.name)
        try:
            await client.initialize()
        except Exception as err:
            with contextlib.suppress(Exception):
                await client.close()
            raise PluginManifestError(
                f"plugin '{manifest.name}' MCP server failed to initialize: {err}"
            ) from err
        declared = {t.name: t for t in manifest.tools}
        return PreparedPlugin(
            manifest=manifest,
            client=client,
            effective_capabilities=effective,
            declared=declared,
            auto_approve=self.auto_approve_default,
        )

    async def register(self, prepared: PreparedPlugin, registry: ToolRegistry) -> list[str]:
        """Register a prepared plugin's tools into a (per-session) registry."""
        names = await register_mcp_tools(
            registry, prepared.client, auto_approve=prepared.auto_approve
        )
        prefix = f"mcp__{prepared.manifest.name.replace('-', '_')}__"
        kept: list[str] = []
        for full_name in names:
            short = full_name.removeprefix(prefix)
            decl = prepared.declared.get(short)
            reg_tool = registry.get_tool(full_name)
            if decl is not None:
                requires = decl.requires
                level = decl.permission_level
            else:
                # Undeclared tool: conservative default (matches the MCP bridge).
                requires = frozenset({Capability.WRITE})
                level = None
            if not requires <= prepared.effective_capabilities:
                logger.warning(
                    "plugin_tool_dropped_over_capability",
                    plugin=prepared.manifest.name,
                    tool=short,
                    requires=sorted(c.value for c in requires),
                    granted=sorted(c.value for c in prepared.effective_capabilities),
                )
                registry.unregister(full_name)
                continue
            reg_tool.requires = requires
            if level is not None:
                reg_tool.permission_level = level
            kept.append(full_name)
        if not kept:
            raise PluginCapabilityError(
                f"plugin '{prepared.manifest.name}' has no tools usable under granted "
                f"capabilities {sorted(c.value for c in prepared.effective_capabilities)}"
            )
        return kept

    # -- parse --------------------------------------------------------------

    @staticmethod
    def parse_manifest_file(path: str | Path) -> PluginManifest:
        """Read and validate a manifest JSON file into a :class:`PluginManifest`."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            raise PluginManifestError(f"could not read manifest '{path}': {err}") from err
        return PluginManifest.model_validate(data)

    # -- pipeline steps -----------------------------------------------------

    def _validate(self, manifest: PluginManifest) -> None:
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            raise PluginManifestError(
                f"plugin '{manifest.name}' schema version '{manifest.schema_version}' "
                f"unsupported (expected '{MANIFEST_SCHEMA_VERSION}')"
            )
        if manifest.sbom.digest() != manifest.sbom_digest:
            raise PluginSbomError(
                f"plugin '{manifest.name}' SBOM digest mismatch: pinned "
                f"{manifest.sbom_digest} != computed {manifest.sbom.digest()}"
            )
        declared_caps = set(manifest.capabilities)
        for tool in manifest.tools:
            if not tool.requires <= declared_caps:
                raise PluginManifestError(
                    f"plugin '{manifest.name}' tool '{tool.name}' requires "
                    f"{sorted(c.value for c in tool.requires)} but the plugin declares only "
                    f"{sorted(c.value for c in declared_caps)} — a tool may not require a "
                    "capability the plugin did not declare"
                )

    def _verify_signature(self, manifest: PluginManifest) -> None:
        sig = manifest.signature
        if sig is None:
            if self.require_signature:
                raise PluginSignatureError(
                    f"plugin '{manifest.name}' is unsigned and require_signature=True "
                    "(set require_signature=false explicitly to load trusted-local plugins)"
                )
            logger.warning(
                "plugin_loaded_unsigned",
                plugin=manifest.name,
                hint="require_signature=false: only use for locally-trusted plugins",
            )
            return
        if not self._verifier.available():
            raise PluginSignatureError(
                f"plugin '{manifest.name}' is signed with algorithm '{sig.algorithm}' "
                "but no signature verifier backend is available "
                "(install nullain-sdk[signing] to verify signed plugins)"
            )
        if not self._verifier.verify(manifest, self.trusted_keys):
            raise PluginSignatureError(
                f"plugin '{manifest.name}' signature did not verify against the trusted key set "
                f"(key_id='{sig.key_id}')"
            )

    def _effective_capabilities(self, manifest: PluginManifest) -> frozenset[Capability]:
        """Operator grant ∩ plugin declaration (the P4.24 meet for plugins)."""
        return frozenset(set(manifest.capabilities) & self.allowed_capabilities)


__all__ = ["PluginLoader", "PreparedPlugin"]
