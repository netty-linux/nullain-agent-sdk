"""Unit tests for the plugin pipeline (P4.25): manifest, SBOM, signing, loader.

These tests are 100% offline. The trust invariants (validate, verify, capability
intersection, capability-drop on register) are exercised directly against the
loader's pipeline steps with a fake :class:`SignatureVerifier` (DI). The MCP
launch path is exercised with an in-memory :class:`FakeTransport` MCP client —
no subprocess is spawned. A gated Ed25519 test signs and verifies a real manifest
using the ``cryptography`` extra when it is installed.

The trust invariant under test — fail-closed at every branch:

* unsigned + ``require_signature=True``  => refuse
* signed + no verifier backend available  => refuse
* signed + verifier says no               => refuse
* SBOM digest mismatch                    => refuse
* tool requires ⊄ declared capabilities   => refuse
* operator grant ∩ declared drops a tool  => drop with log (no silent truncation)
* every tool dropped by the grant         => refuse
* otherwise                                => register with manifest requires/level
"""

from __future__ import annotations

import base64
import importlib.util
import json
from typing import Any

import pytest
from nullain.authority import Capability
from nullain.errors import (
    PluginCapabilityError,
    PluginManifestError,
    PluginSbomError,
    PluginSignatureError,
    ToolNotFoundError,
)
from nullain.mcp import MCPClient
from nullain.mcp.protocol import JSONRPCNotification, JSONRPCRequest
from nullain.plugins import (
    MANIFEST_SCHEMA_VERSION,
    SBOM,
    NoVerifier,
    PluginLoader,
    PluginManifest,
    PluginSignature,
    PluginToolDecl,
    PluginTransportDecl,
    PreparedPlugin,
    SBOMDependency,
    select_verifier,
)
from nullain.plugins.signing import Ed25519Verifier
from nullain.tools import ToolRegistry
from nullain.tools.permissions import PermissionLevel

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTransport:
    """In-memory MCP transport with scripted responses (mirrors test_mcp_client)."""

    def __init__(self, responses: dict[str, list[Any]]) -> None:
        self._responses: dict[str, list[Any]] = {k: list(v) for k, v in responses.items()}
        self.requests: list[JSONRPCRequest] = []
        self.notifications: list[JSONRPCNotification] = []
        self.closed = False

    async def start(self) -> None:
        """No-op."""

    async def send_request(self, request: JSONRPCRequest) -> str:
        self.requests.append(request)
        queue = self._responses.get(request.method)
        if not queue:
            raise AssertionError(f"no scripted response for {request.method}")
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return json.dumps({"jsonrpc": "2.0", "id": request.id, "result": result})

    async def send_notification(self, notification: JSONRPCNotification) -> None:
        self.notifications.append(notification)

    async def close(self) -> None:
        self.closed = True


class FakeVerifier:
    """DI signature verifier with scripted availability/result."""

    name = "fake"

    def __init__(self, *, available: bool = True, result: bool = True) -> None:
        self._available = available
        self._result = result
        self.verify_calls = 0

    def available(self) -> bool:
        return self._available

    def verify(self, manifest: PluginManifest, trusted_keys: dict[str, str]) -> bool:
        self.verify_calls += 1
        return self._result


def _ok_init_result() -> dict[str, Any]:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "serverInfo": {"name": "fake-server", "version": "0.0.1"},
    }


def _tools_list_result(names: list[str]) -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": n,
                "description": f"tool {n}",
                "inputSchema": {"type": "object", "properties": {}},
            }
            for n in names
        ]
    }


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _sbom(deps: list[SBOMDependency] | None = None) -> SBOM:
    return SBOM(dependencies=deps or [])


def _manifest(
    *,
    name: str = "demo",
    capabilities: frozenset[Capability] = frozenset({Capability.READ, Capability.WRITE}),
    tools: list[PluginToolDecl] | None = None,
    sbom: SBOM | None = None,
    sbom_digest: str | None = None,
    signature: PluginSignature | None = None,
    transport: PluginTransportDecl | None = None,
    schema_version: str = MANIFEST_SCHEMA_VERSION,
) -> PluginManifest:
    sbom = sbom if sbom is not None else _sbom()
    return PluginManifest(
        schema_version=schema_version,
        name=name,
        version="0.1.0",
        publisher="test",
        transport=transport or PluginTransportDecl(command="python", args=["-c", "pass"]),
        capabilities=capabilities,
        tools=tools or [],
        sbom=sbom,
        sbom_digest=sbom_digest if sbom_digest is not None else sbom.digest(),
        signature=signature,
    )


def _signed_manifest(
    *, key_id: str = "k1", **kwargs: Any
) -> tuple[PluginManifest, PluginSignature]:
    """Build a manifest carrying a placeholder signature object (not cryptographically valid)."""
    sig = PluginSignature(key_id=key_id, value="fakesig")
    return _manifest(signature=sig, **kwargs), sig


def _client_with_tools(tool_names: list[str], name: str = "demo") -> MCPClient:
    transport = FakeTransport(
        {
            "initialize": [_ok_init_result()],
            "tools/list": [_tools_list_result(tool_names)],
        }
    )
    return MCPClient(transport=transport, name=name)


# ---------------------------------------------------------------------------
# SBOM / manifest data model
# ---------------------------------------------------------------------------


def test_sbom_digest_is_deterministic_and_order_independent() -> None:
    a = _sbom([SBOMDependency(name="z", version="1"), SBOMDependency(name="a", version="2")])
    b = _sbom([SBOMDependency(name="a", version="2"), SBOMDependency(name="z", version="1")])
    assert a.digest() == b.digest()
    assert a.digest().startswith("sha256:")


def test_signed_message_excludes_signature_and_collapses_sbom() -> None:
    manifest, _ = _signed_manifest()
    payload = json.loads(manifest.signed_message())
    assert "signature" not in payload
    assert payload["sbom"] == manifest.sbom_digest


def test_signed_message_changes_when_transport_command_tampered() -> None:
    m1, _ = _signed_manifest()
    m2 = _manifest(
        signature=PluginSignature(key_id="k1", value="fakesig"),
        transport=PluginTransportDecl(command="evil", args=[]),
    )
    assert m1.signed_message() != m2.signed_message()


# ---------------------------------------------------------------------------
# validate: schema / SBOM / tool-requires-subset
# ---------------------------------------------------------------------------


def test_validate_rejects_unknown_schema_version() -> None:
    loader = PluginLoader(FakeVerifier(), allowed_capabilities=frozenset(Capability))
    manifest = _manifest(schema_version="nullain-plugin/9.9")
    with pytest.raises(PluginManifestError, match="schema version"):
        loader._validate(manifest)  # type: ignore[reportPrivateUsage]


def test_validate_rejects_sbom_digest_mismatch() -> None:
    loader = PluginLoader(FakeVerifier(), allowed_capabilities=frozenset(Capability))
    manifest = _manifest(sbom_digest="sha256:deadbeef")
    with pytest.raises(PluginSbomError, match="SBOM digest mismatch"):
        loader._validate(manifest)  # type: ignore[reportPrivateUsage]


def test_validate_rejects_tool_requiring_undeclared_capability() -> None:
    loader = PluginLoader(FakeVerifier(), allowed_capabilities=frozenset(Capability))
    manifest = _manifest(
        capabilities=frozenset({Capability.READ}),
        tools=[PluginToolDecl(name="t", requires=frozenset({Capability.WRITE}))],
    )
    with pytest.raises(PluginManifestError, match="may not require a capability"):
        loader._validate(manifest)  # type: ignore[reportPrivateUsage]


def test_validate_accepts_well_formed_manifest() -> None:
    loader = PluginLoader(FakeVerifier(), allowed_capabilities=frozenset(Capability))
    manifest = _manifest(
        capabilities=frozenset({Capability.READ, Capability.WRITE}),
        tools=[PluginToolDecl(name="t", requires=frozenset({Capability.WRITE}))],
    )
    loader._validate(manifest)  # no raise  # type: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# verify: fail-closed signature policy
# ---------------------------------------------------------------------------


def test_verify_refuses_unsigned_when_signature_required() -> None:
    loader = PluginLoader(
        FakeVerifier(), require_signature=True, allowed_capabilities=frozenset(Capability)
    )
    with pytest.raises(PluginSignatureError, match="unsigned"):
        loader._verify_signature(_manifest())  # type: ignore[reportPrivateUsage]


def test_verify_allows_unsigned_when_opted_in() -> None:
    loader = PluginLoader(
        FakeVerifier(), require_signature=False, allowed_capabilities=frozenset(Capability)
    )
    loader._verify_signature(_manifest())  # no raise  # type: ignore[reportPrivateUsage]


def test_verify_refuses_signed_when_no_backend_available() -> None:
    loader = PluginLoader(NoVerifier(), allowed_capabilities=frozenset(Capability))
    manifest, _ = _signed_manifest()
    with pytest.raises(PluginSignatureError, match="no signature verifier backend"):
        loader._verify_signature(manifest)  # type: ignore[reportPrivateUsage]


def test_verify_refuses_when_verifier_says_no() -> None:
    loader = PluginLoader(
        FakeVerifier(available=True, result=False), allowed_capabilities=frozenset(Capability)
    )
    manifest, _ = _signed_manifest()
    with pytest.raises(PluginSignatureError, match="did not verify"):
        loader._verify_signature(manifest)  # type: ignore[reportPrivateUsage]


def test_verify_accepts_when_verifier_says_yes() -> None:
    verifier = FakeVerifier(available=True, result=True)
    loader = PluginLoader(verifier, allowed_capabilities=frozenset(Capability))
    manifest, _ = _signed_manifest()
    loader._verify_signature(manifest)  # type: ignore[reportPrivateUsage]
    assert verifier.verify_calls == 1


# ---------------------------------------------------------------------------
# capability intersection (P4.24 meet applied to plugins)
# ---------------------------------------------------------------------------


def test_effective_capabilities_is_operator_grant_intersect_declared() -> None:
    loader = PluginLoader(
        FakeVerifier(),
        allowed_capabilities=frozenset({Capability.READ, Capability.NETWORK}),
    )
    manifest = _manifest(capabilities=frozenset({Capability.READ, Capability.WRITE}))
    assert loader._effective_capabilities(manifest) == frozenset({Capability.READ})  # type: ignore[reportPrivateUsage]


def test_effective_capabilities_empty_when_grant_excludes_all_declared() -> None:
    loader = PluginLoader(
        FakeVerifier(),
        allowed_capabilities=frozenset({Capability.EXEC}),
    )
    manifest = _manifest(capabilities=frozenset({Capability.READ, Capability.WRITE}))
    assert loader._effective_capabilities(manifest) == frozenset()  # type: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# register: capability-drop, requires/level override, all-dropped refusal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_drops_over_capable_tool_and_keeps_rest() -> None:
    loader = PluginLoader(
        FakeVerifier(),
        allowed_capabilities=frozenset({Capability.READ}),  # grants READ only
    )
    tools = [
        PluginToolDecl(name="reader", requires=frozenset({Capability.READ})),
        PluginToolDecl(name="writer", requires=frozenset({Capability.WRITE})),
    ]
    manifest = _manifest(capabilities=frozenset({Capability.READ, Capability.WRITE}), tools=tools)
    client = _client_with_tools(["reader", "writer"])
    prepared = PreparedPlugin(
        manifest=manifest,
        client=client,
        effective_capabilities=frozenset({Capability.READ}),
        declared={t.name: t for t in tools},
        auto_approve=True,
    )
    registry = ToolRegistry()
    kept = await loader.register(prepared, registry)
    assert kept == ["mcp__demo__reader"]
    assert registry.get_tool("mcp__demo__reader").requires == frozenset({Capability.READ})
    # the over-capable tool was dropped, not left as a latent over-grant
    with pytest.raises(ToolNotFoundError):
        registry.get_tool("mcp__demo__writer")
    await prepared.client.close()


@pytest.mark.asyncio
async def test_register_applies_manifest_permission_level_override() -> None:
    loader = PluginLoader(FakeVerifier(), allowed_capabilities=frozenset(Capability))
    tools = [
        PluginToolDecl(
            name="t", requires=frozenset({Capability.READ}), permission_level=PermissionLevel.ALLOW
        )
    ]
    manifest = _manifest(capabilities=frozenset({Capability.READ}), tools=tools)
    client = _client_with_tools(["t"])
    prepared = PreparedPlugin(
        manifest=manifest,
        client=client,
        effective_capabilities=frozenset({Capability.READ}),
        declared={t.name: t for t in tools},
        auto_approve=False,
    )
    registry = ToolRegistry()
    await loader.register(prepared, registry)
    tool = registry.get_tool("mcp__demo__t")
    assert tool.requires == frozenset({Capability.READ})
    assert tool.permission_level == PermissionLevel.ALLOW
    await prepared.client.close()


@pytest.mark.asyncio
async def test_register_refuses_when_every_tool_dropped() -> None:
    loader = PluginLoader(FakeVerifier(), allowed_capabilities=frozenset({Capability.READ}))
    tools = [
        PluginToolDecl(name="w1", requires=frozenset({Capability.WRITE})),
        PluginToolDecl(name="w2", requires=frozenset({Capability.WRITE})),
    ]
    manifest = _manifest(capabilities=frozenset({Capability.WRITE}), tools=tools)
    client = _client_with_tools(["w1", "w2"])
    prepared = PreparedPlugin(
        manifest=manifest,
        client=client,
        effective_capabilities=frozenset(),  # grant intersect declared => empty
        declared={t.name: t for t in tools},
        auto_approve=True,
    )
    registry = ToolRegistry()
    with pytest.raises(PluginCapabilityError, match="no tools usable"):
        await loader.register(prepared, registry)
    await prepared.client.close()


@pytest.mark.asyncio
async def test_register_defaults_undeclared_tool_to_write_capability() -> None:
    """A tool the manifest didn't declare is conservatively treated as WRITE."""
    loader = PluginLoader(FakeVerifier(), allowed_capabilities=frozenset(Capability))
    manifest = _manifest(capabilities=frozenset({Capability.WRITE}), tools=[])  # no decls
    client = _client_with_tools(["surprise"])  # server exposes an undeclared tool
    prepared = PreparedPlugin(
        manifest=manifest,
        client=client,
        effective_capabilities=frozenset({Capability.WRITE}),
        declared={},
        auto_approve=True,
    )
    registry = ToolRegistry()
    kept = await loader.register(prepared, registry)
    assert kept == ["mcp__demo__surprise"]
    assert registry.get_tool("mcp__demo__surprise").requires == frozenset({Capability.WRITE})
    await prepared.client.close()


# ---------------------------------------------------------------------------
# verifier selection
# ---------------------------------------------------------------------------


def test_select_verifier_picks_no_verifier_when_cryptography_absent() -> None:
    # The selector prefers Ed25519 when available; this asserts the fallback
    # shape (NoVerifier) is fail-closed regardless of host install state.
    verifier = select_verifier()
    if importlib.util.find_spec("cryptography") is None:
        assert verifier.name == "none"
        assert verifier.available() is False
    else:
        assert verifier.name == "ed25519"


# ---------------------------------------------------------------------------
# Ed25519 sign + verify (gated on the optional cryptography extra)
# ---------------------------------------------------------------------------

_HAS_CRYPTO = importlib.util.find_spec("cryptography") is not None


def _ed25519_keypair() -> tuple[Any, bytes]:
    """Generate an Ed25519 keypair via importlib.

    The optional ``cryptography`` extra is resolved at runtime (not a static
    import) so the test stays clean under pyright strict when the extra is
    absent — mirroring :mod:`nullain.plugins.signing`.
    """
    ed25519 = importlib.import_module("cryptography.hazmat.primitives.asymmetric.ed25519")
    serialization = importlib.import_module("cryptography.hazmat.primitives.serialization")
    priv: Any = ed25519.Ed25519PrivateKey.generate()
    pub_bytes: bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv, pub_bytes


def _sign_manifest(priv: Any, manifest: PluginManifest, key_id: str) -> None:
    """Sign ``manifest.signed_message()`` and attach the signature in-place."""
    sig_bytes: bytes = priv.sign(manifest.signed_message())
    manifest.signature = PluginSignature(
        key_id=key_id,
        value=base64.b64encode(sig_bytes).decode("ascii"),
    )


@pytest.mark.skipif(not _HAS_CRYPTO, reason="cryptography extra not installed")
def test_ed25519_roundtrip_signs_and_verifies() -> None:
    priv, pub_bytes = _ed25519_keypair()
    manifest = _manifest(
        capabilities=frozenset({Capability.READ}),
        tools=[PluginToolDecl(name="t", requires=frozenset({Capability.READ}))],
    )
    _sign_manifest(priv, manifest, "k1")
    verifier = Ed25519Verifier()
    assert verifier.available() is True
    assert verifier.verify(manifest, {"k1": base64.b64encode(pub_bytes).decode("ascii")}) is True


@pytest.mark.skipif(not _HAS_CRYPTO, reason="cryptography extra not installed")
def test_ed25519_rejects_tampered_manifest() -> None:
    priv, pub_bytes = _ed25519_keypair()
    pub_b64 = base64.b64encode(pub_bytes).decode("ascii")
    manifest = _manifest(
        capabilities=frozenset({Capability.READ}),
        tools=[PluginToolDecl(name="t", requires=frozenset({Capability.READ}))],
    )
    _sign_manifest(priv, manifest, "k1")
    # tamper after signing — the signature no longer covers this manifest
    manifest.publisher = "evil"
    verifier = Ed25519Verifier()
    assert verifier.verify(manifest, {"k1": pub_b64}) is False


@pytest.mark.skipif(not _HAS_CRYPTO, reason="cryptography extra not installed")
def test_ed25519_rejects_unknown_key_id() -> None:
    priv, pub_bytes = _ed25519_keypair()
    pub_b64 = base64.b64encode(pub_bytes).decode("ascii")
    manifest = _manifest()
    _sign_manifest(priv, manifest, "unknown")
    verifier = Ed25519Verifier()
    # trusted_keys has the key under a different id => refuse
    assert verifier.verify(manifest, {"other": pub_b64}) is False
