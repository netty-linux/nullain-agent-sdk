"""Nullain Agent SDK — Plugin Signature Verification (P4.25).

Hexagonal: a :class:`SignatureVerifier` port with an optional Ed25519 adapter
(the ``nullain-sdk[signing]`` extra, backed by ``cryptography``) and a
fail-closed :class:`NoVerifier`. The selector returns the best available
verifier, falling back to ``NoVerifier`` — so a signed plugin is REFUSED when no
backend is installed, exactly mirroring the P4.23 sandbox discipline
(fail-closed, never silently unverified).
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
from typing import Any, Protocol, runtime_checkable

from nullain.plugins.manifest import PluginManifest


@runtime_checkable
class SignatureVerifier(Protocol):
    """Port: verify a plugin manifest's signature against a trusted key set."""

    name: str

    def available(self) -> bool:
        """Whether this verifier's backend is usable on the current host."""
        ...

    def verify(self, manifest: PluginManifest, trusted_keys: dict[str, str]) -> bool:
        """Verify ``manifest.signature`` against ``trusted_keys`` (key_id -> base64 public key).

        Returns True iff the signature is valid and the key_id is trusted.
        Returns False on signature mismatch, unknown key id, unsupported
        algorithm, or a missing signature (callers handle the unsigned-policy
        branch before calling). MUST NOT raise on mismatch.
        """
        ...


class NoVerifier:
    """Fail-closed verifier: never available, never verifies.

    Selected when no signing backend is installed. A signed plugin under
    :class:`NoVerifier` is refused by the loader (cannot be verified) rather
    than loaded on trust — the safe default.
    """

    name = "none"

    def available(self) -> bool:
        return False

    def verify(self, manifest: PluginManifest, trusted_keys: dict[str, str]) -> bool:
        return False


class Ed25519Verifier:
    """Ed25519 signature verifier using the optional ``cryptography`` extra.

    Not a hard dependency: ``available()`` probes for ``cryptography`` and
    returns False when it is absent, so the SDK installs cleanly without it.
    Install ``nullain-sdk[signing]`` to enable signed-plugin loading.
    """

    name = "ed25519"

    def available(self) -> bool:
        return importlib.util.find_spec("cryptography") is not None

    def verify(self, manifest: PluginManifest, trusted_keys: dict[str, str]) -> bool:
        sig = manifest.signature
        if sig is None or sig.algorithm != "ed25519":
            return False
        pubkey_b64 = trusted_keys.get(sig.key_id)
        if pubkey_b64 is None:
            return False  # unknown / untrusted key id
        # importlib (not a static import) so the optional `cryptography` extra
        # is resolved at runtime and does not trip static analysis when absent.
        try:
            ed25519_mod = importlib.import_module(
                "cryptography.hazmat.primitives.asymmetric.ed25519"
            )
            exceptions_mod = importlib.import_module("cryptography.exceptions")
        except ImportError:
            return False
        invalid_signature: Any = exceptions_mod.InvalidSignature
        ed25519_public_key: Any = ed25519_mod.Ed25519PublicKey
        try:
            public_key = ed25519_public_key.from_public_bytes(base64.b64decode(pubkey_b64))
            public_key.verify(base64.b64decode(sig.value), manifest.signed_message())
        except (invalid_signature, ValueError):
            return False
        return True


def select_verifier() -> SignatureVerifier:
    """Pick the best available verifier; fail-closed to :class:`NoVerifier`."""
    ed25519 = Ed25519Verifier()
    if ed25519.available():
        return ed25519
    return NoVerifier()


__all__ = [
    "Ed25519Verifier",
    "NoVerifier",
    "SignatureVerifier",
    "select_verifier",
]
