"""Unit tests for the daemon's plugin wiring (P4.25): capability parsing and
the fail-closed load pipeline.

These are 100% offline. ``_cap_set`` is a pure function. ``_load_plugins`` is
exercised only along branches that fail BEFORE any MCP subprocess is spawned:

* plugins disabled            => empty list, no work
* a missing/unreadable manifest => the entry is skipped with a structured log
  (``parse_manifest_file`` raises before ``prepare`` ever constructs a transport)

The successful prepare path spawns a real stdio MCP subprocess and is covered by
the in-memory ``FakeTransport`` tests in ``test_plugins.py`` instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nullain.authority import Capability
from nullain.config import NullainSettings, PluginEntryConfig, PluginsConfig
from nullain_agentd.main import _cap_set, _load_plugins  # type: ignore[reportPrivateUsage]


def test_cap_set_universe_when_unset() -> None:
    assert _cap_set(None) == frozenset(Capability)
    assert _cap_set([]) == frozenset(Capability)


def test_cap_set_parses_lowercase_names() -> None:
    assert _cap_set(["read", "write"]) == frozenset({Capability.READ, Capability.WRITE})


def test_cap_set_is_case_insensitive() -> None:
    assert _cap_set(["READ", "Network"]) == frozenset({Capability.READ, Capability.NETWORK})


@pytest.mark.asyncio
async def test_load_plugins_returns_empty_when_disabled() -> None:
    settings = NullainSettings(plugins=PluginsConfig(enabled=False))
    assert await _load_plugins(settings) == []


@pytest.mark.asyncio
async def test_load_plugins_skips_missing_manifest_with_log(tmp_path: Path) -> None:
    # A manifest path that does not exist: parse fails before any subprocess is
    # spawned, so the entry is skipped (fail-closed) and the daemon continues.
    settings = NullainSettings(
        plugins=PluginsConfig(
            enabled=True,
            require_signature=False,
            entries={"ghost": PluginEntryConfig(manifest=str(tmp_path / "nope.json"))},
        )
    )
    prepared = await _load_plugins(settings)
    assert prepared == []


@pytest.mark.asyncio
async def test_load_plugins_skips_disabled_entries(tmp_path: Path) -> None:
    settings = NullainSettings(
        plugins=PluginsConfig(
            enabled=True,
            entries={
                "off": PluginEntryConfig(manifest=str(tmp_path / "nope.json"), enabled=False),
            },
        )
    )
    assert await _load_plugins(settings) == []
