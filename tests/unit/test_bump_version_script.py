"""Unit tests for scripts/bump_version.py.

Runs against a temporary copy of the three packages' pyproject.toml files
laid out in the same relative structure the real script expects, rather
than mocking file I/O — the bug this script exists to prevent (a second
transform on the same file silently discarding the first) only shows up
against a real accumulate-then-write flow, not a mocked one.

scripts/ isn't on pytest's pythonpath (it's a one-off utility, not part of
the package), so the module is loaded directly from its file path.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "bump_version.py"


def _load_module(repo_root: Path) -> types.ModuleType:
    """Load bump_version.py fresh, pointed at ``repo_root`` instead of the
    real repository — REPO_ROOT is computed at import time from
    ``__file__``, so each test gets an isolated module instance rather than
    monkeypatching a shared cached import."""
    spec = importlib.util.spec_from_file_location("bump_version_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # REPO_ROOT and everything derived from it (PACKAGES, INTERNAL_PINS) was
    # computed against the real repo on exec — rebuild it against the test
    # fixture's root instead. types.ModuleType has no static attributes for
    # any of this (they're module-level names created dynamically by
    # exec_module), hence the ignores below rather than a false structural
    # type for a dynamically-loaded module.
    module.REPO_ROOT = repo_root  # pyright: ignore[reportAttributeAccessIssue]
    module.PACKAGES = {  # pyright: ignore[reportAttributeAccessIssue]
        "nullain-sdk": repo_root / "nullain-sdk" / "pyproject.toml",
        "nullain-tools": repo_root / "nullain-tools" / "pyproject.toml",
        "nullain-agentd": repo_root / "nullain-agentd" / "pyproject.toml",
    }
    module.INTERNAL_PINS = [  # pyright: ignore[reportAttributeAccessIssue]
        (repo_root / "nullain-sdk" / "pyproject.toml", "nullain-tools"),
        (repo_root / "nullain-tools" / "pyproject.toml", "nullain-sdk"),
        (repo_root / "nullain-agentd" / "pyproject.toml", "nullain-sdk"),
        (repo_root / "nullain-agentd" / "pyproject.toml", "nullain-tools"),
    ]
    return module


def _write_fixture_repo(
    root: Path,
    *,
    sdk_version: str = "0.7.1",
    tools_version: str = "0.4.0",
    agentd_version: str = "0.1.0",
) -> None:
    """Lay out the three packages' pyproject.toml + internal pins, matching
    the real monorepo's independent-versioning structure: each package has
    its own version, and pins on other packages start deliberately stale
    (mirroring nullain-tools/nullain-agentd's real >=0.1.0 pin on
    nullain-sdk while nullain-sdk itself has moved on)."""
    (root / "pyproject.toml").write_text(
        '[project]\nname = "nullain-monorepo"\nversion = "0.1.0"\n', encoding="utf-8"
    )

    sdk = root / "nullain-sdk"
    sdk.mkdir(parents=True)
    (sdk / "pyproject.toml").write_text(
        f'[project]\nname = "nullain-sdk"\nversion = "{sdk_version}"\n'
        f'dependencies = [\n    "nullain-tools>={tools_version}",\n]\n',
        encoding="utf-8",
    )

    tools = root / "nullain-tools"
    tools.mkdir(parents=True)
    (tools / "pyproject.toml").write_text(
        f'[project]\nname = "nullain-tools"\nversion = "{tools_version}"\n'
        f'dependencies = [\n    "nullain-sdk>=0.1.0",\n]\n',
        encoding="utf-8",
    )

    agentd = root / "nullain-agentd"
    agentd.mkdir(parents=True)
    (agentd / "pyproject.toml").write_text(
        f'[project]\nname = "nullain-agentd"\nversion = "{agentd_version}"\n'
        f'dependencies = [\n    "nullain-sdk>=0.1.0",\n    "nullain-tools>=0.1.0",\n]\n',
        encoding="utf-8",
    )


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    _write_fixture_repo(tmp_path)
    return tmp_path


def test_current_version_reads_the_named_package(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    assert mod._current_version(mod.PACKAGES["nullain-sdk"]) == "0.7.1"
    assert mod._current_version(mod.PACKAGES["nullain-tools"]) == "0.4.0"


def test_compute_changes_only_touches_the_named_package_by_default(fixture_repo: Path) -> None:
    """Without --bump-dependents, bumping nullain-sdk must not touch
    nullain-tools/nullain-agentd's pins on it — that's the whole point of
    making dependent-pin bumps opt-in."""
    mod = _load_module(fixture_repo)
    changes = mod._compute_changes("nullain-sdk", "0.8.0", bump_dependents=False)

    assert len(changes) == 1
    sdk_content = changes[fixture_repo / "nullain-sdk" / "pyproject.toml"]
    assert 'version = "0.8.0"' in sdk_content


def test_compute_changes_with_bump_dependents_updates_all_pins(fixture_repo: Path) -> None:
    """Regression shape carried over from the old synchronized-bump test:
    a file targeted by MULTIPLE transforms (nullain-agentd/pyproject.toml
    pins both nullain-sdk and nullain-tools) must apply every transform on
    the ACCUMULATED content, not silently discard all but the last."""
    mod = _load_module(fixture_repo)
    changes = mod._compute_changes("nullain-sdk", "0.8.0", bump_dependents=True)

    assert len(changes) == 3  # sdk's own pyproject + tools' pin + agentd's pin

    sdk_content = changes[fixture_repo / "nullain-sdk" / "pyproject.toml"]
    assert 'version = "0.8.0"' in sdk_content

    tools_content = changes[fixture_repo / "nullain-tools" / "pyproject.toml"]
    assert '"nullain-sdk>=0.8.0"' in tools_content

    agentd_content = changes[fixture_repo / "nullain-agentd" / "pyproject.toml"]
    assert '"nullain-sdk>=0.8.0"' in agentd_content
    # agentd's OTHER pin (nullain-tools) must survive untouched.
    assert '"nullain-tools>=0.1.0"' in agentd_content


def test_pins_referencing_finds_stale_pins_excluding_self(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    pins = mod._pins_referencing("nullain-sdk")
    paths = {p for p, _version in pins}
    assert paths == {
        fixture_repo / "nullain-tools" / "pyproject.toml",
        fixture_repo / "nullain-agentd" / "pyproject.toml",
    }
    # nullain-sdk's own pyproject.toml pins nullain-tools, not itself —
    # must never appear when asking "who pins nullain-sdk".
    assert fixture_repo / "nullain-sdk" / "pyproject.toml" not in paths


def test_apply_writes_only_the_named_package_file(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    changes = mod._compute_changes("nullain-tools", "0.5.0", bump_dependents=False)
    for path, new_content in changes.items():
        path.write_text(new_content, encoding="utf-8")

    assert "0.5.0" in (fixture_repo / "nullain-tools" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    # sdk/agentd's pins on nullain-tools were NOT touched.
    assert "0.5.0" not in (fixture_repo / "nullain-sdk" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "0.5.0" not in (fixture_repo / "nullain-agentd" / "pyproject.toml").read_text(
        encoding="utf-8"
    )


def test_bump_dependents_skips_packages_with_no_pin_at_all(fixture_repo: Path) -> None:
    """`INTERNAL_PINS` lists every POSSIBLE pin relationship, not a promise
    that each one exists in every fixture — `_pins_referencing` filters to
    only files that actually pin the package, so --bump-dependents must
    silently skip a package that has no pin on this one rather than error."""
    mod = _load_module(fixture_repo)
    sdk_pyproject = fixture_repo / "nullain-sdk" / "pyproject.toml"
    text = sdk_pyproject.read_text(encoding="utf-8")
    sdk_pyproject.write_text(text.replace('"nullain-tools>=0.4.0",\n', ""), encoding="utf-8")

    # nullain-sdk no longer pins nullain-tools at all — bumping
    # nullain-tools with --bump-dependents must not raise just because one
    # of the (up to) two possible pinning files doesn't have the pin.
    changes = mod._compute_changes("nullain-tools", "0.5.0", bump_dependents=True)
    assert fixture_repo / "nullain-sdk" / "pyproject.toml" not in changes
    assert fixture_repo / "nullain-agentd" / "pyproject.toml" in changes


def test_semver_regex_accepts_valid_and_rejects_invalid() -> None:
    spec = importlib.util.spec_from_file_location("bump_version_semver_check", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for valid in ("0.1.0", "1.0.0", "10.20.30", "1.0.0rc1", "1.0.0a1", "1.0.0b2"):
        assert mod._SEMVER_RE.match(valid), valid

    for invalid in ("0.1", "v0.1.0", "0.1.0.1", "latest", ""):
        assert not mod._SEMVER_RE.match(invalid), invalid


def test_main_rejects_invalid_version_string(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    sys.argv = ["bump_version.py", "nullain-sdk", "not-a-version"]
    with pytest.raises(SystemExit):
        mod.main()


def test_main_rejects_unknown_package(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    sys.argv = ["bump_version.py", "not-a-real-package", "0.8.0"]
    with pytest.raises(SystemExit):
        mod.main()


def test_main_rejects_same_as_current_version(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    sys.argv = ["bump_version.py", "nullain-sdk", "0.7.1"]
    with pytest.raises(SystemExit, match="already nullain-sdk's current version"):
        mod.main()


def test_main_dry_run_does_not_write_files(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    sys.argv = ["bump_version.py", "nullain-sdk", "0.8.0"]
    mod.main()

    text = (fixture_repo / "nullain-sdk" / "pyproject.toml").read_text(encoding="utf-8")
    assert '"0.7.1"' in text  # unchanged — no --apply flag


def test_main_dry_run_warns_about_stale_dependent_pins(
    fixture_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_module(fixture_repo)
    sys.argv = ["bump_version.py", "nullain-sdk", "0.8.0"]
    mod.main()

    out = capsys.readouterr().out
    assert "other packages pin nullain-sdk" in out
    assert "--bump-dependents" in out


def test_main_apply_without_bump_dependents_only_writes_own_file(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    sys.argv = ["bump_version.py", "nullain-sdk", "0.8.0", "--apply"]
    mod.main()

    assert "0.8.0" in (fixture_repo / "nullain-sdk" / "pyproject.toml").read_text(encoding="utf-8")
    assert "0.8.0" not in (fixture_repo / "nullain-tools" / "pyproject.toml").read_text(
        encoding="utf-8"
    )


def test_main_apply_with_bump_dependents_writes_pins_too(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    sys.argv = ["bump_version.py", "nullain-sdk", "0.8.0", "--apply", "--bump-dependents"]
    mod.main()

    assert "0.8.0" in (fixture_repo / "nullain-sdk" / "pyproject.toml").read_text(encoding="utf-8")
    assert '"nullain-sdk>=0.8.0"' in (fixture_repo / "nullain-tools" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '"nullain-sdk>=0.8.0"' in (fixture_repo / "nullain-agentd" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
