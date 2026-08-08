"""Unit tests for scripts/bump_version.py.

Runs against a temporary copy of the monorepo's 7 version-carrying files
(4 pyproject.toml + 3 __init__.py) laid out in the same relative structure
the real script expects, rather than mocking file I/O — the bug this
script exists to prevent (a second transform on the same file silently
discarding the first) only shows up against a real accumulate-then-write
flow, not a mocked one.

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
    # REPO_ROOT and everything derived from it (VERSIONED_FILES, etc.) was
    # computed against the real repo on exec — rebuild it against the test
    # fixture's root instead. types.ModuleType has no static attributes for
    # any of this (they're module-level names created dynamically by
    # exec_module), hence the ignores below rather than a false structural
    # type for a dynamically-loaded module.
    module.REPO_ROOT = repo_root  # pyright: ignore[reportAttributeAccessIssue]
    module.PYPROJECT_FILES = [  # pyright: ignore[reportAttributeAccessIssue]
        repo_root / "pyproject.toml",
        repo_root / "nullain-sdk" / "pyproject.toml",
        repo_root / "nullain-tools" / "pyproject.toml",
        repo_root / "nullain-agentd" / "pyproject.toml",
    ]
    module.DUNDER_VERSION_FILES = [  # pyright: ignore[reportAttributeAccessIssue]
        repo_root / "nullain-sdk" / "src" / "nullain" / "__init__.py",
        repo_root / "nullain-tools" / "src" / "nullain_tools" / "__init__.py",
        repo_root / "nullain-agentd" / "src" / "nullain_agentd" / "__init__.py",
    ]
    module.VERSIONED_FILES = (  # pyright: ignore[reportAttributeAccessIssue]
        module.PYPROJECT_FILES + module.DUNDER_VERSION_FILES  # pyright: ignore[reportAttributeAccessIssue]
    )
    module.INTERNAL_PINS = [  # pyright: ignore[reportAttributeAccessIssue]
        (repo_root / "nullain-sdk" / "pyproject.toml", "nullain-tools"),
        (repo_root / "nullain-tools" / "pyproject.toml", "nullain-sdk"),
        (repo_root / "nullain-agentd" / "pyproject.toml", "nullain-sdk"),
        (repo_root / "nullain-agentd" / "pyproject.toml", "nullain-tools"),
    ]
    module._MARKER_RE = {  # pyright: ignore[reportAttributeAccessIssue]
        **{
            p: module.re.compile(r'^version = "[^"]*"$', module.re.MULTILINE)  # pyright: ignore[reportAttributeAccessIssue]
            for p in module.PYPROJECT_FILES  # pyright: ignore[reportAttributeAccessIssue]
        },
        **{
            p: module.re.compile(r'^__version__ = "[^"]*"$', module.re.MULTILINE)  # pyright: ignore[reportAttributeAccessIssue]
            for p in module.DUNDER_VERSION_FILES  # pyright: ignore[reportAttributeAccessIssue]
        },
    }
    return module


def _write_fixture_repo(root: Path, version: str = "0.1.0") -> None:
    """Lay out the 7 version-carrying files + their internal pins, matching
    the real monorepo's structure closely enough for the script's path
    logic to work unmodified."""
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "nullain-monorepo"\nversion = "{version}"\n', encoding="utf-8"
    )

    sdk = root / "nullain-sdk"
    (sdk / "src" / "nullain").mkdir(parents=True)
    (sdk / "pyproject.toml").write_text(
        f'[project]\nname = "nullain-sdk"\nversion = "{version}"\n'
        f'dependencies = [\n    "nullain-tools>={version}",\n]\n',
        encoding="utf-8",
    )
    (sdk / "src" / "nullain" / "__init__.py").write_text(
        f'"""Docstring."""\n\n__version__ = "{version}"\n', encoding="utf-8"
    )

    tools = root / "nullain-tools"
    (tools / "src" / "nullain_tools").mkdir(parents=True)
    (tools / "pyproject.toml").write_text(
        f'[project]\nname = "nullain-tools"\nversion = "{version}"\n'
        f'dependencies = [\n    "nullain-sdk>={version}",\n]\n',
        encoding="utf-8",
    )
    (tools / "src" / "nullain_tools" / "__init__.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )

    agentd = root / "nullain-agentd"
    (agentd / "src" / "nullain_agentd").mkdir(parents=True)
    (agentd / "pyproject.toml").write_text(
        f'[project]\nname = "nullain-agentd"\nversion = "{version}"\n'
        f'dependencies = [\n    "nullain-sdk>={version}",\n    "nullain-tools>={version}",\n]\n',
        encoding="utf-8",
    )
    (agentd / "src" / "nullain_agentd" / "__init__.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    _write_fixture_repo(tmp_path)
    return tmp_path


def test_current_version_reads_root_pyproject(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    assert mod._current_version() == "0.1.0"


def test_check_all_synced_passes_when_consistent(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    mod._check_all_synced("0.1.0")  # must not raise


def test_check_all_synced_rejects_a_mismatched_file(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    # Desync one file, mirroring a partial manual edit.
    tools_init = fixture_repo / "nullain-tools" / "src" / "nullain_tools" / "__init__.py"
    tools_init.write_text('__version__ = "0.0.9"\n', encoding="utf-8")

    with pytest.raises(SystemExit, match="out of sync"):
        mod._check_all_synced("0.1.0")


def test_compute_changes_bumps_all_seven_files(fixture_repo: Path) -> None:
    """Regression: a naive implementation returned separate (path, old, new)
    tuples per transform, so a file targeted by MULTIPLE transforms (like
    nullain-agentd/pyproject.toml, which needs both its own version line
    AND two internal pins updated) had all but the last transform silently
    discarded when writing — confirmed live: only the root pyproject.toml
    ended up bumped, everything else stayed on the old version."""
    mod = _load_module(fixture_repo)
    changes = mod._compute_changes("0.2.0")

    assert len(changes) == 7

    root_content = changes[fixture_repo / "pyproject.toml"]
    assert 'version = "0.2.0"' in root_content

    sdk_content = changes[fixture_repo / "nullain-sdk" / "pyproject.toml"]
    assert 'version = "0.2.0"' in sdk_content
    assert '"nullain-tools>=0.2.0"' in sdk_content

    sdk_init_content = changes[fixture_repo / "nullain-sdk" / "src" / "nullain" / "__init__.py"]
    assert '__version__ = "0.2.0"' in sdk_init_content

    # The critical case: nullain-agentd/pyproject.toml needs its OWN version
    # line updated AND both internal pins bumped — all three in one file.
    agentd_content = changes[fixture_repo / "nullain-agentd" / "pyproject.toml"]
    assert 'version = "0.2.0"' in agentd_content
    assert '"nullain-sdk>=0.2.0"' in agentd_content
    assert '"nullain-tools>=0.2.0"' in agentd_content
    # And the old version must not linger anywhere in that file.
    assert "0.1.0" not in agentd_content


def test_apply_writes_all_seven_files_to_disk(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    changes = mod._compute_changes("0.3.0")
    for path, new_content in changes.items():
        path.write_text(new_content, encoding="utf-8")

    for path in mod.VERSIONED_FILES:
        text = path.read_text(encoding="utf-8")
        assert "0.3.0" in text
        assert "0.1.0" not in text

    agentd_text = (fixture_repo / "nullain-agentd" / "pyproject.toml").read_text(encoding="utf-8")
    assert agentd_text.count("0.3.0") == 3  # own version + 2 pins


def test_compute_changes_rejects_missing_pin(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    sdk_pyproject = fixture_repo / "nullain-sdk" / "pyproject.toml"
    text = sdk_pyproject.read_text(encoding="utf-8")
    sdk_pyproject.write_text(text.replace('"nullain-tools>=0.1.0",\n', ""), encoding="utf-8")

    with pytest.raises(SystemExit, match="no pin for"):
        mod._compute_changes("0.2.0")


def test_semver_regex_accepts_valid_and_rejects_invalid() -> None:
    spec = importlib.util.spec_from_file_location("bump_version_semver_check", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for valid in ("0.1.0", "1.0.0", "10.20.30", "1.0.0rc1", "1.0.0a1", "1.0.0b2"):
        assert mod._SEMVER_RE.match(valid), valid

    for invalid in ("0.1", "v0.1.0", "0.1.0.1", "latest", ""):
        assert not mod._SEMVER_RE.match(invalid), invalid


def test_main_rejects_invalid_version_string(
    fixture_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_module(fixture_repo)
    sys.argv = ["bump_version.py", "not-a-version"]
    with pytest.raises(SystemExit):
        mod.main()


def test_main_rejects_same_as_current_version(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    sys.argv = ["bump_version.py", "0.1.0"]
    with pytest.raises(SystemExit, match="already the current version"):
        mod.main()


def test_main_dry_run_does_not_write_files(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    sys.argv = ["bump_version.py", "0.2.0"]
    mod.main()

    text = (fixture_repo / "pyproject.toml").read_text(encoding="utf-8")
    assert '"0.1.0"' in text  # unchanged — no --apply flag


def test_main_apply_writes_all_files(fixture_repo: Path) -> None:
    mod = _load_module(fixture_repo)
    sys.argv = ["bump_version.py", "0.2.0", "--apply"]
    mod.main()

    for path in mod.VERSIONED_FILES:
        assert "0.2.0" in path.read_text(encoding="utf-8")
