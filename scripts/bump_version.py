"""Nullain Agent SDK — synchronized version bump across the monorepo.

Seven files carry the version number in lockstep:
  - 4 pyproject.toml `version = "..."` lines (root, nullain-sdk,
    nullain-tools, nullain-agentd) — the packaging-level version.
  - 3 `__version__ = "..."` lines (nullain/__init__.py,
    nullain_tools/__init__.py, nullain_agentd/__init__.py) — what
    `nullain --version` and `pip show` actually report to a user; found
    live to be a real gap, since bumping only pyproject.toml leaves
    `nullain --version` silently reporting the old number.
Plus 3 internal dependency pins (nullain-sdk>=X.Y.Z in nullain-tools/
nullain-agentd, nullain-tools>=X.Y.Z in nullain-sdk/nullain-agentd) that
must track the version too — bumping only the `version =` line and
forgetting the pins was a real bug fixed once already (PR #35).

Deliberately NOT bumped: the LSP/MCP client protocol identifiers
(lsp/client.py's clientInfo, mcp/client.py's client_version default) —
those identify this SDK as a protocol client to external servers during a
handshake, not a version a user of this package ever sees, so keeping
them separate from packaging version churn is the right call. Also NOT
touched automatically: tests/unit/test_sentinel.py::test_package_versions
hardcodes the version on purpose (it exists to catch an *accidental*
bump); the script prints a reminder to update it by hand since editing
test assertions automatically would defeat the point of a sentinel.

Usage:
    uv run python scripts/bump_version.py 0.2.0          # preview only
    uv run python scripts/bump_version.py 0.2.0 --apply   # write the files
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Every pyproject.toml carrying the monorepo's version number.
PYPROJECT_FILES = [
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "nullain-sdk" / "pyproject.toml",
    REPO_ROOT / "nullain-tools" / "pyproject.toml",
    REPO_ROOT / "nullain-agentd" / "pyproject.toml",
]

#: Every __init__.py carrying a __version__ string — what `nullain
#: --version` / `pip show` actually report, not just packaging metadata.
DUNDER_VERSION_FILES = [
    REPO_ROOT / "nullain-sdk" / "src" / "nullain" / "__init__.py",
    REPO_ROOT / "nullain-tools" / "src" / "nullain_tools" / "__init__.py",
    REPO_ROOT / "nullain-agentd" / "src" / "nullain_agentd" / "__init__.py",
]

#: All files whose primary version marker gets bumped (pyproject's
#: `version = "..."` or a module's `__version__ = "..."`).
VERSIONED_FILES = PYPROJECT_FILES + DUNDER_VERSION_FILES

#: Internal dependency pins that must move in lockstep with the version
#: they reference — (file, package name pinned inside it). nullain-agentd
#: has TWO pins in the same file (nullain-sdk and nullain-tools), which is
#: exactly the case that broke a naive "list of (path, old, new) tuples"
#: approach: each transform must be applied to the FILE'S ACCUMULATED new
#: content, not re-derived from the original on-disk text, or the second
#: transform for the same file silently discards the first.
INTERNAL_PINS = [
    (REPO_ROOT / "nullain-sdk" / "pyproject.toml", "nullain-tools"),
    (REPO_ROOT / "nullain-tools" / "pyproject.toml", "nullain-sdk"),
    (REPO_ROOT / "nullain-agentd" / "pyproject.toml", "nullain-sdk"),
    (REPO_ROOT / "nullain-agentd" / "pyproject.toml", "nullain-tools"),
]

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+([ab]\d+|rc\d+)?$")

#: Each versioned file's marker uses a different literal spelling
#: (pyproject's `version = "..."` vs. a module's `__version__ = "..."`),
#: keyed here so the rest of the script can treat both uniformly.
_MARKER_RE: dict[Path, re.Pattern[str]] = {
    **{p: re.compile(r'^version = "[^"]*"$', re.MULTILINE) for p in PYPROJECT_FILES},
    **{p: re.compile(r'^__version__ = "[^"]*"$', re.MULTILINE) for p in DUNDER_VERSION_FILES},
}


def _marker_replacement(path: Path, new_version: str) -> str:
    return (
        f'version = "{new_version}"'
        if path in PYPROJECT_FILES
        else f'__version__ = "{new_version}"'
    )


def _current_version() -> str:
    """Read the version from the root pyproject.toml (assumed authoritative)."""
    path = PYPROJECT_FILES[0]
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]*)"$', text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"error: no version line found in {path}")
    return match.group(1)


def _check_all_synced(current: str) -> None:
    """Refuse to bump if any versioned file is already out of sync — that's
    a separate problem this script shouldn't paper over silently."""
    mismatched: list[str] = []
    for path in VERSIONED_FILES:
        text = path.read_text(encoding="utf-8")
        match = _MARKER_RE[path].search(text)
        found = match.group(0).split('"')[1] if match else None
        if found != current:
            mismatched.append(f"  {path.relative_to(REPO_ROOT)}: {found!r} (expected {current!r})")
    if mismatched:
        raise SystemExit(
            "error: versions are already out of sync — fix this by hand before bumping:\n"
            + "\n".join(mismatched)
        )


def _compute_changes(new_version: str) -> dict[Path, str]:
    """Compute the fully-updated content for every affected file.

    Every file starts from its on-disk content, and every applicable
    transform (version/dunder bump, each internal pin bump) is applied on
    top of the RUNNING result for that file — critical for nullain-agentd/
    pyproject.toml, which needs both its own version line updated AND two
    separate internal pins bumped, all in the same file.
    """
    content: dict[Path, str] = {}

    def get(path: Path) -> str:
        if path not in content:
            content[path] = path.read_text(encoding="utf-8")
        return content[path]

    for path in VERSIONED_FILES:
        old = get(path)
        new = _MARKER_RE[path].sub(_marker_replacement(path, new_version), old, count=1)
        if new == old:
            raise SystemExit(f"error: failed to find/replace version marker in {path}")
        content[path] = new

    for path, pkg in INTERNAL_PINS:
        old = get(path)
        pattern = re.compile(rf'"{re.escape(pkg)}>=[^"]*"')
        new, count = pattern.subn(f'"{pkg}>={new_version}"', old)
        if count == 0:
            raise SystemExit(f"error: no pin for {pkg!r} found in {path}")
        content[path] = new

    return content


def _print_diff(path: Path, old: str, new: str) -> None:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    rel = path.relative_to(REPO_ROOT)
    printed_header = False
    for old_line, new_line in zip(old_lines, new_lines, strict=True):
        if old_line != new_line:
            if not printed_header:
                print(f"  {rel}")
                printed_header = True
            print(f"    - {old_line}")
            print(f"    + {new_line}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synchronized version bump across the Nullain Agent SDK monorepo."
    )
    parser.add_argument("version", help="new version, e.g. 0.2.0")
    parser.add_argument(
        "--apply", action="store_true", help="write the changes (default: dry-run preview only)"
    )
    args = parser.parse_args()

    new_version: str = args.version
    if not _SEMVER_RE.match(new_version):
        raise SystemExit(
            f"error: {new_version!r} doesn't look like a version "
            "(expected e.g. 0.2.0, 1.0.0, 1.0.0rc1)"
        )

    current = _current_version()
    _check_all_synced(current)

    if new_version == current:
        raise SystemExit(f"error: {new_version!r} is already the current version")

    print(f"Bumping {current} -> {new_version}\n")

    new_content = _compute_changes(new_version)

    for path, new in new_content.items():
        old = path.read_text(encoding="utf-8")
        _print_diff(path, old, new)

    if not args.apply:
        print("\nDry run only — no files written. Re-run with --apply to write these changes.")
        return 0

    for path, new in new_content.items():
        path.write_text(new, encoding="utf-8")

    print(f"\nDone. {len(new_content)} file(s) updated.")
    print("\nNext steps:")
    print("  1. Update tests/unit/test_sentinel.py::test_package_versions — it")
    print(f'     hardcodes "{current}" on purpose (catches an accidental version')
    print(f'     bump); update the 3 assertions to "{new_version}" now that this one')
    print("     is intentional. `make test` will fail until you do.")
    print("  2. Move [Unreleased] entries in CHANGELOG.md to a new dated section.")
    print("  3. uv sync --all-packages   (refresh uv.lock)")
    print("  4. make check               (lint + typecheck + test, one more time)")
    print("  5. Commit, open a PR, merge, then:")
    print(f"     git tag v{new_version} && git push origin v{new_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
