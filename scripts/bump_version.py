"""Nullain Agent SDK — per-package version bump.

Each of the three publishable packages (`nullain-sdk`, `nullain-tools`,
`nullain-agentd`) versions independently — confirmed by the actual state of
the repo (sdk 0.7.1, tools 0.4.0, agentd 0.1.0, all published separately)
long before this script caught up to it. This bumps exactly one package's
`pyproject.toml` `version = "..."` line. Nothing else needs bumping:

- The root `pyproject.toml`'s version is a monorepo marker, not a published
  package version — never touched by this script.
- The 3 packages' `__init__.py` files derive `__version__` from
  `importlib.metadata.version(...)` at import time (dynamic), not a literal
  `__version__ = "..."` string — there is nothing to bump there anymore.
  (An earlier version of this script bumped a literal marker in those files;
  it was removed once the packages switched to metadata-derived versions.)

Internal dependency pins (`nullain-sdk>=X.Y.Z` inside nullain-tools/
nullain-agentd's pyproject.toml, `nullain-tools>=X.Y.Z` inside nullain-sdk/
nullain-agentd's) are NOT bumped automatically — apart from the package
being bumped, tightening another package's pin on it is a compatibility
decision (does the new version actually require the floor to move?), not a
mechanical part of versioning. Opt in explicitly with --bump-dependents.
Omitting it silently is exactly how the pins went stale in the first place
(nullain-tools/nullain-agentd still pin nullain-sdk>=0.1.0 while nullain-sdk
is at 0.7.1) — so the dry-run always prints a warning when dependent pins
exist and --bump-dependents was not passed, naming which packages/files and
the exact flag to fix it, rather than leaving that silently forgotten.

Usage:
    uv run python scripts/bump_version.py nullain-sdk 0.7.2                # preview only
    uv run python scripts/bump_version.py nullain-sdk 0.7.2 --apply        # write the bump
    uv run python scripts/bump_version.py nullain-sdk 0.7.2 --apply \\
        --bump-dependents  # also tighten dependents' pins
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Every publishable package this script knows how to bump, keyed by its
#: PyPI/pyproject `name`. Each maps to its `pyproject.toml`.
PACKAGES: dict[str, Path] = {
    "nullain-sdk": REPO_ROOT / "nullain-sdk" / "pyproject.toml",
    "nullain-tools": REPO_ROOT / "nullain-tools" / "pyproject.toml",
    "nullain-agentd": REPO_ROOT / "nullain-agentd" / "pyproject.toml",
}

#: Internal dependency pins that reference another package's version —
#: (file, package name pinned inside it). nullain-agentd has TWO pins in
#: the same file (nullain-sdk and nullain-tools); both are tracked
#: separately so bumping nullain-sdk doesn't touch the nullain-tools pin
#: and vice versa.
INTERNAL_PINS: list[tuple[Path, str]] = [
    (REPO_ROOT / "nullain-sdk" / "pyproject.toml", "nullain-tools"),
    (REPO_ROOT / "nullain-tools" / "pyproject.toml", "nullain-sdk"),
    (REPO_ROOT / "nullain-agentd" / "pyproject.toml", "nullain-sdk"),
    (REPO_ROOT / "nullain-agentd" / "pyproject.toml", "nullain-tools"),
]

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+([ab]\d+|rc\d+)?$")
_VERSION_MARKER_RE = re.compile(r'^version = "[^"]*"$', re.MULTILINE)
_PIN_RE = re.compile(r'"{pkg}>=([^",<]*)"')


def _current_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = _VERSION_MARKER_RE.search(text)
    if match is None:
        raise SystemExit(f"error: no version line found in {path}")
    return match.group(0).split('"')[1]


def _pins_referencing(package: str) -> list[tuple[Path, str]]:
    """Every (file, current_pin_version) that pins `package>=...`, excluding
    that package's own pyproject.toml (a package never pins itself)."""
    own_pyproject = PACKAGES[package]
    pattern = re.compile(_PIN_RE.pattern.format(pkg=re.escape(package)))
    found: list[tuple[Path, str]] = []
    for path, pinned_pkg in INTERNAL_PINS:
        if pinned_pkg != package or path == own_pyproject:
            continue
        text = path.read_text(encoding="utf-8")
        match = pattern.search(text)
        if match:
            found.append((path, match.group(1)))
    return found


def _compute_changes(package: str, new_version: str, *, bump_dependents: bool) -> dict[Path, str]:
    """Compute the fully-updated content for every affected file.

    Every file starts from its on-disk content, and every applicable
    transform is applied on top of the RUNNING result for that file —
    matters for nullain-agentd/pyproject.toml, which can carry pins for
    both nullain-sdk and nullain-tools in the same file.
    """
    content: dict[Path, str] = {}

    def get(path: Path) -> str:
        if path not in content:
            content[path] = path.read_text(encoding="utf-8")
        return content[path]

    own_path = PACKAGES[package]
    old = get(own_path)
    new = _VERSION_MARKER_RE.sub(f'version = "{new_version}"', old, count=1)
    if new == old:
        raise SystemExit(f"error: failed to find/replace version marker in {own_path}")
    content[own_path] = new

    if bump_dependents:
        pattern = re.compile(_PIN_RE.pattern.format(pkg=re.escape(package)))
        for path, _pinned_version in _pins_referencing(package):
            old = get(path)
            new, count = pattern.subn(f'"{package}>={new_version}"', old)
            if count == 0:
                raise SystemExit(f"error: no pin for {package!r} found in {path}")
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


def _warn_stale_dependent_pins(package: str, new_version: str) -> None:
    """--bump-dependents is opt-in on purpose (tightening another package's
    floor is a compatibility call, not a mechanical one) — but opt-in with
    no reminder is how the pins went stale in the first place (nullain-tools/
    nullain-agentd still pin nullain-sdk>=0.1.0 while nullain-sdk shipped
    0.2.0 through 0.7.1 in between). Always surface which pins exist and
    the exact command to tighten them, even when the caller didn't ask."""
    pins = _pins_referencing(package)
    if not pins:
        return
    print(f"\nNote: other packages pin {package}:")
    for path, current_pin in pins:
        rel = path.relative_to(REPO_ROOT)
        print(f'  {rel}: "{package}>={current_pin}"')
    print(
        f"These are NOT updated by this run. If {new_version} changes what "
        f"{package}'s dependents require (a new function they need, a "
        "behavior they rely on), re-run with --bump-dependents to tighten "
        "the floor to the new version. If nothing downstream depends on "
        f"what changed in {new_version}, leaving the pins as-is is correct "
        "— don't tighten just because a bump happened."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bump one package's version in the Nullain Agent SDK monorepo."
    )
    parser.add_argument("package", choices=sorted(PACKAGES), help="which package to bump")
    parser.add_argument("version", help="new version, e.g. 0.2.0")
    parser.add_argument(
        "--apply", action="store_true", help="write the changes (default: dry-run preview only)"
    )
    parser.add_argument(
        "--bump-dependents",
        action="store_true",
        help=(
            "also tighten other packages' internal `>=` pin on this package to "
            "the new version (opt-in: see the module docstring for why this "
            "isn't automatic)"
        ),
    )
    args = parser.parse_args()

    package: str = args.package
    new_version: str = args.version
    if not _SEMVER_RE.match(new_version):
        raise SystemExit(
            f"error: {new_version!r} doesn't look like a version "
            "(expected e.g. 0.2.0, 1.0.0, 1.0.0rc1)"
        )

    own_path = PACKAGES[package]
    current = _current_version(own_path)
    if new_version == current:
        raise SystemExit(f"error: {new_version!r} is already {package}'s current version")

    print(f"Bumping {package}: {current} -> {new_version}\n")

    new_content = _compute_changes(package, new_version, bump_dependents=args.bump_dependents)

    for path, new in new_content.items():
        old = path.read_text(encoding="utf-8")
        _print_diff(path, old, new)

    if not args.bump_dependents:
        _warn_stale_dependent_pins(package, new_version)

    if not args.apply:
        print("\nDry run only — no files written. Re-run with --apply to write these changes.")
        return 0

    for path, new in new_content.items():
        path.write_text(new, encoding="utf-8")

    tag = f"{package}-v{new_version}"
    print(f"\nDone. {len(new_content)} file(s) updated.")
    print("\nNext steps:")
    print(f"  1. Move {package}'s [Unreleased] entries in CHANGELOG.md to a new")
    print(f"     dated `## [{new_version}] ({package})` section — leave other")
    print("     packages' [Unreleased] entries alone; they bump separately.")
    print("  2. uv sync --all-packages   (refresh uv.lock)")
    print("  3. make check               (lint + typecheck + test, one more time)")
    print("  4. Commit, open a PR, merge, then:")
    print(f"     git tag {tag} && git push origin {tag}")
    print("     See docs/releasing.md for what happens next in Actions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
