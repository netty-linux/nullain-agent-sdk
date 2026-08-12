# Releasing

Three packages (`nullain-sdk`, `nullain-tools`, `nullain-agentd`) version and
publish **independently**. One release is one package — never release all
three together, and the tooling in this repo does not let you.

## 1. Bump the version

```sh
make bump-version PACKAGE=nullain-sdk VERSION=0.7.2
```

Dry-run by default — prints the diff, writes nothing. Run the `-apply`
target instead to write it:

```sh
make bump-version-apply PACKAGE=nullain-sdk VERSION=0.7.2
```

This only touches `<package>/pyproject.toml`'s `version = "..."` line. It does
**not** touch:

- The root `pyproject.toml` — that's a monorepo marker, not a published
  version.
- Any `__init__.py` — `__version__` is derived from
  `importlib.metadata.version(...)` at import time, so there's nothing to
  bump there.
- Other packages' pins on this one (e.g. `nullain-tools/pyproject.toml`'s
  `nullain-sdk>=X.Y.Z`). If the dry-run prints a note that other packages pin
  the one you're bumping, decide: does the new version actually change what
  they require? If yes, re-run with `BUMP_DEPENDENTS=1` to tighten those pins
  in the same commit. If no, leave them — bumping a pin nobody needs bumped
  is just churn.

Move the bumped package's `[Unreleased]` entries in `CHANGELOG.md` to a new
dated section, e.g. `## [0.7.2] (nullain-sdk)` — leave other packages'
`[Unreleased]` entries alone, they move on their own release. Then
`uv sync --all-packages`, `make check`, commit, open a PR, merge.

## 2. Tag it

Tags are **prefixed by package** — this is what lets the release workflow
know which single package to publish, without guessing from a diff:

```sh
git tag nullain-sdk-v0.7.2
git push origin nullain-sdk-v0.7.2
```

Format: `<package>-v<version>`, e.g. `nullain-tools-v0.5.0`,
`nullain-agentd-v0.2.0`.

**Bare `vX.Y.Z` tags (no package prefix) are legacy** — used before
per-package tagging existed, when every tag published all three packages at
once. Do not create a new one. If you find yourself about to run
`git tag v0.8.0`, stop — that's the old format, and pushing it won't match
any of `release.yml`'s triggers, so nothing will happen except a confusing
tag that looks like a release but isn't one.

## 3. What happens in Actions

Pushing `nullain-sdk-v0.7.2` runs `.github/workflows/release.yml`:

1. **`verify`** — lint, typecheck, full test suite with coverage gate.
2. **`build`** — builds all three packages' distributions (cheap, always all
   three regardless of which one is being released) and checks their
   metadata with `twine check`.
3. **`resolve-target`** — reads the tag (or, for a manual dispatch, the
   `package` input) to determine which package this run is for, then checks
   whether that exact version already exists on the target index (PyPI or
   TestPyPI). **This runs before any environment's approval gate** — a
   duplicate version fails here, not after you've already approved a
   deployment that was going to fail anyway.
4. **`publish-pypi-<package>`** (or `publish-testpypi-<package>` for a manual
   TestPyPI dispatch) — only the job matching the resolved package has a true
   `if:`. **The other two packages' publish jobs are not part of this run at
   all** — GitHub Actions marks a job with a false `if:` as `skipped`, never
   `waiting`. Nothing sits at a reviewer-approval gate for a package this run
   was never going to touch.

The job that does run pauses for manual approval on its environment
(`pypi-nullain-sdk`, `pypi-nullain-tools`, or `pypi-nullain-agentd` — each
package has its own, since PyPI's trusted-publisher registration is keyed on
`(owner, repo, workflow, environment)`, not the PyPI project name). Approve
it in **Actions → the run → Review deployments → Approve**. That's the actual
point of no return — a version can never be re-uploaded to PyPI once
published.

## 4. Manual publish (`workflow_dispatch`)

For a TestPyPI dry-run, or to re-trigger a publish without pushing a new tag:
**Actions → Release → Run workflow**, choose the `package` and `target`
(`testpypi` or `pypi`). Same `resolve-target` gate applies — it still checks
the version isn't already published before anything can reach an approval
gate.

## Why per-package tags

Before this, one `vX.Y.Z` tag or dispatch published all three packages in
parallel jobs. The three packages already version independently in practice
(confirmed by the repo's actual history — `nullain-sdk` and `nullain-tools`
have shipped different version numbers for a long time), so a release that
bumps only one package left the other two's publish jobs trying to
re-publish a version that was already live. PyPI rejects that outright, but
worse, the jobs sat in `waiting` for reviewer approval on packages nobody
intended to release that day — the only way out was cancelling the whole run
by hand. This happened three times before per-package tags replaced the
single shared tag. See issues #80 and #81 for the full history.
