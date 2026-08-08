.PHONY: check test cov lint format typecheck schema audit build clean bump-version bump-version-apply

check: lint typecheck test

test:
	uv run pytest

cov:
	uv run pytest --cov --cov-report=term --cov-report=html

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run pyright

schema:
	uv run python -c "from nullain.protocol.exporter import export_schema; export_schema()"

audit:
	uv run pip-audit

build:
	rm -rf dist
	uv build --package nullain-sdk --out-dir dist
	uv build --package nullain-tools --out-dir dist
	uv build --package nullain-agentd --out-dir dist
	uvx twine check dist/*

clean:
	rm -rf .pytest_cache .ruff_cache .pyright_cache .coverage htmlcov coverage.xml build dist *.egg-info

# Preview a version bump across all 7 version-carrying files (4
# pyproject.toml + 3 __init__.py) plus the 4 internal dependency pins.
# Dry-run only — no files written. Usage: make bump-version VERSION=0.2.0
bump-version:
	@test -n "$(VERSION)" || (echo "usage: make bump-version VERSION=0.2.0" && exit 1)
	uv run python scripts/bump_version.py $(VERSION)

# Same as bump-version but actually writes the files. Usage:
# make bump-version-apply VERSION=0.2.0
bump-version-apply:
	@test -n "$(VERSION)" || (echo "usage: make bump-version-apply VERSION=0.2.0" && exit 1)
	uv run python scripts/bump_version.py $(VERSION) --apply
