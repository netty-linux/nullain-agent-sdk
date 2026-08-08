.PHONY: check test cov lint format typecheck schema audit build clean bump-version bump-version-apply evals evals-test evals-live

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

# Run the eval suite's own unit tests (grader correctness, report schema
# stability, replay determinism) — separate from the SDK's own coverage-gated
# `make test`; evals/ is not part of the published packages.
evals-test:
	PYTHONPATH=evals uv run pytest evals/tests -v

# Run every eval task against its recorded fixture (evals/fixtures/*.json).
# No network access, fully deterministic — safe for CI. Prints a summary and
# writes evals/report.json; diff it by hand against evals/baselines/ to see
# whether a harness change moved the pass rate.
evals:
	PYTHONPATH=evals uv run python -m nullain_evals.cli offline

# Run the eval suite against a real provider. Requires OLLAMA_API_KEY (or
# NULLAIN_OLLAMA_API_KEY) to be set. Usage:
#   make evals-live MODEL=glm-5.2:cloud
#   make evals-live MODEL=glm-5.2:cloud SAVE_FIXTURES=1   (record new fixtures for every passing task)
evals-live:
	@test -n "$(MODEL)" || (echo "usage: make evals-live MODEL=glm-5.2:cloud" && exit 1)
	PYTHONPATH=evals uv run python -m nullain_evals.cli live --provider ollama --model $(MODEL) $(if $(SAVE_FIXTURES),--save-fixtures,)
