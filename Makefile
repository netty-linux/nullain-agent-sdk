.PHONY: check test cov lint format typecheck schema audit build clean

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
