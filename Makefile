.PHONY: check test lint format typecheck schema audit clean

check: lint typecheck test

test:
	uv run pytest

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

clean:
	rm -rf .pytest_cache .ruff_cache .pyright_cache build dist *.egg-info
