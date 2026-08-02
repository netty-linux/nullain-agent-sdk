"""Pytest fixtures and environment configuration."""

import pytest


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path: pytest.TempPathFactory) -> None:
    """Ensure test environment variables and isolation."""
    pass
