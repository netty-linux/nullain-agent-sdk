"""Unit test sentinel for Nullain SDK packages."""

import importlib.metadata

import nullain
import nullain_agentd
import nullain_tools
from nullain.errors import NullainError, ProviderError, ToolError
from nullain.telemetry import get_logger


def test_package_versions() -> None:
    """`__version__` is derived from installed package metadata (see each
    package's `__init__.py`), not hand-copied — a hardcoded string here
    would just be a second place to forget to update on every release
    (confirmed: this test carried "0.1.0" for two SDK releases past that
    being true)."""
    assert nullain.__version__ == importlib.metadata.version("nullain-sdk")
    assert nullain_tools.__version__ == importlib.metadata.version("nullain-tools")
    assert nullain_agentd.__version__ == importlib.metadata.version("nullain-agentd")


def test_exception_hierarchy() -> None:
    err = ProviderError("LLM failed", details={"code": 500})
    assert isinstance(err, NullainError)
    assert str(err) == "LLM failed (details: {'code': 500})"

    tool_err = ToolError("Execution failed")
    assert isinstance(tool_err, NullainError)
    assert str(tool_err) == "Execution failed"


def test_telemetry_logger() -> None:
    logger = get_logger("test")
    assert logger is not None
