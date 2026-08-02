"""Unit test sentinel for Nullain SDK packages."""

import nullain
import nullain_agentd
import nullain_tools
from nullain.errors import NullainError, ProviderError, ToolError
from nullain.telemetry import get_logger


def test_package_versions() -> None:
    assert nullain.__version__ == "0.1.0"
    assert nullain_tools.__version__ == "0.1.0"
    assert nullain_agentd.__version__ == "0.1.0"


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
