"""M9 — Example smoke test.

Imports every runnable example and asserts it exposes a callable ``main``. The
examples build their collaborators inside ``main`` (or at module level only for
pure definitions like the workflow), so importing them is offline and safe.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE_MODULES = [
    "00_llm_smoke",
    "01_basic_agent",
    "02_custom_tool",
    "03_subagent_authority",
    "04_workflow",
    "05_mcp_server",
    "06_openai_compat_smoke",
]


@pytest.mark.parametrize("module_name", EXAMPLE_MODULES)
def test_example_imports_and_has_main(module_name: str) -> None:
    """Each example imports cleanly and exposes a callable ``main``."""
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLES_DIR / f"{module_name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(getattr(module, "main", None)), f"{module_name} lacks a main()"
