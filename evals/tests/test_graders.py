"""Tests each task's grader directly against a hand-built workspace — no
Agent, no provider — proving every grader fails on a known-bad solution and
passes on a known-good one (issue #45's acceptance criteria)."""

from __future__ import annotations

from pathlib import Path

import pytest
from nullain_evals.task import EvalTask, GradeContext
from nullain_evals.tasks import ALL_TASKS

_TASKS_BY_ID = {t.task_id: t for t in ALL_TASKS}


def _ctx(workspace: Path, *, final_text: str = "", run_success: bool = True) -> GradeContext:
    return GradeContext(
        workspace=workspace, final_text=final_text, run_success=run_success, steps=1
    )


class TestSingleFileEditCreate:
    task = _TASKS_BY_ID["single_file_edit_create"]

    def test_good_solution_passes(self, tmp_path: Path) -> None:
        (tmp_path / "greet.py").write_text('def greet(name):\n    return f"Hi {name}"\n')
        assert self.task.grader(_ctx(tmp_path)).passed

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        assert not self.task.grader(_ctx(tmp_path)).passed

    def test_wrong_function_name_fails(self, tmp_path: Path) -> None:
        (tmp_path / "greet.py").write_text("def hello(name):\n    return name\n")
        assert not self.task.grader(_ctx(tmp_path)).passed


class TestSingleFileEditAppend:
    task = _TASKS_BY_ID["single_file_edit_append"]

    def test_good_solution_passes(self, tmp_path: Path) -> None:
        (tmp_path / "constants.py").write_text("PI = 3.14159\nE = 2.71828\n")
        assert self.task.grader(_ctx(tmp_path)).passed

    def test_removed_existing_constant_fails(self, tmp_path: Path) -> None:
        (tmp_path / "constants.py").write_text("E = 2.71828\n")
        assert not self.task.grader(_ctx(tmp_path)).passed

    def test_no_new_constant_fails(self, tmp_path: Path) -> None:
        (tmp_path / "constants.py").write_text("PI = 3.14159\n")
        assert not self.task.grader(_ctx(tmp_path)).passed


class TestRenameFunction:
    task = _TASKS_BY_ID["rename_function"]

    def test_good_solution_passes(self, tmp_path: Path) -> None:
        (tmp_path / "calc.py").write_text(
            "def compute_sum(items):\n    return sum(items)\n\n\n"
            "def unrelated_helper(x):\n    return x * 2\n"
        )
        assert self.task.grader(_ctx(tmp_path)).passed

    def test_old_name_still_present_fails(self, tmp_path: Path) -> None:
        (tmp_path / "calc.py").write_text(
            "def compute_total(items):\n    return sum(items)\n\n\n"
            "def unrelated_helper(x):\n    return x * 2\n"
        )
        assert not self.task.grader(_ctx(tmp_path)).passed

    def test_unrelated_code_modified_fails(self, tmp_path: Path) -> None:
        (tmp_path / "calc.py").write_text(
            "def compute_sum(items):\n    return sum(items)\n\n\n"
            "def unrelated_helper(x):\n    return x * 3\n"  # changed 2 -> 3
        )
        assert not self.task.grader(_ctx(tmp_path)).passed


class TestBugfixFailingTest:
    task = _TASKS_BY_ID["bugfix_failing_test"]

    def test_good_solution_passes(self, tmp_path: Path) -> None:
        (tmp_path / "is_even.py").write_text("def is_even(n):\n    return n % 2 == 0\n")
        (tmp_path / "test_is_even.py").write_text(
            "from is_even import is_even\n\n\n"
            "def test_true():\n    assert is_even(4) is True\n\n\n"
            "def test_false():\n    assert is_even(3) is False\n"
        )
        assert self.task.grader(_ctx(tmp_path)).passed

    def test_still_buggy_fails(self, tmp_path: Path) -> None:
        (tmp_path / "is_even.py").write_text("def is_even(n):\n    return n % 2 == 1\n")
        (tmp_path / "test_is_even.py").write_text(
            "from is_even import is_even\n\n\n"
            "def test_true():\n    assert is_even(4) is True\n\n\n"
            "def test_false():\n    assert is_even(3) is False\n"
        )
        assert not self.task.grader(_ctx(tmp_path)).passed


class TestMultiFileFeature:
    task = _TASKS_BY_ID["multi_file_feature_add_field"]

    def test_good_solution_passes(self, tmp_path: Path) -> None:
        (tmp_path / "models.py").write_text(
            "class User:\n"
            "    def __init__(self, name, email, age):\n"
            "        self.name = name\n"
            "        self.email = email\n"
            "        self.age = age\n"
        )
        (tmp_path / "main.py").write_text('user = User("Ada", "ada@example.com", age=34)\n')
        assert self.task.grader(_ctx(tmp_path)).passed

    def test_only_models_updated_fails(self, tmp_path: Path) -> None:
        (tmp_path / "models.py").write_text(
            "class User:\n"
            "    def __init__(self, name, email, age):\n"
            "        self.name = name\n"
            "        self.email = email\n"
            "        self.age = age\n"
        )
        (tmp_path / "main.py").write_text('user = User("Ada", "ada@example.com")\n')
        assert not self.task.grader(_ctx(tmp_path)).passed

    def test_removed_existing_fields_fails(self, tmp_path: Path) -> None:
        (tmp_path / "models.py").write_text(
            "class User:\n    def __init__(self, age):\n        self.age = age\n"
        )
        (tmp_path / "main.py").write_text("user = User(age=34)\n")
        assert not self.task.grader(_ctx(tmp_path)).passed


class TestExploreAndExplain:
    task = _TASKS_BY_ID["explore_and_explain_fib"]

    def test_good_answer_passes(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path, final_text="This function uses memoization via a cache dict.")
        assert self.task.grader(ctx).passed

    def test_answer_missing_key_concept_fails(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path, final_text="This function computes Fibonacci numbers recursively.")
        assert not self.task.grader(ctx).passed


class TestAddErrorHandling:
    task = _TASKS_BY_ID["add_error_handling"]

    def test_good_solution_passes(self, tmp_path: Path) -> None:
        (tmp_path / "parser_mod.py").write_text(
            "def parse_int(text):\n"
            "    try:\n"
            "        return int(text)\n"
            "    except ValueError:\n"
            "        return None\n"
        )
        (tmp_path / "test_parser_mod.py").write_text(
            "from parser_mod import parse_int\n\n\n"
            "def test_valid():\n    assert parse_int('42') == 42\n\n\n"
            "def test_invalid():\n    assert parse_int('nope') is None\n"
        )
        assert self.task.grader(_ctx(tmp_path)).passed

    def test_still_raises_fails(self, tmp_path: Path) -> None:
        (tmp_path / "parser_mod.py").write_text("def parse_int(text):\n    return int(text)\n")
        (tmp_path / "test_parser_mod.py").write_text(
            "from parser_mod import parse_int\n\n\n"
            "def test_valid():\n    assert parse_int('42') == 42\n\n\n"
            "def test_invalid():\n    assert parse_int('nope') is None\n"
        )
        assert not self.task.grader(_ctx(tmp_path)).passed


class TestRefactorExtractFunction:
    task = _TASKS_BY_ID["refactor_extract_function"]

    def test_good_solution_passes(self, tmp_path: Path) -> None:
        (tmp_path / "pricing.py").write_text(
            "def _sum_items(items):\n"
            "    total = 0\n"
            "    for item in items:\n"
            '        total += item["price"] * item["quantity"]\n'
            "    return total\n\n\n"
            "def total_price_usd(items):\n"
            "    return round(_sum_items(items), 2)\n\n\n"
            "def total_price_eur(items):\n"
            "    return round(_sum_items(items) * 0.92, 2)\n"
        )
        (tmp_path / "test_pricing.py").write_text(
            "from pricing import total_price_usd, total_price_eur\n\n"
            'ITEMS = [{"price": 10.0, "quantity": 2}, {"price": 5.5, "quantity": 1}]\n\n\n'
            "def test_usd():\n    assert total_price_usd(ITEMS) == 25.5\n\n\n"
            "def test_eur():\n    assert total_price_eur(ITEMS) == round(25.5 * 0.92, 2)\n"
        )
        assert self.task.grader(_ctx(tmp_path)).passed

    def test_still_duplicated_fails(self, tmp_path: Path) -> None:
        (tmp_path / "pricing.py").write_text(
            "def total_price_usd(items):\n"
            "    total = 0\n"
            "    for item in items:\n"
            '        total += item["price"] * item["quantity"]\n'
            "    return round(total, 2)\n\n\n"
            "def total_price_eur(items):\n"
            "    total = 0\n"
            "    for item in items:\n"
            '        total += item["price"] * item["quantity"]\n'
            "    return round(total * 0.92, 2)\n"
        )
        (tmp_path / "test_pricing.py").write_text(
            "from pricing import total_price_usd, total_price_eur\n\n"
            'ITEMS = [{"price": 10.0, "quantity": 2}, {"price": 5.5, "quantity": 1}]\n\n\n'
            "def test_usd():\n    assert total_price_usd(ITEMS) == 25.5\n\n\n"
            "def test_eur():\n    assert total_price_eur(ITEMS) == round(25.5 * 0.92, 2)\n"
        )
        assert not self.task.grader(_ctx(tmp_path)).passed


def test_every_task_has_a_test_class_above() -> None:
    """Guards against a new task being added to the registry without a
    corresponding known-good/known-bad test pair here."""
    tested_ids = {
        "single_file_edit_create",
        "single_file_edit_append",
        "rename_function",
        "bugfix_failing_test",
        "multi_file_feature_add_field",
        "explore_and_explain_fib",
        "add_error_handling",
        "refactor_extract_function",
    }
    all_ids = {t.task_id for t in ALL_TASKS}
    missing = all_ids - tested_ids
    assert not missing, f"tasks missing grader tests: {missing}"


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.task_id)
def test_setup_does_not_raise(task: EvalTask, tmp_path: Path) -> None:
    task.setup(tmp_path)
