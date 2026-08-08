"""One-off script that hand-authors offline replay fixtures for every task in
the eval suite — used because no live Ollama Cloud API key was available
when this harness was built (see evals/README.md). Each fixture is the exact
CompletionChunk sequence a real model *should* produce for that task's happy
path: a Plan-phase emit_task_spec call (every task prompt here classifies as
MEDIUM complexity — no LOW-heuristic keyword), then one or more Act-phase
tool calls, ending with a final text-only response (no tool_calls) so the
ReAct loop terminates naturally.

Run: uv run python evals/scripts/author_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "nullain-sdk" / "src"))
sys.path.insert(0, str(REPO_ROOT / "evals"))

from nullain.llm.types import CompletionChunk, TokenUsage, ToolCall  # noqa: E402
from nullain_evals.replay import dump_responses  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "evals" / "fixtures"

_USAGE = TokenUsage(prompt_tokens=800, completion_tokens=120, total_tokens=920)


def _spec_call(objective: str, steps: list[str], target_files: list[str]) -> CompletionChunk:
    return CompletionChunk(
        tool_calls=[
            ToolCall(
                id="call_spec",
                name="emit_task_spec",
                arguments={
                    "objective": objective,
                    "steps": steps,
                    "target_files": target_files,
                    "acceptance_criteria": ["Task completes without errors"],
                },
            )
        ],
        usage=_USAGE,
    )


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> CompletionChunk:
    return CompletionChunk(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        usage=_USAGE,
    )


def _final(text: str) -> CompletionChunk:
    return CompletionChunk(delta_text=text, finish_reason="stop", usage=_USAGE)


def _write(fixture_name: str, responses: list[CompletionChunk]) -> None:
    dump_responses(responses, FIXTURES_DIR / f"{fixture_name}.json")
    print(f"wrote {fixture_name}.json ({len(responses)} responses)")


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    _write(
        "single_file_edit_create",
        [
            _spec_call(
                "Create greet.py with a greet(name) function",
                ["Write greet.py with a greet function returning a greeting string"],
                ["greet.py"],
            ),
            _tool_call(
                "call_1",
                "write_file",
                {
                    "path": "greet.py",
                    "content": 'def greet(name):\n    return f"Hello, {name}!"\n',
                },
            ),
            _final("Created greet.py with a greet(name) function that returns a greeting."),
        ],
    )

    _write(
        "single_file_edit_append",
        [
            _spec_call(
                "Add Euler's number constant E to constants.py",
                ["Read constants.py", "Append E = 2.71828 while keeping PI intact"],
                ["constants.py"],
            ),
            _tool_call("call_1", "read_file", {"path": "constants.py"}),
            _tool_call(
                "call_2",
                "write_file",
                {"path": "constants.py", "content": "PI = 3.14159\nE = 2.71828\n"},
            ),
            _final("Added E = 2.71828 to constants.py, keeping the existing PI constant."),
        ],
    )

    _write(
        "rename_function",
        [
            _spec_call(
                "Rename compute_total to compute_sum in calc.py",
                ["Read calc.py", "Rename compute_total to compute_sum, leave the rest untouched"],
                ["calc.py"],
            ),
            _tool_call("call_1", "read_file", {"path": "calc.py"}),
            _tool_call(
                "call_2",
                "edit_file",
                {
                    "path": "calc.py",
                    "old_str": "def compute_total(items):",
                    "new_str": "def compute_sum(items):",
                },
            ),
            _final("Renamed compute_total to compute_sum without touching unrelated_helper."),
        ],
    )

    _write(
        "bugfix_failing_test",
        [
            _spec_call(
                "Fix is_even.py so test_is_even.py passes",
                ["Read is_even.py", "Read test_is_even.py", "Fix the inverted condition"],
                ["is_even.py"],
            ),
            _tool_call("call_1", "read_file", {"path": "is_even.py"}),
            _tool_call("call_2", "read_file", {"path": "test_is_even.py"}),
            _tool_call(
                "call_3",
                "edit_file",
                {
                    "path": "is_even.py",
                    "old_str": "return n % 2 == 1  # bug: inverted condition",
                    "new_str": "return n % 2 == 0",
                },
            ),
            _final("Fixed the inverted condition in is_even.py; tests should now pass."),
        ],
    )

    _write(
        "multi_file_feature_add_field",
        [
            _spec_call(
                "Add an age field to User and update main.py's construction",
                [
                    "Read models.py",
                    "Add age to User.__init__",
                    "Read main.py",
                    "Update the User(...) call to pass age",
                ],
                ["models.py", "main.py"],
            ),
            _tool_call("call_1", "read_file", {"path": "models.py"}),
            _tool_call(
                "call_2",
                "write_file",
                {
                    "path": "models.py",
                    "content": (
                        "class User:\n"
                        "    def __init__(self, name, email, age):\n"
                        "        self.name = name\n"
                        "        self.email = email\n"
                        "        self.age = age\n"
                    ),
                },
            ),
            _tool_call("call_3", "read_file", {"path": "main.py"}),
            _tool_call(
                "call_4",
                "write_file",
                {
                    "path": "main.py",
                    "content": (
                        "from models import User\n\n"
                        'user = User("Ada", "ada@example.com", age=34)\n'
                        "print(user.name)\n"
                    ),
                },
            ),
            _final("Added an age field to User in models.py and updated main.py accordingly."),
        ],
    )

    _write(
        "explore_and_explain_fib",
        [
            _spec_call(
                "Explain fib.py's caching behavior",
                ["Read fib.py", "Explain the memoization pattern in the response"],
                [],
            ),
            _tool_call("call_1", "read_file", {"path": "fib.py"}),
            _final(
                "fib(n) computes the nth Fibonacci number recursively. It uses a "
                "mutable dict as a default argument (cache={}) to memoize "
                "previously computed results across calls, so repeated or "
                "overlapping calls avoid redundant recursive work — this is a "
                "caching/memoization optimization, though relying on a mutable "
                "default argument for persistent state is a somewhat unusual "
                "(and commonly flagged) Python idiom."
            ),
        ],
    )

    _write(
        "add_error_handling",
        [
            _spec_call(
                "Make parse_int return None instead of raising on invalid input",
                ["Read parser_mod.py", "Wrap int(text) in a try/except returning None on failure"],
                ["parser_mod.py"],
            ),
            _tool_call("call_1", "read_file", {"path": "parser_mod.py"}),
            _tool_call(
                "call_2",
                "write_file",
                {
                    "path": "parser_mod.py",
                    "content": (
                        "def parse_int(text):\n"
                        "    try:\n"
                        "        return int(text)\n"
                        "    except ValueError:\n"
                        "        return None\n"
                    ),
                },
            ),
            _final("parse_int now returns None on invalid input instead of raising."),
        ],
    )

    _write(
        "refactor_extract_function",
        [
            _spec_call(
                "Extract the duplicated summation loop in pricing.py into a helper",
                [
                    "Read pricing.py",
                    "Extract a _sum_items helper used by both total_price_usd and total_price_eur",
                ],
                ["pricing.py"],
            ),
            _tool_call("call_1", "read_file", {"path": "pricing.py"}),
            _tool_call(
                "call_2",
                "write_file",
                {
                    "path": "pricing.py",
                    "content": (
                        "def _sum_items(items):\n"
                        "    total = 0\n"
                        "    for item in items:\n"
                        '        total += item["price"] * item["quantity"]\n'
                        "    return total\n\n\n"
                        "def total_price_usd(items):\n"
                        "    return round(_sum_items(items), 2)\n\n\n"
                        "def total_price_eur(items):\n"
                        "    return round(_sum_items(items) * 0.92, 2)\n"
                    ),
                },
            ),
            _final(
                "Extracted the shared summation loop into _sum_items; both "
                "total_price_usd and total_price_eur now delegate to it."
            ),
        ],
    )


if __name__ == "__main__":
    main()
