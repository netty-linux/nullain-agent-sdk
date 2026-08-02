"""Nullain Tools — File system operations (read_file, write_file, edit_file, grep)."""

import contextlib
import re
from pathlib import Path

from nullain.tools import RegisteredTool, resolve_and_validate_path, tool


def create_filesystem_tools(workspace_root: str | Path) -> list[RegisteredTool]:
    root = Path(workspace_root).resolve()

    @tool(name="read_file", description="Read complete text file content within the workspace.")
    def read_file(path: str) -> str:
        target = resolve_and_validate_path(root, path)
        if not target.exists():
            return f"Error: File '{path}' does not exist."
        if not target.is_file():
            return f"Error: Path '{path}' is not a regular file."
        return target.read_text(encoding="utf-8", errors="replace")

    @tool(name="write_file", description="Write text content to a file in the workspace.")
    def write_file(path: str, content: str) -> str:
        target = resolve_and_validate_path(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{path}'."

    @tool(
        name="edit_file",
        description="Perform exact string replacement (old_str with new_str) in a workspace file.",
    )
    def edit_file(path: str, old_str: str, new_str: str) -> str:
        target = resolve_and_validate_path(root, path)
        if not target.exists():
            return f"Error: File '{path}' does not exist."
        content = target.read_text(encoding="utf-8")
        if old_str not in content:
            return f"Error: Target string 'old_str' not found in '{path}'."
        updated = content.replace(old_str, new_str, 1)
        target.write_text(updated, encoding="utf-8")
        return f"Successfully replaced target string in '{path}'."

    @tool(name="grep", description="Search for regex pattern across files in the workspace.")
    def grep(pattern: str, relative_dir: str = ".") -> str:
        search_dir = resolve_and_validate_path(root, relative_dir)
        if not search_dir.is_dir():
            return f"Error: Path '{relative_dir}' is not a directory."

        try:
            regex = re.compile(pattern)
        except re.error as err:
            return f"Error: Invalid regex pattern '{pattern}': {err}"

        matches: list[str] = []
        ignored_dirs = {
            ".git",
            ".venv",
            "node_modules",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            ".hypothesis",
        }
        for file_path in search_dir.rglob("*"):
            if any(part in ignored_dirs for part in file_path.parts):
                continue
            if file_path.is_file() and not file_path.name.startswith("."):
                with contextlib.suppress(Exception):
                    rel = file_path.relative_to(root)
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    for line_num, line in enumerate(content.splitlines(), start=1):
                        if regex.search(line):
                            matches.append(f"{rel}:{line_num}:{line}")
                            if len(matches) >= 50:
                                break
            if len(matches) >= 50:
                break

        if not matches:
            return f"No matches found for pattern '{pattern}'."
        return "\n".join(matches)

    return [read_file, write_file, edit_file, grep]


__all__ = ["create_filesystem_tools"]
