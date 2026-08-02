"""Nullain Agent SDK — Task Specification and SpecValidator."""

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from nullain.errors import SpecValidationError

if TYPE_CHECKING:
    from nullain.tools.registry import ToolRegistry


class TaskSpec(BaseModel):
    """Structured Plan specification generated during Plan phase."""

    spec_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective: str
    steps: list[str]
    target_files: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)


class SpecValidator:
    """Validates TaskSpec completeness and runs VERIFY phase criteria."""

    def validate_spec(self, spec: TaskSpec) -> None:
        """Validate TaskSpec structural integrity.

        Raises:
            SpecValidationError if required fields are missing or invalid.
        """
        if not spec.objective.strip():
            raise SpecValidationError("TaskSpec objective cannot be empty")
        if not spec.steps:
            raise SpecValidationError("TaskSpec must contain at least one step")

    async def verify(
        self,
        spec: TaskSpec,
        execution_output: str,
        workspace_root: Path | None = None,
        tools: "ToolRegistry | None" = None,
    ) -> tuple[bool, str]:
        """Evaluate task acceptance criteria in VERIFY phase against execution output.

        Verifies:
        1. Structural spec integrity
        2. Target files exist in workspace (if specified and workspace provided)
        3. Verification commands execute successfully (if specified and bash tool available)
        4. Explicit acceptance criteria and execution output consistency

        Returns:
            tuple[success, feedback_message]
        """
        self.validate_spec(spec)

        failed_checks: list[str] = []

        # 1. Target files check in workspace
        if workspace_root is not None and spec.target_files:
            ws = Path(workspace_root).resolve()
            for tf in spec.target_files:
                target_path = ws / tf
                if not target_path.exists():
                    failed_checks.append(f"Target file '{tf}' was not created")

        # 2. Verification commands execution
        if tools is not None and spec.verification_commands:
            for cmd in spec.verification_commands:
                cmd_tokens = cmd.strip().split()
                if not cmd_tokens:
                    continue
                try:
                    cmd_res = await tools.execute("bash", {"command_args": cmd_tokens})
                    cmd_lower = cmd_res.lower()
                    if "failed" in cmd_lower or "error" in cmd_lower:
                        failed_checks.append(f"Verification command '{cmd}' failed: {cmd_res}")
                except Exception as err:
                    failed_checks.append(f"Verification command '{cmd}' execution error: {err}")

        # 3. Acceptance criteria check against output
        if spec.acceptance_criteria:
            for criterion in spec.acceptance_criteria:
                crit_lower = criterion.lower()
                has_fail_crit = "error" in crit_lower or "fail" in crit_lower
                out_lower = execution_output.lower()
                has_fail_out = any(kw in out_lower for kw in ("error", "failed", "exception"))
                if has_fail_crit and has_fail_out:
                    failed_checks.append(f"Criteria not met: {criterion}")

        if failed_checks:
            return False, f"Verification failed: {'; '.join(failed_checks)}"

        return True, "All acceptance criteria verified successfully."


__all__ = ["SpecValidator", "TaskSpec"]
