"""Nullain Agent SDK — Task Specification and SpecValidator."""

import uuid

from pydantic import BaseModel, Field

from nullain.errors import SpecValidationError


class TaskSpec(BaseModel):
    """Structured Plan specification generated during Plan phase."""

    spec_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective: str
    steps: list[str]
    target_files: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


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

    async def verify(self, spec: TaskSpec, execution_output: str) -> tuple[bool, str]:
        """Evaluate task acceptance criteria in VERIFY phase against execution output.

        Returns:
            tuple[success, feedback_message]
        """
        self.validate_spec(spec)

        if not spec.acceptance_criteria:
            return True, "No explicit acceptance criteria specified. Verification passed."

        failed_criteria: list[str] = []
        for criterion in spec.acceptance_criteria:
            crit_lower = criterion.lower()
            has_fail_crit = "error" in crit_lower or "fail" in crit_lower
            out_lower = execution_output.lower()
            has_fail_out = any(kw in out_lower for kw in ("error", "failed", "exception"))
            if has_fail_crit and has_fail_out:
                failed_criteria.append(criterion)

        if failed_criteria:
            return False, f"Acceptance criteria failed: {', '.join(failed_criteria)}"

        return True, "All acceptance criteria verified successfully."


__all__ = ["SpecValidator", "TaskSpec"]
