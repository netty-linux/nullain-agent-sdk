"""Nullain Agent SDK — Task Specification and SpecValidator."""

import json as _json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from nullain.errors import SpecValidationError
from nullain.llm.provider import LLMProvider
from nullain.llm.types import ChatMessage, CompletionRequest
from nullain.telemetry import get_logger

if TYPE_CHECKING:
    from nullain.tools.registry import ToolRegistry

logger = get_logger(__name__)

# Output markers that indicate a failed execution. Used by the conservative
# heuristic verifier when no LLM judge is configured. Lowercased comparison.
_FAILURE_MARKERS: tuple[str, ...] = (
    "error:",
    "failed",
    "exception",
    "traceback",
    "command exit code:",
)

# The bash tool prefixes its output with this when the subprocess exits non-zero.
# Public so the Act loop can reuse the same contract for error detection.
BASH_NONZERO_PREFIX = "Command exit code:"


class TaskSpec(BaseModel):
    """Structured Plan specification generated during Plan phase."""

    spec_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective: str
    steps: list[str]
    target_files: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)


class SpecValidator:
    """Validates TaskSpec completeness and runs the VERIFY phase.

    Verification strategy, in increasing rigor:
    1. Structural spec integrity (objective + steps present).
    2. Target files exist on disk (if workspace_root + target_files given).
    3. verification_commands run via the bash tool; a command fails when its
       subprocess exits non-zero (detected via the "Command exit code:" prefix)
       or raises. We no longer substring-match "error"/"failed" in the output,
       which previously produced both false positives ("0 errors") and let
       benign criteria pass unconditionally.
    4. acceptance_criteria:
       - If a ``judge_provider`` is configured, each criterion is evaluated
         semantically by an LLM judge (the only real check of intent).
       - Otherwise a conservative heuristic fails the spec if the execution
         output contains any failure marker. This is stricter than the old
         keyword-coincidence logic, which let criteria without "error"/"fail"
         in their text pass regardless of output.
    """

    def __init__(
        self,
        judge_provider: LLMProvider | None = None,
        judge_model: str | None = None,
    ) -> None:
        self.judge_provider = judge_provider
        self.judge_model = judge_model

    def validate_spec(self, spec: TaskSpec) -> None:
        """Validate TaskSpec structural integrity.

        Raises:
            SpecValidationError if required fields are missing or invalid.
        """
        if not spec.objective.strip():
            raise SpecValidationError("TaskSpec objective cannot be empty")
        if not spec.steps:
            raise SpecValidationError("TaskSpec must contain at least one step")

    async def _judge_criteria(self, spec: TaskSpec, execution_output: str) -> list[str]:
        """Ask the LLM judge which acceptance criteria are met.

        Returns the list of unmet criterion descriptions. Falls back to the
        heuristic when the judge is unavailable or its response is unparseable.
        """
        if self.judge_provider is None:
            return self._heuristic_criteria(spec, execution_output)
        if not self.judge_model:
            logger.warning(
                "spec_judge_missing_model",
                detail="judge_provider set without judge_model; falling back to heuristic",
            )
            return self._heuristic_criteria(spec, execution_output)

        criteria_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(spec.acceptance_criteria))
        judge_prompt = (
            "You are a strict verification judge. Decide which acceptance criteria "
            "are met by the execution output below. Be skeptical: only mark a "
            "criterion as met if the output clearly demonstrates it.\n\n"
            f"Task objective: {spec.objective}\n\n"
            f"Execution output:\n{execution_output}\n\n"
            f"Acceptance criteria:\n{criteria_block}\n\n"
            "Respond ONLY with a JSON object of the form:\n"
            '{"results": [{"criterion": "...", "met": true|false, "reason": "..."}]}'
        )
        req = CompletionRequest(
            model=self.judge_model,
            messages=[
                ChatMessage(
                    role="system",
                    content="You are a verification judge that emits only JSON.",
                ),
                ChatMessage(role="user", content=judge_prompt),
            ],
            stream=False,
        )
        text = ""
        try:
            response = await self.judge_provider.generate(req)
            text = (response.delta_text or "").strip()
            # Tolerate markdown fences around the JSON.
            if "```" in text:
                for part in text.split("```"):
                    cleaned = part.strip()
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()
                    if cleaned.startswith("{"):
                        text = cleaned
                        break
            parsed: object = _json.loads(text)
            if not isinstance(parsed, dict):
                logger.warning("spec_judge_non_dict", raw_text=text[:200])
                return self._heuristic_criteria(spec, execution_output)
            data = cast(dict[str, Any], parsed)
            results_raw: object = data.get("results", [])
            if not isinstance(results_raw, list):
                logger.warning("spec_judge_empty_results", raw_text=text[:200])
                return self._heuristic_criteria(spec, execution_output)
            results = cast(list[Any], results_raw)
            unmet: list[str] = []
            evaluated = 0
            for item in results:
                if not isinstance(item, dict):
                    continue
                item_dict = cast(dict[str, Any], item)
                criterion = str(item_dict.get("criterion", "")).strip()
                met = bool(item_dict.get("met", False))
                if not met:
                    reason = str(item_dict.get("reason", "")).strip()
                    label = criterion or "unnamed criterion"
                    unmet.append(label + (f" ({reason})" if reason else ""))
                evaluated += 1
            # If the judge returned nothing usable, do not silently pass.
            if evaluated == 0:
                logger.warning("spec_judge_empty_results", raw_text=text[:200])
                return self._heuristic_criteria(spec, execution_output)
            return unmet
        except Exception as err:
            logger.warning("spec_judge_failed", error=str(err), raw_text=text[:200])
            return self._heuristic_criteria(spec, execution_output)

    @staticmethod
    def _heuristic_criteria(spec: TaskSpec, execution_output: str) -> list[str]:
        """Conservative fallback: fail if execution output contains failure markers."""
        out_lower = execution_output.lower()
        if any(marker in out_lower for marker in _FAILURE_MARKERS):
            return [
                "execution output contains failure markers; "
                "acceptance criteria cannot be confirmed (no LLM judge configured)"
            ]
        return []

    async def verify(
        self,
        spec: TaskSpec,
        execution_output: str,
        workspace_root: Path | None = None,
        tools: "ToolRegistry | None" = None,
    ) -> tuple[bool, str]:
        """Evaluate task acceptance criteria in VERIFY phase against execution output.

        Returns:
            tuple[success, feedback_message]
        """
        self.validate_spec(spec)

        failed_checks: list[str] = []

        # 1. Target files exist on disk
        if workspace_root is not None and spec.target_files:
            ws = Path(workspace_root).resolve()
            for tf in spec.target_files:
                target_path = ws / tf
                if not target_path.exists():
                    failed_checks.append(f"Target file '{tf}' was not created")

        # 2. Verification commands — fail on non-zero exit, not on substring matches
        if tools is not None and spec.verification_commands:
            for cmd in spec.verification_commands:
                cmd_tokens = cmd.strip().split()
                if not cmd_tokens:
                    continue
                try:
                    cmd_res = await tools.execute("bash", {"command_args": cmd_tokens})
                    if cmd_res.startswith(BASH_NONZERO_PREFIX):
                        failed_checks.append(f"Verification command '{cmd}' failed: {cmd_res}")
                except Exception as err:
                    failed_checks.append(f"Verification command '{cmd}' execution error: {err}")

        # 3. Acceptance criteria — LLM judge when available, else conservative heuristic
        if spec.acceptance_criteria:
            unmet = await self._judge_criteria(spec, execution_output)
            for item in unmet:
                failed_checks.append(f"Criteria not met: {item}")

        if failed_checks:
            return False, f"Verification failed: {'; '.join(failed_checks)}"

        return True, "All acceptance criteria verified successfully."


__all__ = ["BASH_NONZERO_PREFIX", "SpecValidator", "TaskSpec"]
