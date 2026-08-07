"""Nullain Agent SDK — AgentLoop ReAct and Plan/Act Orchestrated Execution Engine."""

import asyncio
import contextlib
import json as _json
import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from anyio import get_cancelled_exc_class
from pydantic import ValidationError

from nullain.agent.checkpoint import RewindPoint, list_rewind_points, rewind_events
from nullain.agent.result import RunResult, RunStatus
from nullain.agent.spec import SpecValidator, TaskSpec
from nullain.authority import Authority, Capability
from nullain.context.assembler import PromptAssembler
from nullain.context.manager import ContextManager
from nullain.errors import (
    BudgetExceededError,
    ContextWindowExhaustedError,
    NullainError,
    ProviderError,
    ToolError,
    ToolPermissionError,
)
from nullain.events import (
    BaseEvent,
    ConversationState,
    ErrorEvent,
    EventBus,
    EventStore,
    IncrementalFold,
    ModelResponseEvent,
    SpecCreatedEvent,
    SpecVerifiedEvent,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from nullain.hooks import HookLifecycle, HookManager, HooksConfig
from nullain.llm import (
    ChatMessage,
    CompletionChunk,
    CompletionRequest,
    FunctionSpec,
    LLMProvider,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from nullain.memory import EpisodicMemory, PersistentMemory, TrajectoryRecord
from nullain.ports.clock import Clock, SystemClock
from nullain.router import Complexity, IntentParser, ModelRouter
from nullain.telemetry import get_cost_tracker, get_logger
from nullain.telemetry import span as telemetry_span
from nullain.tools import PermissionLevel, PermissionPolicy, ToolRegistry
from nullain.tools.result import ToolResult
from nullain.tools.sandbox import execute_subprocess

logger = get_logger(__name__)


def _step_signature(tool_calls: list[ToolCall]) -> str:
    """Build a stable signature for one step's requested tool calls.

    Used by loop detection: two steps with identical (tool_name, arguments)
    produce the same signature regardless of argument dict ordering or
    ``id`` fields, so genuinely-identical retries are detected while benign
    re-issues with changed arguments are not.
    """
    parts: list[str] = []
    for tc in tool_calls:
        args_json = _json.dumps(tc.arguments, sort_keys=True, default=str)
        parts.append(f"{tc.name}:{args_json}")
    return "|".join(parts)


def _project_authority_policy(
    base: PermissionPolicy | None, effective: Authority
) -> PermissionPolicy:
    """Build a child permission policy realising the effective authority.

    The authority gate already refuses tools whose required capabilities the
    subagent lacks. This projection folds the same bound into the existing
    permission engine so deny-pattern narrowing (delegated denies unioned with
    the policy's) and level tightening (WRITE/EXEC default-deny when absent
    from the effective capabilities) are enforced by the already-tested policy
    path too — belt and suspenders, no logic duplicated for the capability
    subset itself.
    """
    workspace = base.workspace_root if base is not None else "."
    write_level = base.default_write_level if base is not None else PermissionLevel.ASK
    exec_level = base.default_exec_level if base is not None else PermissionLevel.ASK
    read_level = base.default_read_level if base is not None else PermissionLevel.ALLOW
    return PermissionPolicy(
        workspace_root=workspace,
        default_read_level=read_level,
        default_write_level=(
            PermissionLevel.DENY if Capability.WRITE not in effective.capabilities else write_level
        ),
        default_exec_level=(
            PermissionLevel.DENY if Capability.EXEC not in effective.capabilities else exec_level
        ),
        # effective.deny_patterns already unions the base policy's denies via
        # the meet with Authority.from_policy(base).
        deny_patterns=sorted(effective.deny_patterns),
    )


@dataclass(frozen=True, slots=True)
class SpawnTask:
    """One sub-agent task for :meth:`AgentLoop.spawn_many` (M13).

    Mirrors :meth:`AgentLoop.spawn`'s per-call arguments so a fan-out is just
    a list of the same options ``spawn`` already accepts individually.
    """

    prompt: str
    tools: "ToolRegistry | None" = None
    model: str | None = None
    max_steps: int | None = None
    authority: "Authority | None" = None
    isolation: Literal["worktree", None] = None


@dataclass(frozen=True, slots=True)
class SpawnOutcome:
    """Result of one task from :meth:`AgentLoop.spawn_many`.

    ``error`` is set (and ``text`` is empty) when the task's ``spawn`` call
    raised a domain error (e.g. worktree setup failure, authority denial) —
    captured per-task so one failing sub-agent does not cancel its siblings.
    """

    text: str
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class AgentLoop:
    """Orchestrated Agent Execution Engine with Plan/Act/Verify, Routing & Memory."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        event_bus: EventBus | None = None,
        event_store: EventStore | None = None,
        model: str | None = None,
        router: ModelRouter | None = None,
        intent_parser: IntentParser | None = None,
        context_manager: ContextManager | None = None,
        spec_validator: SpecValidator | None = None,
        prompt_assembler: PromptAssembler | None = None,
        episodic_memory: EpisodicMemory | None = None,
        clock: Clock | None = None,
        hooks: HookManager | HooksConfig | None = None,
        persistent_memory: PersistentMemory | None = None,
        workspace_root: str | Path = ".",
        max_steps: int = 100,
        max_tokens: int | None = 2_000_000,
        timeout: float = 300.0,
        self_correction_max: int = 3,
        max_compaction_attempts: int = 3,
        loop_detection_threshold: int = 3,
        verify_retry_max: int = 2,
        max_steps_extension_ratio: float = 0.5,
        progress_window: int = 5,
        authority: Authority | None = None,
        tool_factory: Callable[[Path], ToolRegistry] | None = None,
    ) -> None:
        """Initialize AgentLoop with integrated engine collaborators.

        Args:
            provider: LLM provider adapter (required — no insecure default).
            tools: Tool registry with permission policy (required).
            event_bus: Optional event bus for pub-sub trajectory events.
            event_store: Optional persistent event store.
            model: Explicit model override (bypasses router).
            router: Model router for tier-based selection.
            intent_parser: Task intent classifier.
            context_manager: Context window compaction manager.
            spec_validator: Plan/Verify spec validator.
            prompt_assembler: Layered system prompt assembler.
            episodic_memory: SQLite-backed episodic memory.
            clock: Injectable clock for deterministic testing.
            hooks: Lifecycle hook manager (or raw HooksConfig) for pre_tool,
                post_tool, stop, and pre_compact command hooks.
            persistent_memory: Optional file-backed persistent memory whose
                index is re-injected into the system prompt (survives
                compaction). When None, no persistent memory is used.
            workspace_root: Workspace root directory path.
            max_steps: Maximum ReAct loop iterations.
            max_tokens: Token budget ceiling.
            timeout: Execution timeout in seconds.
            self_correction_max: Max self-correction retries per session.
            verify_retry_max: Max number of verify-fix-reverify cycles when
                the VERIFY phase finds unmet acceptance criteria. On each
                failed verification (with budget remaining and steps left),
                the feedback is injected as a correction message and the Act
                phase resumes instead of terminating the run, mirroring
                Claude Code's fix-and-reverify loop instead of surfacing a
                single-shot "verification_failed" the caller must retry
                itself.
            max_steps_extension_ratio: Fraction of ``max_steps`` granted as a
                single, one-time extension when the loop reaches ``max_steps``
                but has shown real progress recently (M14) — a write to one
                of the active spec's ``target_files`` within the last
                ``progress_window`` steps. E.g. 0.5 with ``max_steps=25``
                grants 12 extra steps (rounded down), once. A run stuck
                without progress gets no extension: it hits ``max_steps`` as
                before. Set to 0 to disable extensions entirely (strict fixed
                cap, pre-M14 behavior).
            progress_window: How many trailing steps count as "recent" for
                the extension check above. A target-file write further back
                than this does not qualify the run for an extension.
            authority: Authority enforced for this loop (P4.24). None = the
                unrestricted trust root (no capability/allowed-tool gate). A
                materialised Authority is set on spawned subagents by
                ``spawn`` and enforced at the registry execution gate.
            tool_factory: Callable that builds a fresh :class:`ToolRegistry`
                rooted at a given workspace path. Required for
                ``isolation="worktree"`` subagents (M11.4): tools bind to their
                workspace root at creation time, so a worktree-isolated child
                needs a registry re-rooted at the worktree directory. When
                None, worktree isolation is refused.
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.provider = provider
        self.tools = tools
        self.authority = authority
        self.event_bus = event_bus or EventBus()
        self.event_store = event_store
        self.explicit_model = model
        self.router = router or ModelRouter()
        self.intent_parser = intent_parser or IntentParser()
        self.context_manager = context_manager or ContextManager()
        self.spec_validator = spec_validator or SpecValidator()
        self.prompt_assembler = prompt_assembler or PromptAssembler(
            workspace_root=self.workspace_root
        )
        self.episodic_memory = episodic_memory
        self.clock = clock or SystemClock()
        if hooks is None or isinstance(hooks, HookManager):
            self.hooks: HookManager = hooks or HookManager()
        else:
            self.hooks = HookManager(hooks)
        self.persistent_memory = persistent_memory
        self._episodic_few_shot: str | None = None
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.self_correction_max = self_correction_max
        self.max_compaction_attempts = max_compaction_attempts
        # Consecutive identical step signatures (by tool_name + arguments) at
        # which the loop is considered stuck and is broken out of. Mirrors the
        # loop-detection Gemini CLI uses to escape thrash.
        self.loop_detection_threshold = loop_detection_threshold
        self.verify_retry_max = verify_retry_max
        self.max_steps_extension_ratio = max_steps_extension_ratio
        self.progress_window = progress_window
        self.tool_factory = tool_factory
        self._compaction_attempts = 0
        self._step_extension_granted = False
        # Incremental conversation fold (M10 D3): a cached ConversationState
        # updated by appending only new events, shared between the budget check
        # and message building in the same step to avoid folding the whole
        # history twice. Re-initialized per run in _run_pipeline_body.
        self._fold: IncrementalFold | None = None
        self._folded_count = 0

    async def _emit(self, event: BaseEvent) -> None:
        """Emit event to bus and persist to store if configured."""
        await self.event_bus.publish(event)
        if self.event_store is not None:
            await self.event_store.append(event)

    def _assemble_system_prompt(self) -> str:
        """Re-assemble the layered system prompt from current on-disk state.

        Re-reads AGENTS.md/SOUL and the persistent-memory index each call so
        that facts written mid-run (and project conventions) are picked up
        fresh — in particular they survive context compaction, mirroring
        Claude Code re-injecting CLAUDE.md / memory after a compaction event.
        """
        return self.prompt_assembler.assemble(
            episodic_memory=self._episodic_few_shot,
            memory_index=self.persistent_memory.to_context() if self.persistent_memory else None,
        )

    async def _generate_spec(self, prompt: str, model: str, system_prompt: str) -> TaskSpec:
        """Ask the LLM to generate a structured TaskSpec via forced tool-calling.

        Uses a synthetic ``emit_task_spec`` tool so the model returns
        structured, schema-validated arguments through the same tool-calling
        path already exercised for real tools — rather than free-text JSON
        that has to be fished out of markdown fences and can be truncated or
        preceded by chatter. Smaller open-source models are more prone to
        malformed free-text JSON than frontier models, so this path matters
        more here than it would for a stronger model.

        Falls back to parsing ``delta_text`` as JSON (legacy path, for models
        or providers that ignore the tool and answer in prose), then to a
        minimal generic spec if both fail.
        """
        spec_tool = ToolSpec(
            function=FunctionSpec(
                name="emit_task_spec",
                description="Emit the structured task plan for the requested work.",
                parameters={
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "steps": {"type": "array", "items": {"type": "string"}},
                        "target_files": {"type": "array", "items": {"type": "string"}},
                        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["objective", "steps"],
                },
            )
        )
        spec_instruction = (
            "Analyze the following task and call `emit_task_spec` with a "
            "structured plan: the objective, ordered steps, any files you "
            "expect to create or modify, and the acceptance criteria a "
            "verifier could check.\n\n"
            f"Task: {prompt}"
        )
        spec_messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=spec_instruction),
        ]
        req = CompletionRequest(
            model=model,
            messages=spec_messages,
            tools=[spec_tool],
            stream=False,
        )
        response = await self.provider.generate(req)

        for tc in response.tool_calls or []:
            if tc.name != "emit_task_spec":
                continue
            raw_args = tc.arguments
            args: dict[str, Any]
            if isinstance(raw_args, str):
                try:
                    parsed_args: object = _json.loads(raw_args)
                except _json.JSONDecodeError:
                    continue
                if not isinstance(parsed_args, dict):
                    continue
                args = cast(dict[str, Any], parsed_args)
            else:
                args = raw_args
            try:
                return TaskSpec.model_validate(args)
            except ValidationError:
                logger.warning("spec_tool_call_validation_failed", raw_args=str(args)[:200])

        # Legacy fallback: some models ignore `tools` and answer in prose.
        text = (response.delta_text or "").strip()
        if "```" in text:
            for part in text.split("```"):
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                if cleaned.startswith("{"):
                    text = cleaned
                    break

        try:
            data = _json.loads(text)
            if isinstance(data, dict):
                return TaskSpec.model_validate(data)
            raise ValueError("Expected JSON object for TaskSpec")
        except (_json.JSONDecodeError, ValueError, ValidationError, KeyError):
            logger.warning(
                "spec_generation_parse_failed",
                raw_text=text[:200],
            )
            return TaskSpec(
                objective=prompt,
                steps=["Execute the requested task"],
                acceptance_criteria=["Task completes without errors"],
            )

    async def _retrieve_few_shot(self, intent_type: str, sess_id: str) -> str | None:
        """Retrieve episodic memory few-shot examples for prompt injection."""
        if self.episodic_memory is None:
            return None
        try:
            past = await self.episodic_memory.get_relevant_examples(intent_type, limit=2)
            if past:
                lines = [f"- Objective: {ex.objective} (Steps: {ex.steps_count})" for ex in past]
                return "\n".join(lines)
        except Exception as err:
            logger.warning(
                "episodic_memory_lookup_failed",
                session_id=sess_id,
                error=str(err),
            )
            await self._emit(
                ErrorEvent(
                    session_id=sess_id,
                    error_type="EpisodicMemoryLookupError",
                    message=f"Few-shot retrieval failed: {err}",
                )
            )
        return None

    async def _run_tool_call(
        self,
        tc: ToolCall,
        sess_id: str,
    ) -> ToolResult:
        """Execute one tool call, returning a structured :class:`ToolResult`.

        Exceptions are normalized into structural error results so callers can
        dispatch a batch concurrently without ``gather`` short-circuiting.
        ``error_type`` is set when the failure warranted an ErrorEvent. Emits a
        ``tool`` telemetry span tagged with the outcome, including
        ``tool.blocked`` when execution was denied by the permission policy or
        vetoed by a ``pre_tool`` hook.
        """
        with telemetry_span("tool", **{"tool.name": tc.name}) as tool_span:
            # pre_tool hooks: exit 2 vetoes the call before it runs.
            if self.hooks.enabled:
                pre_outcomes = await self.hooks.run(
                    HookLifecycle.PRE_TOOL,
                    {
                        "session_id": sess_id,
                        "call_id": tc.id,
                        "tool_name": tc.name,
                        "arguments": tc.arguments,
                    },
                )
                if any(o.blocked for o in pre_outcomes):
                    reason = next(
                        (o.stdout.strip() for o in pre_outcomes if o.blocked and o.stdout.strip()),
                        "",
                    )
                    msg = f"Tool '{tc.name}' blocked by pre_tool hook"
                    if reason:
                        msg = f"{msg}: {reason}"
                    logger.info(
                        "tool_blocked_by_hook",
                        session_id=sess_id,
                        tool=tc.name,
                    )
                    tool_span.set_attributes(
                        {"tool.is_error": True, "tool.blocked": True, "tool.hook_blocked": True}
                    )
                    return ToolResult(output=msg, is_error=True, error_type="HookBlocked")

            try:
                if isinstance(tc.arguments, str):
                    # A call reaching execution must have complete, parsed
                    # arguments (the streaming merge finalizes them); a raw
                    # fragment here indicates an incomplete call.
                    raise ToolError(f"Tool '{tc.name}' called with incomplete (unparsed) arguments")
                result = await self.tools.execute(tc.name, tc.arguments)
                tool_span.set_attributes({"tool.is_error": result.is_error, "tool.blocked": False})
            except ToolPermissionError as err:
                logger.warning(
                    "tool_permission_denied",
                    session_id=sess_id,
                    tool=tc.name,
                    error=str(err),
                )
                tool_span.set_attributes({"tool.is_error": True, "tool.blocked": True})
                result = ToolResult(
                    output=f"Tool Execution Error ({tc.name}): {err}",
                    is_error=True,
                    error_type="ToolExecutionError",
                )
            except ToolError as err:
                logger.warning(
                    "tool_execution_failed",
                    session_id=sess_id,
                    tool=tc.name,
                    error=str(err),
                )
                tool_span.set_attributes({"tool.is_error": True, "tool.blocked": False})
                result = ToolResult(
                    output=f"Tool Execution Error ({tc.name}): {err}",
                    is_error=True,
                    error_type="ToolExecutionError",
                )
            except Exception as err:
                logger.error(
                    "unexpected_tool_error",
                    session_id=sess_id,
                    tool=tc.name,
                    error=str(err),
                )
                tool_span.set_attributes({"tool.is_error": True, "tool.blocked": False})
                result = ToolResult(
                    output=f"Unexpected Error ({tc.name}): {err}",
                    is_error=True,
                    error_type="UnexpectedToolError",
                )

            # post_tool hooks: notification only (no block semantics).
            if self.hooks.enabled:
                await self.hooks.run(
                    HookLifecycle.POST_TOOL,
                    {
                        "session_id": sess_id,
                        "call_id": tc.id,
                        "tool_name": tc.name,
                        "arguments": tc.arguments,
                        "output": result.output,
                        "is_error": result.is_error,
                    },
                )
            return result

    @staticmethod
    def _resource_key(tc: ToolCall) -> str | None:
        """Derive the filesystem resource a tool call targets, if any.

        Mirrors the path-argument heuristic ``ToolRegistry._evaluate_permission``
        already uses for permission checks (``file_path`` / ``target_file`` /
        ``path``). Returns ``None`` when the call's arguments are unparsed
        (streaming fragment) or carry no recognizable path — such calls are
        treated as touching an tool-scoped resource (see ``_batch_by_conflict``)
        rather than assumed independent, since a tool like ``bash`` can touch
        anything.
        """
        args = tc.arguments
        if not isinstance(args, dict):
            return None
        for key in ("file_path", "target_file", "path"):
            value = args.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @classmethod
    def _batch_by_conflict(
        cls, tool_calls: list[ToolCall], tools: ToolRegistry
    ) -> list[list[ToolCall]]:
        """Group tool calls into ordered batches that can run concurrently.

        Two calls conflict — and must land in different batches, preserving
        submission order between them — when they are not both read-only AND
        they touch the same resource. A call whose resource cannot be
        determined (no recognizable path argument, e.g. ``bash``) is treated
        as touching a resource scoped to its own tool name, so two ``bash``
        calls still serialize with each other while a ``bash`` call runs
        alongside unrelated file edits.

        This subsumes the previous read-only-only concurrency rule: batches
        of read-only calls with no path in common, or read/write calls on
        disjoint files, now run in parallel too — matching Claude Code's
        finer-grained parallel tool dispatch instead of an all-or-nothing gate.
        """
        batches: list[list[ToolCall]] = []
        # For each batch, the resources already claimed by a non-read-only
        # call in it (so a later read-only call on the same resource still
        # conflicts and must wait for the next batch).
        batch_resources: list[dict[str, bool]] = []  # resource -> is_write

        for tc in tool_calls:
            read_only = tools.is_read_only(tc.name)
            resource = cls._resource_key(tc) or f"__tool__:{tc.name}"

            placed = False
            for batch, resources in zip(batches, batch_resources, strict=True):
                existing_is_write = resources.get(resource)
                conflicts = existing_is_write is not None and (existing_is_write or not read_only)
                if conflicts:
                    continue
                batch.append(tc)
                resources[resource] = resources.get(resource, False) or not read_only
                placed = True
                break

            if not placed:
                batches.append([tc])
                batch_resources.append({resource: not read_only})

        return batches

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        sess_id: str,
        accumulated_events: list[BaseEvent],
        correction_budget: int,
    ) -> tuple[str, int]:
        """Execute tool calls with self-correction injection on errors.

        Dispatch policy (M13): calls are grouped into conflict-free batches
        via ``_batch_by_conflict`` — same-resource writes and any write
        sharing a resource with a read stay ordered relative to each other,
        while independent calls (disjoint files, or all read-only) run
        concurrently within a batch. Batches themselves run in submission
        order so a later batch never starts before an earlier one settles.

        Returns:
            Tuple of (last_output, remaining_correction_budget).
        """
        # Emit all ToolCallEvents up front, in submission order, so the
        # trajectory records the model's requested batch before any results.
        for tc in tool_calls:
            tc_ev = ToolCallEvent(
                session_id=sess_id,
                call_id=tc.id,
                tool_name=tc.name,
                arguments=tc.arguments,
            )
            await self._emit(tc_ev)
            accumulated_events.append(tc_ev)

        batches = self._batch_by_conflict(tool_calls, self.tools)
        outcomes_by_id: dict[str, ToolResult] = {}
        for batch in batches:
            if len(batch) > 1:
                batch_outcomes = await asyncio.gather(
                    *(self._run_tool_call(tc, sess_id) for tc in batch)
                )
            else:
                batch_outcomes = [await self._run_tool_call(batch[0], sess_id)]
            for tc, res in zip(batch, batch_outcomes, strict=True):
                outcomes_by_id[tc.id] = res
        # Restore submission order regardless of batch grouping, so events are
        # emitted (and self-correction budget consumed) deterministically.
        outcomes = [outcomes_by_id[tc.id] for tc in tool_calls]

        last_output = ""
        for tc, res in zip(tool_calls, outcomes, strict=True):
            if res.error_type is not None:
                err_ev = ErrorEvent(
                    session_id=sess_id,
                    error_type=res.error_type,
                    message=res.output,
                )
                await self._emit(err_ev)
                accumulated_events.append(err_ev)

            res_ev = ToolResultEvent(
                session_id=sess_id,
                call_id=tc.id,
                tool_name=tc.name,
                output=res.output,
                is_error=res.is_error,
            )
            await self._emit(res_ev)
            accumulated_events.append(res_ev)
            last_output = res.output

            # Self-correction: inject reflection on error
            if res.is_error and correction_budget > 0:
                correction_budget -= 1
                reflection = UserMessageEvent(
                    session_id=sess_id,
                    content=(
                        f"[SELF-CORRECTION] Tool '{tc.name}' "
                        f"failed: {res.output}\n"
                        "Analyze this error. Was the input "
                        "malformed? Is there an alternative "
                        "approach? Try a different strategy."
                    ),
                )
                await self._emit(reflection)
                accumulated_events.append(reflection)

        return last_output, correction_budget

    def _step_touched_target_file(
        self, tool_calls: list[ToolCall], active_spec: TaskSpec | None
    ) -> bool:
        """Whether this step's calls wrote/edited one of the spec's target_files.

        Used to gate incremental verification (M14): running
        ``spec.verification_commands`` (e.g. a test suite) is only worth its
        cost right after a write that could plausibly satisfy or break them,
        not on every step (which would re-run the command uselessly on
        read-only exploration steps) and not only once at the very end (which
        is what full-run VERIFY already does).
        """
        if active_spec is None or not active_spec.target_files:
            return False
        target_set = set(active_spec.target_files)
        for tc in tool_calls:
            if self.tools.is_read_only(tc.name):
                continue
            resource = self._resource_key(tc)
            if resource is not None and resource in target_set:
                return True
        return False

    async def _run_incremental_verify(
        self,
        active_spec: TaskSpec,
        sess_id: str,
        accumulated_events: list[BaseEvent],
    ) -> None:
        """Run spec.verification_commands right after a target-file write (M14).

        Unlike the end-of-run VERIFY phase, this only runs
        ``verification_commands`` (a test suite, a linter, ...) — not the
        target-file-exists check or the acceptance-criteria judge, both of
        which are meaningful only once the model believes it is done. A
        failure here is reported the same way a tool error is: as a
        [SELF-CORRECTION]-style reflection, so the model learns it broke
        something several steps before it would otherwise find out at the
        final VERIFY — mirroring Claude Code running tests as it goes rather
        than only at the end of a task.

        Silently no-ops when there are no ``verification_commands`` to run.
        """
        if not active_spec.verification_commands:
            return
        failed: list[str] = []
        for cmd in active_spec.verification_commands:
            cmd_tokens = cmd.strip().split()
            if not cmd_tokens:
                continue
            try:
                cmd_res = await self.tools.execute("bash", {"command_args": cmd_tokens})
                if cmd_res.is_error:
                    failed.append(f"'{cmd}' failed: {cmd_res.output}")
            except Exception as err:
                failed.append(f"'{cmd}' execution error: {err}")

        if not failed:
            return
        reflection = UserMessageEvent(
            session_id=sess_id,
            content=(
                "[INCREMENTAL-VERIFY] Running the task's verification commands "
                f"after your last change found a problem: {'; '.join(failed)}\n"
                "Fix this before continuing."
            ),
        )
        await self._emit(reflection)
        accumulated_events.append(reflection)

    async def _call_provider(
        self,
        req: CompletionRequest,
        active_model: str,
        sess_id: str,
    ) -> CompletionChunk:
        """Call provider.generate() with circuit breaker integration."""
        try:
            response = await self.provider.generate(req)
            self.router.circuit_breaker.record_success(active_model)
            return response
        except Exception as err:
            self.router.circuit_breaker.record_failure(active_model)
            logger.error(
                "model_generation_failed",
                session_id=sess_id,
                model=active_model,
                error=str(err),
            )
            err_ev = ErrorEvent(
                session_id=sess_id,
                error_type="ModelGenerationError",
                message=(f"Generation failed on model {active_model}: {err}"),
            )
            await self._emit(err_ev)
            raise

    def _sync_fold(self, sess_id: str, accumulated_events: list[BaseEvent]) -> ConversationState:
        """Append newly-accumulated events to the incremental fold and return state.

        Only the events not yet folded are applied, so the derived state is
        shared (and updated once) across the budget check and message building
        in the same step instead of folding the whole history twice.
        """
        if self._fold is None:
            self._fold = IncrementalFold(sess_id)
        new_events = accumulated_events[self._folded_count :]
        if new_events:
            self._fold.append(new_events)
            self._folded_count = len(accumulated_events)
        return self._fold.state

    def _build_messages(
        self,
        sess_id: str,
        accumulated_events: list[BaseEvent],
        system_prompt: str,
        step: int,
    ) -> list[ChatMessage]:
        """Fold events into messages with system prompt and centrifugation.

        The immutable rules system prompt is ALWAYS re-injected as the first
        message. After compaction, ``Conversation.fold`` places a compaction
        summary as a system message at index 0; the rules must precede it so
        persistent operational rules survive compaction (mirrors Claude Code
        re-injecting CLAUDE.md after compaction) rather than being replaced by
        the recap.
        """
        state = self._sync_fold(sess_id, accumulated_events)
        # Copy so the shared incremental state's message list is not mutated.
        messages = list(state.messages)
        messages.insert(0, ChatMessage(role="system", content=system_prompt))
        return self.context_manager.reinject_instructions(messages, step)

    async def _check_budget_and_compact(
        self,
        sess_id: str,
        accumulated_events: list[BaseEvent],
        total_tokens: int,
        active_spec: TaskSpec | None,
        system_prompt: str,
        active_model: str,
    ) -> None:
        """Check token budget and compact context if needed.

        Thrashing protection: if compaction is required on more than
        ``max_compaction_attempts`` consecutive steps (i.e. compacting did not
        free enough tokens to get back under threshold), raise
        ``ContextWindowExhaustedError`` instead of looping forever — a single
        huge tool output that refills the window immediately after each
        compaction would otherwise spin indefinitely.
        """
        if self.max_tokens is not None and total_tokens >= self.max_tokens:
            err_ev = ErrorEvent(
                session_id=sess_id,
                error_type="BudgetExceededError",
                message=f"Token budget of {self.max_tokens} exceeded",
            )
            await self._emit(err_ev)
            raise BudgetExceededError(f"Token budget of {self.max_tokens} exceeded")

        # Compact based on actual context size, not cumulative. Reuse the
        # incremental fold shared with _build_messages (M10 D3). Prefer the
        # provider-reported real prompt-token count (M10 D5) when available,
        # falling back to the len//4 estimate otherwise.
        state = self._sync_fold(sess_id, accumulated_events)
        if state.last_prompt_tokens > 0:
            ctx_tokens = state.last_prompt_tokens
        else:
            messages = list(state.messages)
            messages.insert(0, ChatMessage(role="system", content=system_prompt))
            ctx_tokens = self.context_manager.estimate_context_tokens(messages)
        if not self.context_manager.should_compact(ctx_tokens):
            # Context is under control again; reset the thrash counter.
            self._compaction_attempts = 0
            return

        self._compaction_attempts += 1
        if self._compaction_attempts > self.max_compaction_attempts:
            err_ev = ErrorEvent(
                session_id=sess_id,
                error_type="ContextWindowExhausted",
                message=(
                    f"Context window exhausted: compaction attempted "
                    f"{self.max_compaction_attempts} times without convergence"
                ),
            )
            await self._emit(err_ev)
            raise ContextWindowExhaustedError(
                f"Context window exhausted after {self.max_compaction_attempts} compaction attempts"
            )

        # pre_compact hooks: exit 2 vetoes compaction for this step. The
        # thrash counter has already been incremented; we skip the compaction
        # itself and let the next step re-evaluate. Context will keep growing
        # and may hit the hard budget — that is the honest consequence of a
        # user-configured veto.
        if self.hooks.enabled:
            pre_outcomes = await self.hooks.run(
                HookLifecycle.PRE_COMPACT,
                {
                    "session_id": sess_id,
                    "current_tokens": ctx_tokens,
                    "event_count": len(accumulated_events),
                },
            )
            if any(o.blocked for o in pre_outcomes):
                logger.info(
                    "compaction_blocked_by_hook",
                    session_id=sess_id,
                    current_tokens=ctx_tokens,
                )
                return

        compact_ev = await self.context_manager.compact(
            session_id=sess_id,
            events=accumulated_events,
            active_spec=active_spec,
            provider=self.provider,
            model=active_model,
        )
        await self._emit(compact_ev)
        accumulated_events.append(compact_ev)

    async def _record_trajectory(
        self,
        sess_id: str,
        intent: str,
        model: str,
        steps: int,
        success: bool,
        objective: str,
    ) -> None:
        """Record completed trajectory to episodic memory."""
        if self.episodic_memory is None:
            return
        try:
            rec = TrajectoryRecord(
                session_id=sess_id,
                intent=intent,
                model=model,
                steps_count=steps,
                success=success,
                objective=objective,
            )
            await self.episodic_memory.record_trajectory(rec)
        except Exception as err:
            logger.warning(
                "episodic_memory_record_failed",
                session_id=sess_id,
                error=str(err),
            )
            await self._emit(
                ErrorEvent(
                    session_id=sess_id,
                    error_type="EpisodicMemoryRecordError",
                    message=f"Trajectory record failed: {err}",
                )
            )

    @staticmethod
    def _merge_tool_call_deltas(
        accumulator: dict[str, ToolCall],
        deltas: list[ToolCall],
    ) -> None:
        """Merge streaming tool-call argument fragments by id (M10 D4).

        OpenAI-style streaming delivers a tool call's ``arguments`` as a sequence
        of raw JSON string fragments across chunks (the id is present on the
        first chunk and empty on the rest); Ollama native may deliver complete
        dicts. Fragments for the same id are concatenated; a call is only
        complete once its JSON parses or the stream ends (see
        :meth:`_finalize_tool_calls`).
        """
        for delta in deltas:
            key = delta.id or (next(reversed(accumulator)) if accumulator else "")
            if not key:
                continue
            existing = accumulator.get(key)
            if existing is None:
                accumulator[key] = delta
                continue
            if isinstance(delta.arguments, str):
                if isinstance(existing.arguments, str):
                    existing.arguments = existing.arguments + delta.arguments
                else:
                    existing.arguments = str(existing.arguments) + delta.arguments
            else:
                # Complete dict delivered in one chunk (e.g. Ollama native).
                existing.arguments = delta.arguments

    @staticmethod
    def _finalize_tool_calls(accumulator: dict[str, ToolCall]) -> list[ToolCall]:
        """Parse any remaining string argument fragments into complete dicts.

        A call whose JSON still does not parse at stream end is incomplete and
        cannot be executed, so it is dropped.
        """
        finalized: list[ToolCall] = []
        for tc in accumulator.values():
            if isinstance(tc.arguments, str):
                try:
                    parsed: object = _json.loads(tc.arguments)
                except _json.JSONDecodeError:
                    continue
                if not isinstance(parsed, dict):
                    continue
                tc.arguments = parsed
            finalized.append(tc)
        return finalized

    async def _generate_step_response(
        self,
        req: CompletionRequest,
        active_model: str,
        sess_id: str,
        streaming: bool,
    ) -> tuple[str, list[ToolCall], TokenUsage | None]:
        """Generate response chunk either via streaming or direct generate.

        Emits an ``llm_request`` telemetry span with model, token usage,
        time-to-first-token (streaming) or latency (non-streaming), and the
        computed cost from the cost tracker.
        """
        tracker = get_cost_tracker()
        with telemetry_span(
            "llm_request", **{"llm.model": active_model, "llm.stream": streaming}
        ) as llm_span:
            start = self.clock.now()
            if streaming:
                full_text = ""
                tool_call_acc: dict[str, ToolCall] = {}
                usage: TokenUsage | None = None
                ttft: float | None = None
                try:
                    async for chunk in self.provider.stream(req):
                        if ttft is None:
                            ttft = (self.clock.now() - start) * 1000.0
                        if chunk.delta_text:
                            full_text += chunk.delta_text
                            delta_ev = StreamDeltaEvent(
                                session_id=sess_id,
                                delta=chunk.delta_text,
                                model=active_model,
                            )
                            await self._emit(delta_ev)
                        if chunk.tool_calls:
                            # Merge argument fragments by id (M10 D4) instead of
                            # appending each partial call separately.
                            self._merge_tool_call_deltas(tool_call_acc, chunk.tool_calls)
                        if chunk.usage:
                            usage = chunk.usage
                    self.router.circuit_breaker.record_success(active_model)
                    latency_ms = (self.clock.now() - start) * 1000.0
                    self._record_llm_telemetry(
                        llm_span, tracker, active_model, sess_id, usage, ttft, latency_ms
                    )
                    all_tool_calls = self._finalize_tool_calls(tool_call_acc)
                    return full_text, all_tool_calls, usage
                except Exception as err:
                    self.router.circuit_breaker.record_failure(active_model)
                    logger.error(
                        "model_stream_failed",
                        session_id=sess_id,
                        model=active_model,
                        error=str(err),
                    )
                    err_ev = ErrorEvent(
                        session_id=sess_id,
                        error_type="ModelStreamError",
                        message=f"Stream failed on model {active_model}: {err}",
                    )
                    await self._emit(err_ev)
                    raise
            else:
                response = await self._call_provider(req, active_model, sess_id)
                tool_calls = list(response.tool_calls) if response.tool_calls else []
                # Non-streaming calls carry complete dict arguments; parse any
                # delivered as a raw string (malformed JSON) for safety.
                tool_calls = self._finalize_tool_calls({tc.id: tc for tc in tool_calls})
                latency_ms = (self.clock.now() - start) * 1000.0
                self._record_llm_telemetry(
                    llm_span, tracker, active_model, sess_id, response.usage, latency_ms, latency_ms
                )
                return response.delta_text or "", tool_calls, response.usage

    @staticmethod
    def _record_llm_telemetry(
        llm_span: Any,
        tracker: Any,
        model: str,
        sess_id: str,
        usage: TokenUsage | None,
        ttft_ms: float | None,
        latency_ms: float,
    ) -> None:
        """Record token usage + cost attributes on an llm_request span."""
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        cost = tracker.record(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            agent=sess_id,
        )
        llm_span.set_attributes(
            {
                "llm.prompt_tokens": prompt_tokens,
                "llm.completion_tokens": completion_tokens,
                "llm.total_tokens": total_tokens,
                "llm.ttft_ms": ttft_ms if ttft_ms is not None else latency_ms,
                "llm.latency_ms": latency_ms,
                "llm.cost_usd": cost,
            }
        )

    async def _run_act_phase(
        self,
        sess_id: str,
        accumulated_events: list[BaseEvent],
        active_model: str,
        active_spec: TaskSpec | None,
        start_time: float,
        streaming: bool,
        step: int,
        total_tokens: int,
        correction_budget: int,
        system_prompt: str,
    ) -> tuple[int, int, int, str, str, bool, RunStatus | None, str | None, bool]:
        """Run the ReAct Act loop until a final answer, terminal failure, or cap.

        Called once per verify-fix-reverify cycle (M12): ``step`` and
        ``total_tokens`` carry over across calls so the step cap and token
        budget span the whole run, not just one cycle. ``correction_budget``
        carries over *within* a cycle (across the internal ReAct steps this
        call itself makes) but the caller resets it to
        ``self.self_correction_max`` before each new verify-fix-reverify
        cycle (M14) — each cycle is a fresh attempt at the task and gets its
        own self-correction allowance, rather than a first attempt's tool
        retries starving a later cycle's budget. Loop-detection state
        (``last_step_signature`` / ``repeat_count``) is local to each call —
        a fresh cycle starts after a VERIFY-CORRECTION message has been
        injected, which is itself a different step signature, so carrying
        detection state over would only cost an extra step before the new
        pattern is recognized.

        Returns:
            Tuple of (step, total_tokens, correction_budget, final_text,
            last_output, completed, terminal_status, terminal_error,
            loop_detected).
        """
        final_text = ""
        last_output = ""
        completed = False
        terminal_status: RunStatus | None = None
        terminal_error: str | None = None
        loop_detected = False
        last_step_signature: str | None = None
        repeat_count = 0
        # Steps since a target-file write, for the adaptive extension check
        # below. Starts at progress_window + 1 (i.e. "no recent progress")
        # rather than 0, so a run that never touches a target file (no
        # active_spec, or a task with none) never qualifies for an extension.
        steps_since_progress = self.progress_window + 1

        try:
            effective_max_steps = self.max_steps
            while step < effective_max_steps:
                # Timeout check
                elapsed = self.clock.now() - start_time
                if elapsed > self.timeout:
                    err_ev = ErrorEvent(
                        session_id=sess_id,
                        error_type="TimeoutError",
                        message=f"Agent loop timed out after {self.timeout} seconds",
                    )
                    await self._emit(err_ev)
                    raise NullainError(f"Agent loop timed out after {self.timeout} seconds")

                # Budget & compaction
                await self._check_budget_and_compact(
                    sess_id,
                    accumulated_events,
                    total_tokens,
                    active_spec,
                    system_prompt,
                    active_model,
                )

                step += 1
                # Re-assemble the system prompt so persistent memory and
                # AGENTS.md are re-injected after compaction (and stay fresh
                # if memory was written earlier this step).
                system_prompt = self._assemble_system_prompt()
                messages = self._build_messages(sess_id, accumulated_events, system_prompt, step)

                tool_specs = self.tools.list_specs()
                req = CompletionRequest(
                    model=active_model,
                    messages=messages,
                    tools=tool_specs if tool_specs else None,
                    stream=streaming,
                )

                step_text, tool_calls, usage = await self._generate_step_response(
                    req=req,
                    active_model=active_model,
                    sess_id=sess_id,
                    streaming=streaming,
                )

                if usage:
                    total_tokens += usage.total_tokens
                    # Calibrate the context estimator against the provider's
                    # real prompt-token count for exactly the messages we just
                    # sent, so compaction fires on measured size rather than on
                    # a fixed chars-per-token guess.
                    self.context_manager.calibrate(messages, usage.prompt_tokens)

                model_ev = ModelResponseEvent(
                    session_id=sess_id,
                    model=active_model,
                    content=step_text or None,
                    tool_calls=(tuple(tool_calls) if tool_calls else None),
                    usage=usage,
                )
                await self._emit(model_ev)
                accumulated_events.append(model_ev)

                # No tool calls = final answer
                if not tool_calls:
                    final_text = step_text
                    last_output = final_text
                    completed = True
                    break

                # Loop detection: hash the requested tool calls (name + args) for
                # this step. If the same signature repeats for
                # ``loop_detection_threshold`` consecutive steps, the agent is
                # stuck; inject a strong self-correction and stop the loop
                # rather than burning the remaining step budget.
                step_signature = _step_signature(tool_calls)
                if step_signature == last_step_signature:
                    repeat_count += 1
                else:
                    last_step_signature = step_signature
                    repeat_count = 1

                if repeat_count >= self.loop_detection_threshold:
                    loop_detected = True
                    err_ev = ErrorEvent(
                        session_id=sess_id,
                        error_type="LoopDetected",
                        message=(
                            f"Agent loop repeated the same tool calls for "
                            f"{repeat_count} consecutive steps; stopping to avoid thrash."
                        ),
                    )
                    await self._emit(err_ev)
                    accumulated_events.append(err_ev)
                    reflection = UserMessageEvent(
                        session_id=sess_id,
                        content=(
                            "[SELF-CORRECTION] You have repeated the same tool calls "
                            f"{repeat_count} times in a row with no progress. "
                            "Stop repeating. Either produce a final answer, or choose a "
                            "different approach with different tool arguments."
                        ),
                    )
                    await self._emit(reflection)
                    accumulated_events.append(reflection)
                    break

                # Execute tools with self-correction
                last_output, correction_budget = await self._execute_tools(
                    tool_calls,
                    sess_id,
                    accumulated_events,
                    correction_budget,
                )

                # Incremental verification (M14): a write to one of the
                # spec's target_files runs verification_commands right away
                # instead of waiting for the end-of-run VERIFY phase, so a
                # broken test/lint surfaces while there are still steps left
                # to fix it.
                touched_target = self._step_touched_target_file(tool_calls, active_spec)
                if touched_target:
                    assert active_spec is not None  # guaranteed by the check above
                    await self._run_incremental_verify(active_spec, sess_id, accumulated_events)
                    steps_since_progress = 0
                else:
                    steps_since_progress += 1

                # Adaptive step budget (M14): reaching the cap with recent
                # real progress (a target-file write within the last
                # progress_window steps) grants a one-time extension instead
                # of ending the run right where it was still moving forward.
                # A run with no recent progress gets no extension — it hits
                # max_steps as it always did. Granted at most once per run
                # (across verify-fix-reverify cycles too, via the instance
                # flag reset in _run_pipeline_body).
                if (
                    step >= effective_max_steps
                    and not self._step_extension_granted
                    and self.max_steps_extension_ratio > 0
                    and steps_since_progress <= self.progress_window
                ):
                    extension = int(self.max_steps * self.max_steps_extension_ratio)
                    if extension > 0:
                        effective_max_steps += extension
                        self._step_extension_granted = True
                        logger.info(
                            "step_budget_extended",
                            session_id=sess_id,
                            original_max_steps=self.max_steps,
                            effective_max_steps=effective_max_steps,
                        )
        except BudgetExceededError as err:
            terminal_status = "budget"
            terminal_error = str(err)
        except ContextWindowExhaustedError as err:
            terminal_status = "context_exhausted"
            terminal_error = str(err)
        except ProviderError as err:
            # A provider/API failure (bad request, auth, rate limit, ...)
            # that exhausted the provider's own retries. Reported distinctly
            # from "timeout" — this is not a step-budget or wall-clock issue,
            # it is the provider rejecting or failing the request itself, and
            # collapsing it into "timeout" (as the broader NullainError catch
            # below used to) hid the real cause from RunResult.error.
            terminal_status = "error"
            terminal_error = str(err)
        except NullainError as err:
            # Timeout (raised as a plain NullainError) and any other domain
            # failure surfaced from the act phase.
            terminal_status = "timeout"
            terminal_error = str(err)

        return (
            step,
            total_tokens,
            correction_budget,
            final_text,
            last_output,
            completed,
            terminal_status,
            terminal_error,
            loop_detected,
        )

    async def _run_pipeline(
        self,
        prompt: str,
        session_id: str | None = None,
        events_history: list[BaseEvent] | None = None,
        streaming: bool = False,
    ) -> RunResult:
        """Run orchestrated agent execution pipeline (telemetry root span).

        Wraps the pipeline body in an ``agent.run`` span so the full
        interaction (intent → route → plan → act → verify → memory) is
        recorded as one trace, with child ``llm_request`` / ``tool`` spans.
        """
        with telemetry_span("agent.run") as root_span:
            try:
                return await self._run_pipeline_body(
                    prompt, session_id, events_history, streaming, root_span
                )
            except get_cancelled_exc_class():
                # Cancellation requested (e.g. daemon session.cancel or closed
                # stdin). asyncio propagates the cancellation to in-flight child
                # work (HTTP request, subprocess, subagent spawn) and awaits its
                # cleanup; surface the outcome as a structured "cancelled"
                # RunResult rather than a raw exception. (A bare anyio.CancelScope
                # would re-raise on exit when the body returns normally, so the
                # cancellation is caught directly here.)
                return RunResult(
                    session_id=session_id or str(uuid.uuid4()),
                    status="cancelled",
                    success=False,
                    error="Agent run cancelled",
                )

    async def _run_pipeline_body(
        self,
        prompt: str,
        session_id: str | None,
        events_history: list[BaseEvent] | None,
        streaming: bool,
        root_span: Any,
    ) -> RunResult:
        """Run orchestrated agent execution pipeline.

        Pipeline: Intent → Route → Plan → Act → Verify → Memory.

        Returns a structured ``RunResult``. Terminal-resource failures
        (budget, timeout, context exhaustion) are captured as statuses rather
        than raised, so structured callers can branch on outcome.
        """
        sess_id = session_id or str(uuid.uuid4())
        root_span.set_attribute("session.id", sess_id)
        start_time = self.clock.now()
        accumulated_events: list[BaseEvent] = list(events_history or [])
        correction_budget = self.self_correction_max
        self._compaction_attempts = 0
        # Adaptive step budget (M14): the one-time extension is granted at
        # most once per run, tracked across verify-fix-reverify cycles.
        self._step_extension_granted = False
        # Fresh incremental fold per run (M10 D3).
        self._fold = None
        self._folded_count = 0

        # 1. Intent Parsing & Model Routing
        # M11.3: parse_async runs the deterministic heuristics first and only
        # calls the LLM classifier (router.classifier_model) when no heuristic
        # matches with confidence, falling back to the heuristic on any failure.
        intent_res = await self.intent_parser.parse_async(
            prompt, self.provider, self.router.config.classifier_model
        )
        active_model = self.explicit_model or self.router.route_intent(intent_res)
        logger.info(
            "task_intent_classified",
            session_id=sess_id,
            intent=intent_res.intent_type,
            complexity=intent_res.complexity,
            model=active_model,
            streaming=streaming,
        )
        root_span.set_attributes({"llm.model": active_model, "task.intent": intent_res.intent_type})

        # Emit initial UserMessageEvent
        user_ev = UserMessageEvent(session_id=sess_id, content=prompt)
        await self._emit(user_ev)
        accumulated_events.append(user_ev)

        # 2. Few-Shot Memory Retrieval
        few_shot_str = await self._retrieve_few_shot(intent_res.intent_type, sess_id)
        self._episodic_few_shot = few_shot_str

        # Assemble Layered System Prompt (re-assembled each step so persistent
        # memory / AGENTS.md survive compaction — see _assemble_system_prompt).
        system_prompt = self._assemble_system_prompt()

        # 3. Plan Phase — LLM generates spec for MEDIUM/HIGH tasks
        active_spec: TaskSpec | None = None
        if intent_res.complexity in (
            Complexity.MEDIUM,
            Complexity.HIGH,
        ):
            # A provider failure during Plan (e.g. the API rejecting the
            # request, or exhausted-retry auth/rate-limit errors) must not
            # propagate as a raw exception — run_result() promises to always
            # return a structured RunResult for domain failures, and this
            # call sits outside the Act phase's own ProviderError handling.
            try:
                active_spec = await self._generate_spec(prompt, active_model, system_prompt)
            except ProviderError as err:
                err_ev = ErrorEvent(
                    session_id=sess_id,
                    error_type="ProviderError",
                    message=f"Plan phase failed: {err}",
                )
                await self._emit(err_ev)
                return RunResult(
                    session_id=sess_id,
                    status="error",
                    success=False,
                    model=active_model,
                    intent=intent_res.intent_type,
                    error=str(err),
                )
            self.spec_validator.validate_spec(active_spec)
            spec_ev = SpecCreatedEvent(
                session_id=sess_id,
                spec_id=active_spec.spec_id,
                title=active_spec.objective,
                steps=tuple(active_spec.steps),
            )
            await self._emit(spec_ev)
            accumulated_events.append(spec_ev)

        # 4. Act Phase (ReAct Loop), wrapped in a verify-fix-reverify cycle
        # (M12): a failed VERIFY re-enters Act with the feedback injected as a
        # correction message, instead of terminating on the first miss.
        step = 0
        total_tokens = 0
        final_text = ""
        last_output = ""
        completed = False
        terminal_status: RunStatus | None = None
        terminal_error: str | None = None
        loop_detected = False
        feedback: str | None = None
        verify_failed = False
        verify_attempt = 0

        while True:
            (
                step,
                total_tokens,
                correction_budget,
                final_text,
                last_output,
                completed,
                terminal_status,
                terminal_error,
                loop_detected,
            ) = await self._run_act_phase(
                sess_id=sess_id,
                accumulated_events=accumulated_events,
                active_model=active_model,
                active_spec=active_spec,
                start_time=start_time,
                streaming=streaming,
                step=step,
                total_tokens=total_tokens,
                correction_budget=correction_budget,
                system_prompt=system_prompt,
            )

            feedback = None
            verify_failed = False
            if completed and active_spec and terminal_status is None:
                verified, feedback = await self.spec_validator.verify(
                    active_spec,
                    last_output,
                    workspace_root=self.workspace_root,
                    tools=self.tools,
                )
                verify_failed = not verified
                verify_ev = SpecVerifiedEvent(
                    session_id=sess_id,
                    spec_id=active_spec.spec_id,
                    success=verified,
                    feedback=feedback,
                )
                await self._emit(verify_ev)
                accumulated_events.append(verify_ev)

            if not (
                verify_failed
                and verify_attempt < self.verify_retry_max
                and step < self.max_steps
                and terminal_status is None
                and not loop_detected
            ):
                break

            verify_attempt += 1
            fix_ev = UserMessageEvent(
                session_id=sess_id,
                content=(
                    f"[VERIFY-CORRECTION {verify_attempt}/{self.verify_retry_max}] "
                    "The previous answer did not satisfy the task's acceptance "
                    f"criteria: {feedback}\n"
                    "Fix the issues above, then provide a new final answer."
                ),
            )
            await self._emit(fix_ev)
            accumulated_events.append(fix_ev)
            completed = False
            # Each verify-fix-reverify cycle is a fresh attempt at the task,
            # so it gets its own self-correction allowance (M14) rather than
            # inheriting whatever was left over from the first cycle's tool
            # errors. Without this, a first attempt that burns its whole
            # self_correction_max on tool retries would leave later
            # verify-fix cycles with zero self-correction budget even though
            # they have their own verify_retry_max budget to work with.
            correction_budget = self.self_correction_max

        # Resolve final status. Terminal-resource failures win; otherwise a
        # completed run is "success" unless verification failed, a detected
        # loop is "loop_detected", and an incomplete run (hit the step cap)
        # is "max_steps".
        if terminal_status is not None:
            status: RunStatus = terminal_status
        elif loop_detected:
            status = "loop_detected"
        elif not completed and step >= self.max_steps:
            status = "max_steps"
            extension_note = (
                f" (extended from {self.max_steps} via the adaptive step budget)"
                if self._step_extension_granted
                else ""
            )
            err_ev = ErrorEvent(
                session_id=sess_id,
                error_type="MaxStepsExceeded",
                message=f"Agent loop reached maximum step count ({step}){extension_note}",
            )
            await self._emit(err_ev)
        elif verify_failed:
            status = "verification_failed"
        else:
            status = "success"

        is_success = status == "success"

        # 6. Episodic Memory Recording
        await self._record_trajectory(
            sess_id,
            intent_res.intent_type,
            active_model,
            step,
            is_success,
            prompt,
        )

        # 7. stop hooks: notification that the run finished. A blocking outcome
        # (exit 2) is logged but, in this MVP, cannot retroactively resume the
        # loop — the run has already terminated. The contract is honored for
        # pre_tool/pre_compact, where a veto is actionable mid-flight.
        if self.hooks.enabled:
            stop_outcomes = await self.hooks.run(
                HookLifecycle.STOP,
                {
                    "session_id": sess_id,
                    "status": status,
                    "steps": step,
                    "success": is_success,
                },
            )
            if any(o.blocked for o in stop_outcomes):
                logger.info("stop_hook_blocked", session_id=sess_id, status=status)

        root_span.set_attributes({"run.status": status, "run.steps": step})

        return RunResult(
            session_id=sess_id,
            status=status,
            success=is_success,
            final_text=final_text,
            steps=step,
            model=active_model,
            intent=intent_res.intent_type,
            feedback=feedback,
            error=terminal_error,
        )

    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
        events_history: list[BaseEvent] | None = None,
    ) -> str:
        """Run orchestrated agent execution loop (non-streaming).

        Pipeline: Intent → Route → Plan → Act → Verify → Memory.

        Returns the model's final answer text. Terminal-resource failures
        (budget / timeout / context exhaustion) are re-raised as exceptions for
        backwards compatibility; use ``run_result`` for a structured outcome
        that never raises.
        """
        result = await self.run_result(
            prompt=prompt,
            session_id=session_id,
            events_history=events_history,
            streaming=False,
        )
        self._raise_terminal(result)
        return result.final_text

    async def run_streaming(
        self,
        prompt: str,
        session_id: str | None = None,
        events_history: list[BaseEvent] | None = None,
    ) -> str:
        """Run agent loop with streaming token output.

        Emits StreamDeltaEvent for each token chunk during generation.
        Returns the final accumulated text. Terminal failures are re-raised.
        """
        result = await self.run_result(
            prompt=prompt,
            session_id=session_id,
            events_history=events_history,
            streaming=True,
        )
        self._raise_terminal(result)
        return result.final_text

    async def run_result(
        self,
        prompt: str,
        session_id: str | None = None,
        events_history: list[BaseEvent] | None = None,
        streaming: bool = False,
    ) -> RunResult:
        """Run the agent loop and return a structured ``RunResult``.

        Unlike ``run`` / ``run_streaming``, this never raises for
        terminal-resource failures — they are reported via ``status`` and
        ``error``. Unexpected non-domain exceptions still propagate.
        """
        return await self._run_pipeline(
            prompt=prompt,
            session_id=session_id,
            events_history=events_history,
            streaming=streaming,
        )

    def list_rewind_points(self, events_history: list[BaseEvent]) -> list[RewindPoint]:
        """Enumerate the points a prior trajectory can be rewound to (M14).

        Thin pass-through to :func:`nullain.agent.checkpoint.list_rewind_points`
        exposed on ``AgentLoop`` so callers driving a rewind don't need a
        separate import for the common case.
        """
        return list_rewind_points(events_history)

    async def rewind_and_run(
        self,
        events_history: list[BaseEvent],
        to_step: int,
        prompt: str,
        session_id: str | None = None,
        streaming: bool = False,
    ) -> RunResult:
        """Truncate a trajectory back to ``to_step`` and resume with a new prompt (M14).

        Equivalent to ``run_result(prompt, events_history=rewind_events(events_history,
        to_step))`` — provided as one call since rewind-then-resume is always
        done together in practice: a caller (a user saying "try that
        differently", or a scripted retry policy) picks a step via
        :meth:`list_rewind_points`, then wants the truncated trajectory acted
        on immediately rather than juggling the intermediate event list
        itself.

        Args:
            events_history: The full prior trajectory to rewind.
            to_step: 1-indexed step number to rewind to (see
                :func:`nullain.agent.checkpoint.rewind_events`).
            prompt: The new instruction to run from the rewound context —
                typically a correction or a different approach than what the
                original step N attempted.
            session_id: Session id for the resumed run. When None, a fresh id
                is generated (matching ``run_result``'s default), which is
                usually not what a rewind wants — pass the original session's
                id to keep the resumed run in the same session's trajectory.
            streaming: Whether to stream the resumed run.

        Returns:
            The resumed run's ``RunResult``.

        Raises:
            ValueError: if ``to_step`` is not a valid step number in
                ``events_history`` (propagated from ``rewind_events``).
        """
        truncated = rewind_events(events_history, to_step)
        return await self.run_result(
            prompt=prompt,
            session_id=session_id,
            events_history=truncated,
            streaming=streaming,
        )

    @staticmethod
    def _raise_terminal(result: RunResult) -> None:
        """Re-raise the domain exception corresponding to a terminal status.

        Keeps ``run`` / ``run_streaming`` backwards-compatible with callers
        that catch ``BudgetExceededError`` / ``ContextWindowExhaustedError`` /
        ``NullainError`` (timeout). No-op for non-terminal statuses.
        """
        if result.status == "budget":
            raise BudgetExceededError(result.error or "Token budget exceeded")
        if result.status == "context_exhausted":
            raise ContextWindowExhaustedError(result.error or "Context window exhausted")
        if result.status == "timeout":
            raise NullainError(result.error or "Agent loop timed out")
        if result.status == "error":
            raise ProviderError(result.error or "Agent run failed with a provider error")
        if result.status == "cancelled":
            # Legacy run()/run_streaming() callers still observe cancellation
            # as an exception; run_result() returns the structured status.
            raise get_cancelled_exc_class()(result.error or "Agent run cancelled")

    async def spawn(
        self,
        prompt: str,
        tools: ToolRegistry | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        authority: Authority | None = None,
        isolation: Literal["worktree", None] = None,
    ) -> str:
        """Run a sub-agent with fresh context and return its final answer text.

        The sub-agent shares this loop's provider and workspace but runs with
        an isolated event bus and a fresh context window (no inherited
        conversation history). Use ``spawn`` for focused subtasks: the parent
        receives only the sub-agent's final text, so a long-running
        investigation does not blow out the parent's context.

        Authority-intersection law (P4.24): when ``authority`` is delegated,
        the child's effective authority is the meet of four factors::

            effective = parent_authority ∧ delegation ∧ child_def ∧ policy

        enforced at the registry execution gate. The child can exercise a
        capability only if every factor grants it; any single denial removes
        it. Delegation is clamped to the parent's own authority (a parent can
        never delegate what it does not hold). When ``authority`` is None the
        sub-run is unrestricted and backward compatible — no gate is applied.

        A subagent whose effective authority lacks the SPAWN capability cannot
        itself spawn (``NullainError``); this caps delegation depth.

        Worktree isolation (M11.4): when ``isolation="worktree"`` the child runs
        in a detached ``git worktree`` (``git worktree add --detach``) with a
        tool registry re-rooted at the worktree via ``self.tool_factory``. The
        child edits an isolated checkout; changed files are integrated back into
        the parent workspace afterwards, and the worktree is removed in
        ``finally`` (including on failure/cancellation). Requires
        ``self.tool_factory`` to be set, else ``NullainError``.

        Args:
            prompt: The subtask prompt for the sub-agent.
            tools: Scoped tool registry for the sub-agent. When None, the
                sub-agent inherits the parent's tool registry.
            model: Explicit model for the sub-agent (bypasses routing). When
                None, the sub-agent routes by intent like a normal run.
            max_steps: Step cap for the sub-agent. When None, inherits the
                parent's ``max_steps``.
            authority: Delegated authority for the sub-agent. When None, the
                sub-agent inherits an unrestricted view (backward compatible).
                When provided, it is intersected with the parent's authority,
                the child's tool-defined surface, and the permission policy.
            isolation: ``"worktree"`` runs the child in a detached git worktree
                (M11.4); ``None`` (default) runs in place.

        Returns:
            The sub-agent's final answer text.

        Note:
            This call is synchronous-in-place: the parent blocks until the
            sub-agent finishes. Use :meth:`spawn_many` to run several
            sub-agents concurrently (M13).
        """
        if isolation == "worktree":
            return await self._spawn_worktree(prompt, tools, model, max_steps, authority)
        child_authority, child_tools = self._child_registry(tools or self.tools, authority)
        child = AgentLoop(
            provider=self.provider,
            tools=child_tools,
            model=model,
            workspace_root=self.workspace_root,
            max_steps=max_steps or self.max_steps,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            loop_detection_threshold=self.loop_detection_threshold,
            context_manager=ContextManager(),
            prompt_assembler=self.prompt_assembler,
            episodic_memory=None,  # isolated; the parent records the trajectory
            # Fresh event bus: sub-agent internal events do not pollute the
            # parent's trajectory. The parent only sees the returned text.
            event_bus=EventBus(),
            authority=child_authority,
        )
        result = await child.run_result(prompt=prompt)
        return result.final_text

    async def spawn_many(self, tasks: "list[SpawnTask]") -> "list[SpawnOutcome]":
        """Run several sub-agents concurrently and collect their outcomes (M13).

        Each task runs via :meth:`spawn` inside its own ``asyncio`` coroutine;
        every sub-agent already gets its own event bus, context window, and
        (for ``isolation="worktree"``) its own detached git worktree, so
        concurrent sub-agents do not share mutable state with each other.
        A task's failure is captured in its own ``SpawnOutcome`` rather than
        cancelling siblings — one bad subtask should not sink an otherwise
        successful fan-out.

        Requires this loop to carry the SPAWN capability (same rule as
        ``spawn`` — enforced per-task by ``_child_registry``).

        Args:
            tasks: The sub-agent tasks to run, in the order results are
                returned. Each entry pairs a prompt with :meth:`spawn`'s
                per-task options.

        Returns:
            One :class:`SpawnOutcome` per task, in the same order as ``tasks``.
            A task whose ``isolation="worktree"`` run raises is reported with
            ``error`` set and ``text=""`` rather than propagating, so sibling
            tasks are unaffected; a non-worktree run that raises for a reason
            other than the domain errors ``spawn`` itself may raise (e.g. a
            programming error in a hook) still propagates, matching ``spawn``'s
            own behavior for the same failure.
        """

        async def _run_one(task: SpawnTask) -> SpawnOutcome:
            try:
                text = await self.spawn(
                    prompt=task.prompt,
                    tools=task.tools,
                    model=task.model,
                    max_steps=task.max_steps,
                    authority=task.authority,
                    isolation=task.isolation,
                )
                return SpawnOutcome(text=text, error=None)
            except NullainError as err:
                return SpawnOutcome(text="", error=str(err))

        return list(await asyncio.gather(*(_run_one(t) for t in tasks)))

    def _child_registry(
        self, base_registry: ToolRegistry, authority: Authority | None
    ) -> tuple[Authority | None, ToolRegistry]:
        """Apply the P4.24 authority-intersection law to a child registry.

        Returns ``(child_authority, child_tools)``. When ``authority`` is None
        the child is unrestricted (backward compatible). Otherwise the effective
        authority is the meet of parent ∧ delegation ∧ child_def ∧ policy, and
        the returned registry is scoped to it. ``base_registry`` may be the
        parent's registry or a worktree-rooted registry from ``tool_factory``.
        """
        if authority is None:
            return None, base_registry
        # SPAWN capability: a subagent without it cannot spawn further.
        if self.authority is not None and not self.authority.can_spawn:
            raise NullainError("subagent lacks the SPAWN capability and cannot spawn a child")
        parent_auth = self.authority or Authority.unrestricted()
        # Clamp delegation to the parent's holdings — a parent cannot delegate
        # authority it does not itself hold.
        delegation = authority.meet(parent_auth)
        child_def = base_registry.capability_surface()
        base_policy = base_registry.permission_policy
        policy_auth = (
            Authority.from_policy(base_policy)
            if base_policy is not None
            else Authority.unrestricted()
        )
        effective = delegation.meet(child_def).meet(policy_auth)
        child_tools = base_registry.scoped(
            authority=effective,
            permission_policy=_project_authority_policy(base_policy, effective),
        )
        return effective, child_tools

    async def _spawn_worktree(
        self,
        prompt: str,
        tools: ToolRegistry | None,
        model: str | None,
        max_steps: int | None,
        authority: Authority | None,
    ) -> str:
        """Run a sub-agent in a detached git worktree (M11.4).

        Creates ``git worktree add --detach <tmpdir>`` (detached HEAD, no branch
        pollution), re-roots the child's tool registry at the worktree via
        ``self.tool_factory``, runs the child with ``workspace_root=worktree_dir``,
        integrates changed files back into the parent workspace, and guarantees
        worktree cleanup in ``finally`` (including on failure/cancellation).
        """
        if self.tool_factory is None:
            raise NullainError("worktree isolation requires a tool_factory")
        tmpdir = Path(tempfile.mkdtemp(prefix="nullain-wt-"))
        worktree_dir = tmpdir / "worktree"
        try:
            await self._run_git(
                ["worktree", "add", "--detach", str(worktree_dir)],
                cwd=self.workspace_root,
            )
            base_registry = self.tool_factory(worktree_dir)
            child_authority, child_tools = self._child_registry(base_registry, authority)
            child = AgentLoop(
                provider=self.provider,
                tools=child_tools,
                model=model,
                workspace_root=worktree_dir,
                max_steps=max_steps or self.max_steps,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                loop_detection_threshold=self.loop_detection_threshold,
                context_manager=ContextManager(),
                # Re-root the prompt assembler at the worktree so SOUL.md /
                # AGENTS.md are read from the isolated checkout, not the parent.
                prompt_assembler=PromptAssembler(workspace_root=worktree_dir),
                episodic_memory=None,  # isolated; the parent records the trajectory
                event_bus=EventBus(),
                authority=child_authority,
            )
            result = await child.run_result(prompt=prompt)
            await self._integrate_worktree(worktree_dir)
            return result.final_text
        finally:
            # Guaranteed cleanup: remove the worktree (force, in case the child
            # left untracked files) and the temp dir, even on failure/cancel.
            with contextlib.suppress(Exception):
                await self._run_git(
                    ["worktree", "remove", "--force", str(worktree_dir)],
                    cwd=self.workspace_root,
                )
            shutil.rmtree(tmpdir, ignore_errors=True)

    async def _integrate_worktree(self, worktree_dir: Path) -> None:
        """Copy files the child changed in the worktree back into the workspace.

        The child edits the worktree's working tree without committing. After
        the run, ``git status --porcelain`` lists the changed paths; each is
        copied from the worktree back to the parent workspace so the child's
        edits land in the main checkout. Deleted files are removed from the
        parent too.
        """
        status = await self._run_git(["status", "--porcelain"], cwd=worktree_dir)
        for line in status.splitlines():
            if not line.strip():
                continue
            # Porcelain v1: "XY <path>", with renames as "XY old -> new".
            rel = line[3:].strip()
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[1]
            src = worktree_dir / rel
            dst = self.workspace_root / rel
            if line[0] == "D" or line[1] == "D":
                dst.unlink(missing_ok=True)
            elif src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    async def _run_git(self, args: list[str], cwd: Path) -> str:
        """Run a git command (explicit argv, no shell) and return stdout.

        Raises :class:`~nullain.errors.NullainError` on a non-zero exit so
        worktree setup/cleanup failures surface rather than silently leaving a
        stray worktree behind.
        """
        code, output = await execute_subprocess(["git", *args], cwd=cwd)
        if code != 0:
            raise NullainError(f"git {' '.join(args)} failed (exit {code}): {output.strip()}")
        return output


__all__ = ["AgentLoop", "SpawnOutcome", "SpawnTask"]
