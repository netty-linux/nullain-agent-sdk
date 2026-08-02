"""Nullain Agent SDK — AgentLoop ReAct and Plan/Act Orchestrated Execution Engine."""

import asyncio
import json as _json
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nullain.agent.result import RunResult, RunStatus
from nullain.agent.spec import BASH_NONZERO_PREFIX, SpecValidator, TaskSpec
from nullain.context.assembler import PromptAssembler
from nullain.context.manager import ContextManager
from nullain.errors import (
    BudgetExceededError,
    ContextWindowExhaustedError,
    NullainError,
    ToolError,
    ToolPermissionError,
)
from nullain.events import (
    BaseEvent,
    Conversation,
    ErrorEvent,
    EventBus,
    EventStore,
    ModelResponseEvent,
    SpecCreatedEvent,
    SpecVerifiedEvent,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from nullain.llm import (
    ChatMessage,
    CompletionChunk,
    CompletionRequest,
    LLMProvider,
    TokenUsage,
    ToolCall,
)
from nullain.memory import EpisodicMemory, TrajectoryRecord
from nullain.ports.clock import Clock, SystemClock
from nullain.router import Complexity, IntentParser, ModelRouter
from nullain.telemetry import get_cost_tracker, get_logger
from nullain.telemetry import span as telemetry_span
from nullain.tools import ToolRegistry

logger = get_logger(__name__)

# Output prefixes that mark a tool execution as failed. Mirrors the contracts
# of the bundled tools: filesystem returns "Error: ...", bash prefixes non-zero
# exits with BASH_NONZERO_PREFIX ("Command exit code:"), and git_commit emits
# "Git commit failed: ...". Detection must cover all of these so the Act loop
# flags genuine failures instead of silently swallowing them as success.
ERROR_OUTPUT_PREFIXES: tuple[str, ...] = (
    "Error:",
    "Error ",
    BASH_NONZERO_PREFIX,
    "Git commit failed:",
)


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
        workspace_root: str | Path = ".",
        max_steps: int = 25,
        max_tokens: int | None = 100_000,
        timeout: float = 300.0,
        self_correction_max: int = 3,
        max_compaction_attempts: int = 3,
        loop_detection_threshold: int = 3,
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
            workspace_root: Workspace root directory path.
            max_steps: Maximum ReAct loop iterations.
            max_tokens: Token budget ceiling.
            timeout: Execution timeout in seconds.
            self_correction_max: Max self-correction retries per session.
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.provider = provider
        self.tools = tools
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
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.self_correction_max = self_correction_max
        self.max_compaction_attempts = max_compaction_attempts
        # Consecutive identical step signatures (by tool_name + arguments) at
        # which the loop is considered stuck and is broken out of. Mirrors the
        # loop-detection Gemini CLI uses to escape thrash.
        self.loop_detection_threshold = loop_detection_threshold
        self._compaction_attempts = 0

    async def _emit(self, event: BaseEvent) -> None:
        """Emit event to bus and persist to store if configured."""
        await self.event_bus.publish(event)
        if self.event_store is not None:
            await self.event_store.append(event)

    async def _generate_spec(self, prompt: str, model: str, system_prompt: str) -> TaskSpec:
        """Ask the LLM to generate a structured TaskSpec for the task.

        Falls back to a minimal generic spec if JSON parsing fails.
        """
        spec_instruction = (
            "Analyze the following task and generate a structured plan.\n"
            "Respond ONLY with a JSON object matching this schema:\n"
            '{"objective": "...", "steps": ["step1", ...], '
            '"target_files": ["file1.py", ...], '
            '"acceptance_criteria": ["criterion1", ...]}\n\n'
            f"Task: {prompt}"
        )
        spec_messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=spec_instruction),
        ]
        req = CompletionRequest(
            model=model,
            messages=spec_messages,
            stream=False,
        )
        response = await self.provider.generate(req)
        text = (response.delta_text or "").strip()

        # Extract JSON from markdown code fence if present
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
    ) -> tuple[str, bool, str | None]:
        """Execute one tool call, returning (output, is_error, error_type).

        Exceptions are normalized into error output strings so callers can
        dispatch a batch concurrently without ``gather`` short-circuiting.
        ``error_type`` is set when the failure warranted an ErrorEvent.
        Emits a ``tool`` telemetry span tagged with the outcome, including
        ``tool.blocked`` when execution was denied by the permission policy.
        """
        with telemetry_span("tool", **{"tool.name": tc.name}) as tool_span:
            try:
                res_output = await self.tools.execute(tc.name, tc.arguments)
                is_err = any(res_output.startswith(prefix) for prefix in ERROR_OUTPUT_PREFIXES)
                tool_span.set_attributes({"tool.is_error": is_err, "tool.blocked": False})
                return res_output, is_err, None
            except ToolPermissionError as err:
                logger.warning(
                    "tool_permission_denied",
                    session_id=sess_id,
                    tool=tc.name,
                    error=str(err),
                )
                tool_span.set_attributes({"tool.is_error": True, "tool.blocked": True})
                return f"Tool Execution Error ({tc.name}): {err}", True, "ToolExecutionError"
            except ToolError as err:
                logger.warning(
                    "tool_execution_failed",
                    session_id=sess_id,
                    tool=tc.name,
                    error=str(err),
                )
                tool_span.set_attributes({"tool.is_error": True, "tool.blocked": False})
                return f"Tool Execution Error ({tc.name}): {err}", True, "ToolExecutionError"
            except Exception as err:
                logger.error(
                    "unexpected_tool_error",
                    session_id=sess_id,
                    tool=tc.name,
                    error=str(err),
                )
                tool_span.set_attributes({"tool.is_error": True, "tool.blocked": False})
                return f"Unexpected Error ({tc.name}): {err}", True, "UnexpectedToolError"

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        sess_id: str,
        accumulated_events: list[BaseEvent],
        correction_budget: int,
    ) -> tuple[str, int]:
        """Execute tool calls with self-correction injection on errors.

        Dispatch policy: when every call in the batch targets a read-only
        tool (and there is more than one), the calls run concurrently via
        ``asyncio.gather``. Mixed or side-effecting batches run sequentially
        to preserve ordering and avoid races on shared filesystem state.

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

        all_read_only = len(tool_calls) > 1 and all(
            self.tools.is_read_only(tc.name) for tc in tool_calls
        )
        if all_read_only:
            outcomes = await asyncio.gather(
                *(self._run_tool_call(tc, sess_id) for tc in tool_calls)
            )
        else:
            outcomes = [await self._run_tool_call(tc, sess_id) for tc in tool_calls]

        last_output = ""
        for tc, (res_output, is_err, error_type) in zip(tool_calls, outcomes, strict=True):
            if error_type is not None:
                err_ev = ErrorEvent(
                    session_id=sess_id,
                    error_type=error_type,
                    message=res_output,
                )
                await self._emit(err_ev)
                accumulated_events.append(err_ev)

            res_ev = ToolResultEvent(
                session_id=sess_id,
                call_id=tc.id,
                tool_name=tc.name,
                output=res_output,
                is_error=is_err,
            )
            await self._emit(res_ev)
            accumulated_events.append(res_ev)
            last_output = res_output

            # Self-correction: inject reflection on error
            if is_err and correction_budget > 0:
                correction_budget -= 1
                reflection = UserMessageEvent(
                    session_id=sess_id,
                    content=(
                        f"[SELF-CORRECTION] Tool '{tc.name}' "
                        f"failed: {res_output}\n"
                        "Analyze this error. Was the input "
                        "malformed? Is there an alternative "
                        "approach? Try a different strategy."
                    ),
                )
                await self._emit(reflection)
                accumulated_events.append(reflection)

        return last_output, correction_budget

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
        state = Conversation.fold(sess_id, accumulated_events)
        messages = state.messages
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

        # Compact based on actual context size, not cumulative
        state = Conversation.fold(sess_id, accumulated_events)
        messages = state.messages
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
                all_tool_calls: list[ToolCall] = []
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
                            all_tool_calls.extend(chunk.tool_calls)
                        if chunk.usage:
                            usage = chunk.usage
                    self.router.circuit_breaker.record_success(active_model)
                    latency_ms = (self.clock.now() - start) * 1000.0
                    self._record_llm_telemetry(
                        llm_span, tracker, active_model, sess_id, usage, ttft, latency_ms
                    )
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
            return await self._run_pipeline_body(
                prompt, session_id, events_history, streaming, root_span
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

        # 1. Intent Parsing & Model Routing
        intent_res = self.intent_parser.parse(prompt)
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

        # Assemble Layered System Prompt
        system_prompt = self.prompt_assembler.assemble(
            skills_summary=None,
            episodic_memory=few_shot_str,
        )

        # 3. Plan Phase — LLM generates spec for MEDIUM/HIGH tasks
        active_spec: TaskSpec | None = None
        if intent_res.complexity in (
            Complexity.MEDIUM,
            Complexity.HIGH,
        ):
            active_spec = await self._generate_spec(prompt, active_model, system_prompt)
            self.spec_validator.validate_spec(active_spec)
            spec_ev = SpecCreatedEvent(
                session_id=sess_id,
                spec_id=active_spec.spec_id,
                title=active_spec.objective,
                steps=tuple(active_spec.steps),
            )
            await self._emit(spec_ev)
            accumulated_events.append(spec_ev)

        # 4. Act Phase (ReAct Loop)
        step = 0
        total_tokens = 0
        final_text = ""
        last_output = ""
        completed = False  # True only when the model returned a final answer (no tool calls)
        terminal_status: RunStatus | None = None
        terminal_error: str | None = None
        loop_detected = False
        last_step_signature: str | None = None
        repeat_count = 0

        try:
            while step < self.max_steps:
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
        except BudgetExceededError as err:
            terminal_status = "budget"
            terminal_error = str(err)
        except ContextWindowExhaustedError as err:
            terminal_status = "context_exhausted"
            terminal_error = str(err)
        except NullainError as err:
            # Timeout (raised as a plain NullainError) and any other domain
            # failure surfaced from the act phase.
            terminal_status = "timeout"
            terminal_error = str(err)

        # 5. Verify Phase — only when the loop produced a final answer
        feedback: str | None = None
        verify_failed = False
        if completed and active_spec:
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
            err_ev = ErrorEvent(
                session_id=sess_id,
                error_type="MaxStepsExceeded",
                message=f"Agent loop reached maximum step count ({self.max_steps})",
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

    async def spawn(
        self,
        prompt: str,
        tools: ToolRegistry | None = None,
        model: str | None = None,
        max_steps: int | None = None,
    ) -> str:
        """Run a sub-agent with fresh context and return its final answer text.

        The sub-agent shares this loop's provider and workspace but runs with
        an isolated event bus and a fresh context window (no inherited
        conversation history). Use ``spawn`` for focused subtasks: the parent
        receives only the sub-agent's final text, so a long-running
        investigation does not blow out the parent's context.

        Args:
            prompt: The subtask prompt for the sub-agent.
            tools: Scoped tool registry for the sub-agent. When None, the
                sub-agent inherits the parent's tool registry.
            model: Explicit model for the sub-agent (bypasses routing). When
                None, the sub-agent routes by intent like a normal run.
            max_steps: Step cap for the sub-agent. When None, inherits the
                parent's ``max_steps``.

        Returns:
            The sub-agent's final answer text.

        Note:
            v1 is synchronous-in-place: the parent blocks until the sub-agent
            finishes. Background worktrees / concurrent sub-agents are deferred
            to a later milestone.
        """
        child = AgentLoop(
            provider=self.provider,
            tools=tools or self.tools,
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
        )
        result = await child.run_result(prompt=prompt)
        return result.final_text


__all__ = ["AgentLoop"]
