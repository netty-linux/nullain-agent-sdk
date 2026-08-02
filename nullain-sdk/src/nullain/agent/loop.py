"""Nullain Agent SDK — AgentLoop ReAct and Plan/Act Orchestrated Execution Engine."""

import json as _json
import uuid
from pathlib import Path

from pydantic import ValidationError

from nullain.agent.spec import SpecValidator, TaskSpec
from nullain.context.assembler import PromptAssembler
from nullain.context.manager import ContextManager
from nullain.errors import BudgetExceededError, NullainError, ToolError
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
from nullain.telemetry import get_logger
from nullain.tools import ToolRegistry

logger = get_logger(__name__)


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

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        sess_id: str,
        accumulated_events: list[BaseEvent],
        correction_budget: int,
    ) -> tuple[str, int]:
        """Execute tool calls with self-correction injection on errors.

        Returns:
            Tuple of (last_output, remaining_correction_budget).
        """
        last_output = ""
        for tc in tool_calls:
            tc_ev = ToolCallEvent(
                session_id=sess_id,
                call_id=tc.id,
                tool_name=tc.name,
                arguments=tc.arguments,
            )
            await self._emit(tc_ev)
            accumulated_events.append(tc_ev)

            try:
                res_output = await self.tools.execute(tc.name, tc.arguments)
                is_err = res_output.startswith("Error:") or res_output.startswith("Error ")
            except ToolError as err:
                logger.warning(
                    "tool_execution_failed",
                    session_id=sess_id,
                    tool=tc.name,
                    error=str(err),
                )
                res_output = f"Tool Execution Error ({tc.name}): {err}"
                is_err = True
                err_ev = ErrorEvent(
                    session_id=sess_id,
                    error_type="ToolExecutionError",
                    message=res_output,
                )
                await self._emit(err_ev)
            except Exception as err:
                logger.error(
                    "unexpected_tool_error",
                    session_id=sess_id,
                    tool=tc.name,
                    error=str(err),
                )
                res_output = f"Unexpected Error ({tc.name}): {err}"
                is_err = True
                err_ev = ErrorEvent(
                    session_id=sess_id,
                    error_type="UnexpectedToolError",
                    message=res_output,
                )
                await self._emit(err_ev)

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
        """Fold events into messages with system prompt and centrifugation."""
        state = Conversation.fold(sess_id, accumulated_events)
        messages = state.messages
        if not messages or messages[0].role != "system":
            messages.insert(0, ChatMessage(role="system", content=system_prompt))
        return self.context_manager.reinject_instructions(messages, step)

    async def _check_budget_and_compact(
        self,
        sess_id: str,
        accumulated_events: list[BaseEvent],
        total_tokens: int,
        active_spec: TaskSpec | None,
        system_prompt: str,
    ) -> None:
        """Check token budget and compact context if needed."""
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
        if not messages or messages[0].role != "system":
            messages.insert(0, ChatMessage(role="system", content=system_prompt))
        ctx_tokens = self.context_manager.estimate_context_tokens(messages)
        if self.context_manager.should_compact(ctx_tokens):
            compact_ev = self.context_manager.compact(
                session_id=sess_id,
                events=accumulated_events,
                active_spec=active_spec,
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
        """Generate response chunk either via streaming or direct generate."""
        if streaming:
            full_text = ""
            all_tool_calls: list[ToolCall] = []
            usage: TokenUsage | None = None
            try:
                async for chunk in self.provider.stream(req):
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
            return response.delta_text or "", tool_calls, response.usage

    async def _run_pipeline(
        self,
        prompt: str,
        session_id: str | None = None,
        events_history: list[BaseEvent] | None = None,
        streaming: bool = False,
    ) -> str:
        """Run orchestrated agent execution pipeline.

        Pipeline: Intent → Route → Plan → Act → Verify → Memory.
        """
        sess_id = session_id or str(uuid.uuid4())
        start_time = self.clock.now()
        accumulated_events: list[BaseEvent] = list(events_history or [])
        correction_budget = self.self_correction_max

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

            # Execute tools with self-correction
            last_output, correction_budget = await self._execute_tools(
                tool_calls,
                sess_id,
                accumulated_events,
                correction_budget,
            )

        # 5. Verify Phase
        is_success = True
        if active_spec:
            verified, feedback = await self.spec_validator.verify(
                active_spec,
                last_output,
                workspace_root=self.workspace_root,
                tools=self.tools,
            )
            is_success = verified
            verify_ev = SpecVerifiedEvent(
                session_id=sess_id,
                spec_id=active_spec.spec_id,
                success=verified,
                feedback=feedback,
            )
            await self._emit(verify_ev)
            accumulated_events.append(verify_ev)

        # Only flag MaxStepsExceeded when the loop exited by hitting the step
        # cap WITHOUT producing a final answer. A final answer returned on the
        # last allowed step (completed=True, step==max_steps) is a success.
        if not completed and step >= self.max_steps:
            is_success = False
            err_ev = ErrorEvent(
                session_id=sess_id,
                error_type="MaxStepsExceeded",
                message=f"Agent loop reached maximum step count ({self.max_steps})",
            )
            await self._emit(err_ev)

        # 6. Episodic Memory Recording
        await self._record_trajectory(
            sess_id,
            intent_res.intent_type,
            active_model,
            step,
            is_success,
            prompt,
        )

        return final_text

    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
        events_history: list[BaseEvent] | None = None,
    ) -> str:
        """Run orchestrated agent execution loop (non-streaming).

        Pipeline: Intent → Route → Plan → Act → Verify → Memory.
        """
        return await self._run_pipeline(
            prompt=prompt,
            session_id=session_id,
            events_history=events_history,
            streaming=False,
        )

    async def run_streaming(
        self,
        prompt: str,
        session_id: str | None = None,
        events_history: list[BaseEvent] | None = None,
    ) -> str:
        """Run agent loop with streaming token output.

        Emits StreamDeltaEvent for each token chunk during generation.
        Returns the final accumulated text.
        """
        return await self._run_pipeline(
            prompt=prompt,
            session_id=session_id,
            events_history=events_history,
            streaming=True,
        )


__all__ = ["AgentLoop"]
