"""Nullain Agent SDK — AgentLoop ReAct and Plan/Act Orchestrated Execution Engine."""

import time
import uuid
from pathlib import Path

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
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from nullain.llm import (
    ChatMessage,
    CompletionChunk,
    CompletionRequest,
    LLMProvider,
    OllamaCloudProvider,
)
from nullain.memory import EpisodicMemory, TrajectoryRecord
from nullain.router import Complexity, IntentParser, ModelRouter
from nullain.telemetry import get_logger
from nullain.tools import ToolRegistry

logger = get_logger(__name__)


class AgentLoop:
    """Orchestrated Agent Execution Engine integrating Plan/Act, Routing, Context & Memory."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        tools: ToolRegistry | None = None,
        event_bus: EventBus | None = None,
        event_store: EventStore | None = None,
        model: str | None = None,
        router: ModelRouter | None = None,
        intent_parser: IntentParser | None = None,
        context_manager: ContextManager | None = None,
        spec_validator: SpecValidator | None = None,
        prompt_assembler: PromptAssembler | None = None,
        episodic_memory: EpisodicMemory | None = None,
        workspace_root: str | Path = ".",
        max_steps: int = 25,
        max_tokens: int | None = 100_000,
        timeout: float = 300.0,
    ) -> None:
        """Initialize AgentLoop with integrated engine collaborators."""
        self.workspace_root = Path(workspace_root).resolve()
        self.provider = provider or OllamaCloudProvider()
        self.tools = tools or ToolRegistry()
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
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def _emit(self, event: BaseEvent) -> None:
        """Emit event to event bus and persist to event store if configured."""
        await self.event_bus.publish(event)
        if self.event_store is not None:
            await self.event_store.append(event)

    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
        events_history: list[BaseEvent] | None = None,
    ) -> str:
        """Run orchestrated agent execution loop.

        Pipeline: Intent Parsing -> Routing -> Plan -> Act -> Verify -> Memory.
        """
        sess_id = session_id or str(uuid.uuid4())
        start_time = time.time()
        accumulated_events: list[BaseEvent] = list(events_history or [])

        # 1. Intent Parsing & Model Routing
        intent_res = self.intent_parser.parse(prompt)
        active_model = self.explicit_model or self.router.route_intent(intent_res)
        logger.info(
            "task_intent_classified",
            session_id=sess_id,
            intent=intent_res.intent_type,
            complexity=intent_res.complexity,
            model=active_model,
        )

        # Emit initial UserMessageEvent
        user_ev = UserMessageEvent(session_id=sess_id, content=prompt)
        await self._emit(user_ev)
        accumulated_events.append(user_ev)

        # 2. Episodic Memory Retrieval for Few-Shot Examples
        few_shot_str: str | None = None
        if self.episodic_memory is not None:
            try:
                past_examples = await self.episodic_memory.get_relevant_examples(
                    intent_res.intent_type, limit=2
                )
                if past_examples:
                    ex_lines = [
                        f"- Objective: {ex.objective} (Steps: {ex.steps_count})"
                        for ex in past_examples
                    ]
                    few_shot_str = "\n".join(ex_lines)
            except Exception as err:
                logger.warning("episodic_memory_lookup_failed", session_id=sess_id, error=str(err))

        # Assemble Layered System Prompt
        system_prompt = self.prompt_assembler.assemble(
            skills_summary=None,
            episodic_memory=few_shot_str,
        )

        # 3. Plan Phase (Spec Generation & Validation)
        active_spec: TaskSpec | None = None
        if intent_res.complexity in (Complexity.MEDIUM, Complexity.HIGH):
            active_spec = TaskSpec(
                objective=prompt,
                steps=[
                    "Analyze codebase and requirements",
                    "Execute necessary edits or commands",
                    "Verify result",
                ],
                acceptance_criteria=["Execution completes with no unhandled errors"],
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

        # 4. Act Phase (ReAct Loop)
        step = 0
        total_tokens = 0
        final_text = ""
        last_execution_output = ""

        while step < self.max_steps:
            # Termination Check: Timeout
            if (time.time() - start_time) > self.timeout:
                err_ev = ErrorEvent(
                    session_id=sess_id,
                    error_type="TimeoutError",
                    message=f"Agent loop timed out after {self.timeout} seconds",
                )
                await self._emit(err_ev)
                raise NullainError(f"Agent loop timed out after {self.timeout} seconds")

            # Termination Check: Token Budget
            if self.max_tokens is not None and total_tokens >= self.max_tokens:
                err_ev = ErrorEvent(
                    session_id=sess_id,
                    error_type="BudgetExceededError",
                    message=f"Token budget of {self.max_tokens} exceeded",
                )
                await self._emit(err_ev)
                raise BudgetExceededError(f"Token budget of {self.max_tokens} exceeded")

            # Context Window Compaction Check
            if self.context_manager.should_compact(total_tokens):
                compact_ev = self.context_manager.compact(
                    session_id=sess_id,
                    events=accumulated_events,
                    active_spec=active_spec,
                )
                await self._emit(compact_ev)
                accumulated_events.append(compact_ev)

            step += 1
            state = Conversation.fold(sess_id, accumulated_events)
            messages = state.messages

            # Prepend system prompt if not present
            if not messages or messages[0].role != "system":
                messages.insert(0, ChatMessage(role="system", content=system_prompt))

            # Apply instruction centrifugation
            messages = self.context_manager.reinject_instructions(messages, step)

            tool_specs = self.tools.list_specs()
            req = CompletionRequest(
                model=active_model,
                messages=messages,
                tools=tool_specs if tool_specs else None,
                stream=False,
            )

            try:
                response_chunk: CompletionChunk = await self.provider.generate(req)
                self.router.circuit_breaker.record_success(active_model)
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
                    message=f"Generation failed on model {active_model}: {err}",
                )
                await self._emit(err_ev)
                raise

            if response_chunk.usage:
                total_tokens += response_chunk.usage.total_tokens

            model_ev = ModelResponseEvent(
                session_id=sess_id,
                model=active_model,
                content=response_chunk.delta_text,
                tool_calls=tuple(response_chunk.tool_calls) if response_chunk.tool_calls else None,
                usage=response_chunk.usage,
            )
            await self._emit(model_ev)
            accumulated_events.append(model_ev)

            # End of step if no tool calls
            if not response_chunk.tool_calls:
                final_text = response_chunk.delta_text or ""
                last_execution_output = final_text
                break

            # Execute tool calls
            for tc in response_chunk.tool_calls:
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
                    is_err = False
                except ToolError as err:
                    logger.warning(
                        "tool_execution_failed", session_id=sess_id, tool=tc.name, error=str(err)
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
                        "unexpected_tool_error", session_id=sess_id, tool=tc.name, error=str(err)
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
                last_execution_output = res_output

        # 5. Verify Phase & Self-Correction Evaluation
        is_success = True
        if active_spec:
            verified_success, feedback = await self.spec_validator.verify(
                active_spec, last_execution_output
            )
            is_success = verified_success
            verify_ev = SpecVerifiedEvent(
                session_id=sess_id,
                spec_id=active_spec.spec_id,
                success=verified_success,
                feedback=feedback,
            )
            await self._emit(verify_ev)
            accumulated_events.append(verify_ev)

        if step >= self.max_steps:
            is_success = False
            err_ev = ErrorEvent(
                session_id=sess_id,
                error_type="MaxStepsExceeded",
                message=f"Agent loop reached maximum step count ({self.max_steps})",
            )
            await self._emit(err_ev)

        # 6. Episodic Memory Trajectory Recording
        if self.episodic_memory is not None:
            try:
                rec = TrajectoryRecord(
                    session_id=sess_id,
                    intent=intent_res.intent_type,
                    model=active_model,
                    steps_count=step,
                    success=is_success,
                    objective=prompt,
                )
                await self.episodic_memory.record_trajectory(rec)
            except Exception as err:
                logger.warning("episodic_memory_record_failed", session_id=sess_id, error=str(err))

        return final_text


__all__ = ["AgentLoop"]
