"""
Agent Runtime

The SINGLE orchestration point for agent execution.
This is the boundary where all agent operations are coordinated.

Design decisions:
- Owns execution lifecycle (state machine)
- Coordinates all components (reasoning, memory, tools, RAG, safety)
- Enforces budgets (tokens, tool calls, time)
- Emits events for observability
- Handles errors and recovery
- NEVER knows about HTTP, persistence implementation, or LLM provider internals

Debugging at 3AM:
- All state transitions are logged
- Every decision point emits events
- Trace ID correlates all operations
- Checkpoints enable replay
"""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from src.core.exceptions import (
    MaxIterationsExceededError,
)
from src.core.interfaces import (
    GuardrailProtocol,
    GuardrailResult,
    InputValidatorProtocol,
    LLMAdapterProtocol,
    MemoryProtocol,
    RetrievedContext,
    RetrieverProtocol,
    ToolExecutorProtocol,
    TracerProtocol,
    ValidationResult,
)
from src.core.types import (
    ExecutionContext,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)


class AgentState(str, Enum):
    """
    Agent execution state machine.

    Valid transitions:
    IDLE → VALIDATING → RETRIEVING → REASONING → EXECUTING_TOOL → APPLYING_GUARDRAILS → COMPLETED
                                          ↑__________________|
    Any state → FAILED
    Any state → CANCELLED
    """

    IDLE = "idle"
    VALIDATING = "validating"
    RETRIEVING = "retrieving"
    REASONING = "reasoning"
    EXECUTING_TOOL = "executing_tool"
    APPLYING_GUARDRAILS = "applying_guardrails"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentEventType(str, Enum):
    """Types of events emitted during execution."""

    # Lifecycle
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"

    # State transitions
    STATE_CHANGED = "state_changed"

    # Reasoning
    LLM_REQUEST_STARTED = "llm_request_started"
    LLM_REQUEST_COMPLETED = "llm_request_completed"
    CONTENT_CHUNK = "content_chunk"  # For streaming

    # Tools
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"

    # RAG
    RAG_RETRIEVAL_STARTED = "rag_retrieval_started"
    RAG_RETRIEVAL_COMPLETED = "rag_retrieval_completed"

    # Safety
    INPUT_VALIDATED = "input_validated"
    INPUT_REJECTED = "input_rejected"
    GUARDRAIL_APPLIED = "guardrail_applied"
    OUTPUT_BLOCKED = "output_blocked"

    # Budget
    TOKEN_BUDGET_WARNING = "token_budget_warning"
    TOOL_BUDGET_WARNING = "tool_budget_warning"


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""

    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Correlation
    trace_id: str | None = None
    span_id: str | None = None
    execution_id: str | None = None


@dataclass
class AgentResult:
    """
    Final result of an agent execution.

    Contains the response plus comprehensive metadata for debugging.
    """

    # Response
    content: str

    # Status
    success: bool = True
    blocked: bool = False
    blocked_reason: str | None = None

    # Metadata
    execution_id: UUID = field(default_factory=uuid4)
    model: str | None = None

    # Metrics
    total_tokens: int = 0
    tool_calls_count: int = 0
    iterations: int = 0
    duration_ms: float = 0.0

    # Tool results for transparency
    tool_results: list[ToolResult] = field(default_factory=list)

    # RAG sources for attribution
    sources: list[str] = field(default_factory=list)

    # For debugging
    trace_id: str | None = None


@dataclass
class RuntimeConfig:
    """
    Configuration for AgentRuntime.

    Defines budgets and limits for execution.
    """

    # LLM
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens_per_call: int = 4096

    # System prompt
    system_prompt: str = "You are a helpful AI assistant."

    # Budgets
    max_iterations: int = 10
    max_tool_calls: int = 20
    max_total_tokens: int = 100000
    timeout_seconds: float = 300.0

    # Features
    enable_memory: bool = True
    enable_rag: bool = True
    enable_tools: bool = True

    # RAG
    rag_top_k: int = 5
    rag_min_score: float = 0.5


class AgentRuntime:
    """
    The SINGLE orchestration point for agent execution.

    This is the canonical way to run an agent. All agent executions
    must go through this class.

    Responsibilities:
    - State machine management
    - Component coordination
    - Budget enforcement
    - Event emission
    - Error handling

    Does NOT know about:
    - HTTP/transport
    - Persistence implementation
    - LLM provider internals
    - Vector store implementation
    """

    def __init__(
        self,
        llm: LLMAdapterProtocol,
        tool_executor: ToolExecutorProtocol | None = None,
        memory: MemoryProtocol | None = None,
        retriever: RetrieverProtocol | None = None,
        input_validator: InputValidatorProtocol | None = None,
        guardrails: GuardrailProtocol | None = None,
        tracer: TracerProtocol | None = None,
        config: RuntimeConfig | None = None,
    ):
        # Required
        self._llm = llm

        # Optional components
        self._tools = tool_executor
        self._memory = memory
        self._retriever = retriever
        self._validator = input_validator
        self._guardrails = guardrails
        self._tracer = tracer

        # Configuration
        self._config = config or RuntimeConfig()

        # Execution state (reset per run)
        self._state = AgentState.IDLE
        self._execution_id: UUID | None = None
        self._total_tokens = 0
        self._tool_calls_count = 0
        self._iterations = 0

    @property
    def state(self) -> AgentState:
        """Current execution state."""
        return self._state

    def _set_state(self, new_state: AgentState) -> AgentEvent:
        """Transition to a new state, returning the event."""
        old_state = self._state
        self._state = new_state

        return AgentEvent(
            type=AgentEventType.STATE_CHANGED,
            data={"from": old_state.value, "to": new_state.value},
            execution_id=str(self._execution_id) if self._execution_id else None,
        )

    async def run(
        self,
        message: str,
        context: ExecutionContext,
        history: list[Message] | None = None,
    ) -> AgentResult:
        """
        Execute a single turn (non-streaming).

        This is the main entry point for agent execution.

        Args:
            message: User message
            context: Execution context with permissions and metadata
            history: Optional conversation history (if not using memory)

        Returns:
            AgentResult with response and metadata
        """
        start_time = time.perf_counter()

        # Reset execution state
        self._execution_id = context.execution_id
        self._total_tokens = 0
        self._tool_calls_count = 0
        self._iterations = 0

        # Start trace span
        span = None
        if self._tracer:
            span = self._tracer.start_span(
                "agent.run",
                parent_id=context.parent_span_id,
                attributes={"execution_id": str(self._execution_id)},
            )

        try:
            # Phase 1: Validate input
            self._set_state(AgentState.VALIDATING)
            validation = await self._validate_input(message, context)

            if not validation.is_valid:
                return AgentResult(
                    content="I cannot process that request.",
                    success=False,
                    blocked=True,
                    blocked_reason="; ".join(validation.issues or []),
                    execution_id=self._execution_id,
                    trace_id=context.trace_id,
                )

            # Use sanitized input if available
            safe_message = validation.sanitized_input or message

            # Phase 2: Retrieve context (RAG)
            rag_context: list[RetrievedContext] = []
            if self._retriever and self._config.enable_rag and context.enable_rag:
                self._set_state(AgentState.RETRIEVING)
                rag_context = await self._retrieve_context(safe_message)

            # Phase 3: Build messages
            messages = await self._build_messages(
                message=safe_message,
                context=context,
                history=history,
                rag_context=rag_context,
            )

            # Phase 4: Reasoning loop
            self._set_state(AgentState.REASONING)
            tool_results: list[ToolResult] = []

            for iteration in range(self._config.max_iterations):
                self._iterations = iteration + 1

                # Check budgets
                if self._total_tokens >= self._config.max_total_tokens:
                    raise MaxIterationsExceededError("Token budget exhausted")

                if self._tool_calls_count >= self._config.max_tool_calls:
                    raise MaxIterationsExceededError("Tool call budget exhausted")

                # Get tool definitions if tools enabled
                tool_defs = None
                if self._tools and self._config.enable_tools:
                    tool_defs = self._tools.get_tool_definitions(context)

                # Call LLM
                response = await self._llm.complete(
                    messages,
                    tools=tool_defs,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens_per_call,
                )

                self._total_tokens += response.total_tokens

                # Check if we need to execute tools
                if response.has_tool_calls and self._tools:
                    self._set_state(AgentState.EXECUTING_TOOL)

                    for tool_call in response.tool_calls:
                        result = await self._execute_tool(tool_call, context)
                        tool_results.append(result)

                        # Add tool call and result to messages
                        messages.append(
                            Message(
                                role=MessageRole.ASSISTANT,
                                content="",
                                tool_calls=[tool_call],
                            )
                        )
                        messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=(
                                    str(result.result)
                                    if not result.error
                                    else f"Error: {result.error}"
                                ),
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                            )
                        )

                    self._set_state(AgentState.REASONING)
                    continue  # Loop for next LLM call

                # No tool calls - we have the final response
                final_content = response.content or ""
                break
            else:
                # Max iterations reached
                raise MaxIterationsExceededError(
                    f"Max iterations ({self._config.max_iterations}) exceeded"
                )

            # Phase 5: Apply guardrails
            self._set_state(AgentState.APPLYING_GUARDRAILS)
            guardrail_result = await self._apply_guardrails(final_content, context)

            if guardrail_result.blocked:
                return AgentResult(
                    content="I cannot provide that response.",
                    success=True,
                    blocked=True,
                    blocked_reason=guardrail_result.reason,
                    execution_id=self._execution_id,
                    tool_results=tool_results,
                    trace_id=context.trace_id,
                )

            # Use modified content if guardrails changed it
            output_content = guardrail_result.modified_content or final_content

            # Phase 6: Complete
            self._set_state(AgentState.COMPLETED)

            duration = (time.perf_counter() - start_time) * 1000

            return AgentResult(
                content=output_content,
                success=True,
                execution_id=self._execution_id,
                model=self._llm.model,
                total_tokens=self._total_tokens,
                tool_calls_count=self._tool_calls_count,
                iterations=self._iterations,
                duration_ms=duration,
                tool_results=tool_results,
                sources=[rc.source for rc in rag_context if rc.source],
                trace_id=context.trace_id,
            )

        except asyncio.CancelledError:
            self._set_state(AgentState.CANCELLED)
            raise

        except Exception as e:
            self._set_state(AgentState.FAILED)

            if span:
                span.set_status("error", str(e))

            return AgentResult(
                content=f"An error occurred: {type(e).__name__}",
                success=False,
                execution_id=self._execution_id,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                trace_id=context.trace_id,
            )

        finally:
            if span:
                span.end()

    async def run_stream(
        self,
        message: str,
        context: ExecutionContext,
        history: list[Message] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Execute with streaming.

        Yields events as they occur during execution.
        """
        start_time = time.perf_counter()

        # Reset execution state
        self._execution_id = context.execution_id
        self._total_tokens = 0
        self._tool_calls_count = 0
        self._iterations = 0

        yield AgentEvent(
            type=AgentEventType.RUN_STARTED,
            data={"execution_id": str(self._execution_id)},
            trace_id=context.trace_id,
        )

        try:
            # Phase 1: Validate
            self._set_state(AgentState.VALIDATING)
            yield self._set_state(AgentState.VALIDATING)

            validation = await self._validate_input(message, context)

            if not validation.is_valid:
                yield AgentEvent(
                    type=AgentEventType.INPUT_REJECTED,
                    data={"issues": validation.issues},
                )
                yield AgentEvent(
                    type=AgentEventType.RUN_FAILED,
                    data={"reason": "Input validation failed"},
                )
                return

            yield AgentEvent(
                type=AgentEventType.INPUT_VALIDATED,
                data={"risk_score": validation.risk_score},
            )

            safe_message = validation.sanitized_input or message

            # Phase 2: RAG
            rag_context: list[RetrievedContext] = []
            if self._retriever and self._config.enable_rag and context.enable_rag:
                yield self._set_state(AgentState.RETRIEVING)
                yield AgentEvent(type=AgentEventType.RAG_RETRIEVAL_STARTED)

                rag_context = await self._retrieve_context(safe_message)

                yield AgentEvent(
                    type=AgentEventType.RAG_RETRIEVAL_COMPLETED,
                    data={"documents_count": len(rag_context)},
                )

            # Phase 3: Build messages
            messages = await self._build_messages(
                message=safe_message,
                context=context,
                history=history,
                rag_context=rag_context,
            )

            # Phase 4: Reasoning loop with streaming
            yield self._set_state(AgentState.REASONING)
            tool_results: list[ToolResult] = []
            final_content = ""

            for iteration in range(self._config.max_iterations):
                self._iterations = iteration + 1

                # Check budgets
                if self._total_tokens >= self._config.max_total_tokens:
                    yield AgentEvent(
                        type=AgentEventType.TOKEN_BUDGET_WARNING,
                        data={"used": self._total_tokens, "max": self._config.max_total_tokens},
                    )
                    break

                tool_defs = None
                if self._tools and self._config.enable_tools:
                    tool_defs = self._tools.get_tool_definitions(context)

                yield AgentEvent(type=AgentEventType.LLM_REQUEST_STARTED)

                # Stream from LLM
                content_buffer = ""
                tool_calls: list[ToolCall] = []

                async for chunk in self._llm.stream(
                    messages,
                    tools=tool_defs,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens_per_call,
                ):
                    if chunk.content:
                        content_buffer += chunk.content
                        yield AgentEvent(
                            type=AgentEventType.CONTENT_CHUNK,
                            data={"content": chunk.content},
                        )

                    if chunk.tool_calls:
                        tool_calls.extend(chunk.tool_calls)

                    self._total_tokens = chunk.total_tokens

                yield AgentEvent(
                    type=AgentEventType.LLM_REQUEST_COMPLETED,
                    data={"tokens": self._total_tokens},
                )

                # Execute tools if needed
                if tool_calls and self._tools:
                    yield self._set_state(AgentState.EXECUTING_TOOL)

                    for tool_call in tool_calls:
                        yield AgentEvent(
                            type=AgentEventType.TOOL_CALL_STARTED,
                            data={"tool": tool_call.name, "id": tool_call.id},
                        )

                        result = await self._execute_tool(tool_call, context)
                        tool_results.append(result)

                        if result.error:
                            yield AgentEvent(
                                type=AgentEventType.TOOL_CALL_FAILED,
                                data={"tool": tool_call.name, "error": result.error},
                            )
                        else:
                            yield AgentEvent(
                                type=AgentEventType.TOOL_CALL_COMPLETED,
                                data={"tool": tool_call.name, "duration_ms": result.duration_ms},
                            )

                        # Update messages for next iteration
                        messages.append(
                            Message(
                                role=MessageRole.ASSISTANT,
                                content="",
                                tool_calls=[tool_call],
                            )
                        )
                        messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=(
                                    str(result.result)
                                    if not result.error
                                    else f"Error: {result.error}"
                                ),
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                            )
                        )

                    yield self._set_state(AgentState.REASONING)
                    continue

                # No tools - done
                final_content = content_buffer
                break

            # Phase 5: Guardrails
            yield self._set_state(AgentState.APPLYING_GUARDRAILS)

            guardrail_result = await self._apply_guardrails(final_content, context)

            if guardrail_result.blocked:
                yield AgentEvent(
                    type=AgentEventType.OUTPUT_BLOCKED,
                    data={"reason": guardrail_result.reason},
                )
                final_content = "I cannot provide that response."
            else:
                yield AgentEvent(type=AgentEventType.GUARDRAIL_APPLIED)
                if guardrail_result.modified_content:
                    final_content = guardrail_result.modified_content

            # Complete
            yield self._set_state(AgentState.COMPLETED)

            duration = (time.perf_counter() - start_time) * 1000

            yield AgentEvent(
                type=AgentEventType.RUN_COMPLETED,
                data={
                    "content": final_content,
                    "total_tokens": self._total_tokens,
                    "tool_calls_count": self._tool_calls_count,
                    "iterations": self._iterations,
                    "duration_ms": duration,
                },
            )

        except Exception as e:
            self._set_state(AgentState.FAILED)
            yield AgentEvent(
                type=AgentEventType.RUN_FAILED,
                data={"error": str(e), "type": type(e).__name__},
            )

    # =========================================================================
    # Private Methods
    # =========================================================================

    async def _validate_input(
        self,
        message: str,
        context: ExecutionContext,
    ) -> ValidationResult:
        """Validate user input."""
        if self._validator:
            return await self._validator.validate(message, context)

        # Default: accept all
        return ValidationResult(is_valid=True, sanitized_input=message)

    async def _retrieve_context(
        self,
        query: str,
    ) -> list[RetrievedContext]:
        """Retrieve relevant context from RAG."""
        if not self._retriever:
            return []

        try:
            docs = await self._retriever.retrieve(
                query,
                top_k=self._config.rag_top_k,
            )
            return [d for d in docs if d.score >= self._config.rag_min_score]
        except Exception:
            # RAG failure is non-fatal
            return []

    async def _build_messages(
        self,
        message: str,
        context: ExecutionContext,
        history: list[Message] | None,
        rag_context: list[RetrievedContext],
    ) -> list[Message]:
        """Assemble the message list for the LLM."""
        messages: list[Message] = []

        # System prompt
        messages.append(
            Message(
                role=MessageRole.SYSTEM,
                content=self._config.system_prompt,
            )
        )

        # RAG context (if available)
        if rag_context:
            context_text = "\n\n---\n\n".join(
                f"[Source: {rc.source or 'unknown'}]\n{rc.content}" for rc in rag_context
            )
            messages.append(
                Message(
                    role=MessageRole.SYSTEM,
                    content=f"Relevant context from knowledge base:\n\n{context_text}",
                )
            )

        # Conversation history
        if history:
            messages.extend(history)
        elif self._memory and self._config.enable_memory and context.enable_memory:
            # Load from memory
            try:
                memory_messages = await self._memory.get_context(
                    str(context.session_id),
                    query=message,
                )
                messages.extend(memory_messages)
            except Exception:
                pass  # Memory failure is non-fatal

        # Current message
        messages.append(
            Message(
                role=MessageRole.USER,
                content=message,
            )
        )

        return messages

    async def _execute_tool(
        self,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute a single tool call."""
        self._tool_calls_count += 1

        if not self._tools:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error="Tool execution not available",
            )

        try:
            return await self._tools.execute(
                tool_call.name,
                tool_call.arguments,
                context,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e),
            )

    async def _apply_guardrails(
        self,
        content: str,
        context: ExecutionContext,
    ) -> GuardrailResult:
        """Apply output guardrails."""
        if self._guardrails:
            return await self._guardrails.check(content, context)

        # Default: pass all
        return GuardrailResult(passed=True)
