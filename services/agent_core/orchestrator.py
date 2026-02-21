"""
Agent Orchestrator

The central orchestration point for agent execution.
Coordinates all components and manages execution lifecycle.

Design decisions:
- Owns execution lifecycle (state machine)
- Coordinates reasoning, memory, tools, RAG, safety
- Enforces budgets (tokens, tool calls, time)
- Emits events for observability
- Handles errors and recovery

Based on: src/runtime/agent.py
"""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4


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

    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    STATE_CHANGED = "state_changed"
    LLM_REQUEST_STARTED = "llm_request_started"
    LLM_REQUEST_COMPLETED = "llm_request_completed"
    CONTENT_CHUNK = "content_chunk"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"
    RAG_RETRIEVAL_STARTED = "rag_retrieval_started"
    RAG_RETRIEVAL_COMPLETED = "rag_retrieval_completed"


@dataclass
class AgentEvent:
    """Event emitted during execution."""

    type: AgentEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None


@dataclass
class ExecutionConfig:
    """Configuration for agent execution."""

    max_iterations: int = 10
    max_tokens: int = 100000
    max_tool_calls: int = 20
    timeout_seconds: float = 300.0
    enable_streaming: bool = True
    enable_rag: bool = True
    enable_memory: bool = True


@dataclass
class ExecutionContext:
    """Context for a single execution."""

    session_id: UUID
    request_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    tenant_id: str | None = None
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of agent execution."""

    success: bool
    response: str = ""
    state: AgentState = AgentState.COMPLETED
    iterations: int = 0
    tokens_used: int = 0
    tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    duration: float = 0.0
    duration_ms: float = 0.0
    events: list[AgentEvent] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Protocol definitions for dependencies
class LLMAdapterProtocol(Protocol):
    """Protocol for LLM adapters."""

    async def complete(self, messages: list[dict], **kwargs) -> dict: ...
    async def stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]: ...


class MemoryProtocol(Protocol):
    """Protocol for memory systems."""

    async def get_context(self, session_id: UUID) -> list[dict]: ...
    async def add_message(self, session_id: UUID, role: str, content: str) -> None: ...


class RetrieverProtocol(Protocol):
    """Protocol for RAG retriever."""

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]: ...


class ToolExecutorProtocol(Protocol):
    """Protocol for tool executor."""

    async def execute(self, tool_name: str, arguments: dict) -> dict: ...


class GuardrailProtocol(Protocol):
    """Protocol for guardrails."""

    async def check_input(self, content: str) -> dict: ...
    async def check_output(self, content: str, input_content: str) -> dict: ...


class AgentOrchestrator:
    """
    Main agent orchestrator.

    Coordinates all components and manages the execution lifecycle.
    """

    def __init__(
        self,
        llm: LLMAdapterProtocol,
        memory: MemoryProtocol | None = None,
        retriever: RetrieverProtocol | None = None,
        tool_executor: ToolExecutorProtocol | None = None,
        guardrails: list[GuardrailProtocol] | None = None,
        config: ExecutionConfig | None = None,
    ):
        self._llm = llm
        self._memory = memory
        self._retriever = retriever
        self._tool_executor = tool_executor
        self._guardrails = guardrails or []
        self._config = config or ExecutionConfig()

        # Execution state
        self._state = AgentState.IDLE
        self._events: list[AgentEvent] = []
        self._current_context: ExecutionContext | None = None

    @property
    def state(self) -> AgentState:
        return self._state

    async def run(
        self,
        query: str,
        context: ExecutionContext,
        system_prompt: str | None = None,
    ) -> ExecutionResult:
        """
        Execute a single query.

        Args:
            query: User's input query
            context: Execution context
            system_prompt: Optional system prompt override

        Returns:
            ExecutionResult with response and metadata
        """
        start_time = time.time()
        self._current_context = context
        self._events = []

        result = ExecutionResult(
            success=False,
            metadata={"session_id": str(context.session_id)},
        )

        try:
            self._emit_event(AgentEventType.RUN_STARTED, {"query": query[:100]})

            # Phase 1: Input validation
            self._transition_to(AgentState.VALIDATING)
            validated_query = await self._validate_input(query)
            if validated_query is None:
                result.error = "Input validation failed"
                return result

            # Phase 2: Memory retrieval
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if self._memory and self._config.enable_memory:
                history = await self._memory.get_context(context.session_id)
                messages.extend(history)

            # Phase 3: RAG retrieval
            rag_context = ""
            if self._retriever and self._config.enable_rag:
                self._transition_to(AgentState.RETRIEVING)
                rag_context = await self._retrieve_context(validated_query)

            # Build user message with RAG context
            user_message = validated_query
            if rag_context:
                user_message = f"Context:\n{rag_context}\n\nUser Query: {validated_query}"
            messages.append({"role": "user", "content": user_message})

            # Phase 4: Reasoning loop
            iteration = 0
            final_response = ""

            while iteration < self._config.max_iterations:
                iteration += 1
                self._transition_to(AgentState.REASONING)

                self._emit_event(
                    AgentEventType.LLM_REQUEST_STARTED,
                    {
                        "iteration": iteration,
                        "message_count": len(messages),
                    },
                )

                response = await self._llm.complete(messages)

                # Convert CompletionResult to dict if needed
                if hasattr(response, "content"):
                    # It's a CompletionResult object
                    response_dict = {
                        "content": response.content,
                        "tool_calls": getattr(response, "tool_calls", []),
                    }
                    result.tokens_used += getattr(response, "input_tokens", 0) + getattr(
                        response, "output_tokens", 0
                    )
                else:
                    # It's already a dict
                    response_dict = response

                self._emit_event(
                    AgentEventType.LLM_REQUEST_COMPLETED,
                    {
                        "iteration": iteration,
                    },
                )

                # Check for tool calls
                if response_dict.get("tool_calls"):
                    self._transition_to(AgentState.EXECUTING_TOOL)
                    tool_results = await self._execute_tools(response_dict["tool_calls"])

                    # Add assistant message with tool calls
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response_dict.get("content", ""),
                            "tool_calls": response_dict["tool_calls"],
                        }
                    )

                    # Add tool results
                    for tool_result in tool_results:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_result["tool_call_id"],
                                "content": tool_result["content"],
                            }
                        )

                    result.tool_calls += len(response_dict["tool_calls"])
                    # Track tool names used
                    for tc in response_dict["tool_calls"]:
                        tool_name = tc.get("function", {}).get("name", "unknown")
                        if tool_name not in result.tools_used:
                            result.tools_used.append(tool_name)
                    continue

                # No tool calls - we have a final response
                final_response = response_dict.get("content", "")
                break

            # Phase 5: Output guardrails
            self._transition_to(AgentState.APPLYING_GUARDRAILS)
            final_response = await self._apply_output_guardrails(final_response, validated_query)

            # Phase 6: Update memory
            if self._memory and self._config.enable_memory:
                await self._memory.add_message(context.session_id, "user", validated_query)
                await self._memory.add_message(context.session_id, "assistant", final_response)

            # Success
            self._transition_to(AgentState.COMPLETED)
            result.success = True
            result.response = final_response
            result.iterations = iteration
            result.state = AgentState.COMPLETED

            self._emit_event(
                AgentEventType.RUN_COMPLETED,
                {
                    "iterations": iteration,
                    "response_length": len(final_response),
                },
            )

        except asyncio.CancelledError:
            self._transition_to(AgentState.CANCELLED)
            result.state = AgentState.CANCELLED
            result.error = "Execution cancelled"

        except Exception as e:
            self._transition_to(AgentState.FAILED)
            result.state = AgentState.FAILED
            result.error = str(e)

            self._emit_event(AgentEventType.RUN_FAILED, {"error": str(e)})

        finally:
            elapsed = time.time() - start_time
            result.duration = elapsed
            result.duration_ms = elapsed * 1000
            result.events = self._events.copy()
            self._current_context = None

        return result

    async def run_stream(
        self,
        query: str,
        context: ExecutionContext,
        system_prompt: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Execute with streaming response.

        Yields events including content chunks.
        """
        self._current_context = context
        self._events = []

        try:
            yield AgentEvent(
                type=AgentEventType.RUN_STARTED,
                data={"query": query[:100]},
            )

            # Validate input
            self._transition_to(AgentState.VALIDATING)
            validated_query = await self._validate_input(query)
            if validated_query is None:
                yield AgentEvent(
                    type=AgentEventType.RUN_FAILED,
                    data={"error": "Input validation failed"},
                )
                return

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if self._memory and self._config.enable_memory:
                history = await self._memory.get_context(context.session_id)
                messages.extend(history)

            # RAG retrieval
            if self._retriever and self._config.enable_rag:
                self._transition_to(AgentState.RETRIEVING)
                yield AgentEvent(type=AgentEventType.RAG_RETRIEVAL_STARTED)
                rag_context = await self._retrieve_context(validated_query)
                yield AgentEvent(
                    type=AgentEventType.RAG_RETRIEVAL_COMPLETED,
                    data={"context_length": len(rag_context)},
                )

                if rag_context:
                    validated_query = f"Context:\n{rag_context}\n\nUser Query: {validated_query}"

            messages.append({"role": "user", "content": validated_query})

            # Stream LLM response
            self._transition_to(AgentState.REASONING)
            yield AgentEvent(type=AgentEventType.LLM_REQUEST_STARTED)

            full_response = ""
            async for chunk in self._llm.stream(messages):
                full_response += chunk
                yield AgentEvent(
                    type=AgentEventType.CONTENT_CHUNK,
                    data={"content": chunk},
                )

            # Apply output guardrails
            self._transition_to(AgentState.APPLYING_GUARDRAILS)
            final_response = await self._apply_output_guardrails(full_response, query)

            # Update memory
            if self._memory and self._config.enable_memory:
                await self._memory.add_message(context.session_id, "user", query)
                await self._memory.add_message(context.session_id, "assistant", final_response)

            self._transition_to(AgentState.COMPLETED)
            yield AgentEvent(
                type=AgentEventType.RUN_COMPLETED,
                data={"response_length": len(final_response)},
            )

        except Exception as e:
            self._transition_to(AgentState.FAILED)
            yield AgentEvent(
                type=AgentEventType.RUN_FAILED,
                data={"error": str(e)},
            )

        finally:
            self._current_context = None

    def _transition_to(self, new_state: AgentState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._emit_event(
            AgentEventType.STATE_CHANGED,
            {
                "from": old_state.value,
                "to": new_state.value,
            },
        )

    def _emit_event(self, event_type: AgentEventType, data: dict | None = None) -> None:
        """Emit an event."""
        event = AgentEvent(
            type=event_type,
            data=data or {},
            trace_id=self._current_context.request_id if self._current_context else None,
        )
        self._events.append(event)

    async def _validate_input(self, content: str) -> str | None:
        """Validate input through guardrails."""
        validated = content
        for guardrail in self._guardrails:
            result = await guardrail.check_input(validated)
            if result.get("action") == "block":
                return None
            if result.get("modified_content"):
                validated = result["modified_content"]
        return validated

    async def _apply_output_guardrails(self, output: str, input_content: str) -> str:
        """Apply output guardrails."""
        filtered = output
        for guardrail in self._guardrails:
            result = await guardrail.check_output(filtered, input_content)
            if result.get("modified_content"):
                filtered = result["modified_content"]
        return filtered

    async def _retrieve_context(self, query: str) -> str:
        """Retrieve RAG context."""
        if not self._retriever:
            return ""

        self._emit_event(AgentEventType.RAG_RETRIEVAL_STARTED, {"query": query[:50]})

        docs = await self._retriever.retrieve(query, top_k=5)

        self._emit_event(
            AgentEventType.RAG_RETRIEVAL_COMPLETED,
            {
                "document_count": len(docs),
            },
        )

        if not docs:
            return ""

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source", "Unknown")
            content = doc.get("content", "")
            context_parts.append(f"[{i}] ({source}): {content}")

        return "\n\n".join(context_parts)

    async def _execute_tools(self, tool_calls: list[dict]) -> list[dict]:
        """Execute tool calls."""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name", "")
            tool_args = tool_call.get("function", {}).get("arguments", {})
            tool_call_id = tool_call.get("id", str(uuid4()))

            self._emit_event(
                AgentEventType.TOOL_CALL_STARTED,
                {
                    "tool": tool_name,
                    "tool_call_id": tool_call_id,
                },
            )

            try:
                if self._tool_executor:
                    result = await self._tool_executor.execute(tool_name, tool_args)
                    content = result.get("output", str(result))
                else:
                    content = f"Tool '{tool_name}' not available"

                self._emit_event(
                    AgentEventType.TOOL_CALL_COMPLETED,
                    {
                        "tool": tool_name,
                        "tool_call_id": tool_call_id,
                    },
                )

            except Exception as e:
                content = f"Tool error: {str(e)}"
                self._emit_event(
                    AgentEventType.TOOL_CALL_FAILED,
                    {
                        "tool": tool_name,
                        "tool_call_id": tool_call_id,
                        "error": str(e),
                    },
                )

            results.append(
                {
                    "tool_call_id": tool_call_id,
                    "content": content,
                }
            )

        return results
