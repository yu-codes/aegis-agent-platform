"""
Tool Calling Strategy

Uses native LLM tool/function calling for reasoning.
More efficient than ReAct for models with good tool support.

Design decisions:
- Leverages OpenAI/Anthropic native tool calling
- Automatic tool result injection
- Parallel tool execution when possible
- Configurable tool choice modes
- Uses ToolExecutorProtocol from core to avoid circular dependencies
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from src.core.exceptions import MaxIterationsExceededError
from src.core.interfaces import ToolExecutorProtocol
from src.core.types import ExecutionContext, Message, MessageRole, ToolCall, ToolResult
from src.reasoning.llm.base import BaseLLMAdapter
from src.reasoning.strategies.base import (
    ReasoningEvent,
    ReasoningEventType,
    ReasoningResult,
    ReasoningStrategy,
)


class ToolCallingStrategy(ReasoningStrategy):
    """
    Native tool calling strategy.

    Uses the LLM's built-in function/tool calling capability.
    This is typically faster and more reliable than parsing
    tool calls from text (ReAct style).

    Supports:
    - Single and multiple tool calls per turn
    - Parallel tool execution
    - Automatic tool result injection
    - Tool choice control (auto, required, specific)
    """

    def __init__(
        self,
        llm: BaseLLMAdapter,
        tool_executor: ToolExecutorProtocol,
        max_iterations: int = 10,
        parallel_tool_calls: bool = True,
    ):
        self._llm = llm
        self._tool_executor = tool_executor
        self._max_iterations = max_iterations
        self._parallel = parallel_tool_calls

    @property
    def name(self) -> str:
        return "tool_calling"

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        context: ExecutionContext,
    ) -> list[ToolResult]:
        """
        Execute tool calls, optionally in parallel.

        Returns results in the same order as input calls.
        """
        if not tool_calls:
            return []

        if self._parallel and len(tool_calls) > 1:
            # Execute in parallel
            tasks = [
                self._tool_executor.execute(tc.name, tc.arguments, context) for tc in tool_calls
            ]
            return await asyncio.gather(*tasks)
        else:
            # Execute sequentially
            results = []
            for tc in tool_calls:
                result = await self._tool_executor.execute(
                    tc.name,
                    tc.arguments,
                    context,
                )
                results.append(result)
            return results

    async def reason(
        self,
        messages: list[Message],
        context: ExecutionContext,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Execute tool calling reasoning loop.

        The loop continues until:
        1. LLM returns content without tool calls
        2. Max iterations reached
        3. Error occurs
        """
        # Get available tools
        if tools is None:
            tools = self._tool_executor.get_tool_definitions(context)

        # Working copy of messages
        current_messages = list(messages)
        all_tool_results: list[ToolResult] = []
        total_tokens = 0
        total_latency = 0.0

        for iteration in range(self._max_iterations):
            # Call LLM with tools
            response = await self._llm.complete(
                current_messages,
                tools=tools if tools else None,
                tool_choice=tool_choice,
            )

            total_tokens += response.total_tokens
            total_latency += response.latency_ms

            # If no tool calls, we're done
            if not response.has_tool_calls:
                return ReasoningResult(
                    response=response.content or "",
                    tool_results=all_tool_results,
                    iterations=iteration + 1,
                    total_tokens=total_tokens,
                    total_latency_ms=total_latency,
                )

            # Add assistant message with tool calls to history
            current_messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                    metadata={"tool_calls": [tc.model_dump() for tc in response.tool_calls]},
                )
            )

            # Execute tool calls
            results = await self._execute_tool_calls(response.tool_calls, context)
            all_tool_results.extend(results)

            # Add tool results to messages
            for tc, result in zip(response.tool_calls, results, strict=False):
                result_content = (
                    str(result.result) if not result.is_error else f"Error: {result.error}"
                )
                current_messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=result_content,
                        name=tc.name,
                        tool_call_id=tc.id,
                    )
                )

        # Max iterations reached
        raise MaxIterationsExceededError(
            f"Tool calling exceeded {self._max_iterations} iterations",
            context={"tool_calls": len(all_tool_results)},
        )

    async def reason_stream(
        self,
        messages: list[Message],
        context: ExecutionContext,
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ReasoningEvent]:
        """Stream tool calling events."""
        yield ReasoningEvent(type=ReasoningEventType.STARTED)

        if tools is None:
            tools = self._tool_executor.get_tool_definitions(context)

        current_messages = list(messages)
        all_tool_results: list[ToolResult] = []

        for iteration in range(self._max_iterations):
            # Stream LLM response
            content_chunks = []
            tool_calls: list[ToolCall] = []

            async for chunk in self._llm.stream(
                current_messages,
                tools=tools if tools else None,
            ):
                if isinstance(chunk, str):
                    content_chunks.append(chunk)
                    yield ReasoningEvent(
                        type=ReasoningEventType.CONTENT_CHUNK,
                        data=chunk,
                    )
                elif isinstance(chunk, ToolCall):
                    tool_calls.append(chunk)
                    yield ReasoningEvent(
                        type=ReasoningEventType.TOOL_CALL_REQUESTED,
                        data={"name": chunk.name, "arguments": chunk.arguments},
                    )

            content = "".join(content_chunks)

            # If no tool calls, we're done
            if not tool_calls:
                yield ReasoningEvent(
                    type=ReasoningEventType.COMPLETED,
                    data=content,
                    metadata={"iterations": iteration + 1},
                )
                return

            # Add to message history
            current_messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=content,
                    metadata={"tool_calls": [tc.model_dump() for tc in tool_calls]},
                )
            )

            # Execute tools
            results = await self._execute_tool_calls(tool_calls, context)
            all_tool_results.extend(results)

            for tc, result in zip(tool_calls, results, strict=False):
                yield ReasoningEvent(
                    type=ReasoningEventType.TOOL_CALL_COMPLETED,
                    data={
                        "name": tc.name,
                        "result": str(result.result) if not result.is_error else None,
                        "error": result.error,
                    },
                )

                result_content = (
                    str(result.result) if not result.is_error else f"Error: {result.error}"
                )
                current_messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=result_content,
                        name=tc.name,
                        tool_call_id=tc.id,
                    )
                )

        yield ReasoningEvent(
            type=ReasoningEventType.ERROR,
            data=f"Max iterations ({self._max_iterations}) exceeded",
        )
