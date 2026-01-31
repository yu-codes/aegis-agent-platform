"""
ReAct Reasoning Strategy

Implements the ReAct (Reasoning + Acting) framework.
The agent explicitly reasons about each step before acting.

Reference: https://arxiv.org/abs/2210.03629

Design decisions:
- Structured output parsing for thoughts/actions
- Configurable max iterations to prevent loops
- Explicit final answer detection
- Full trace of reasoning steps for debugging
"""

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from src.core.types import ExecutionContext, Message, MessageRole, ToolResult
from src.core.exceptions import MaxIterationsExceededError
from src.reasoning.llm.base import BaseLLMAdapter
from src.reasoning.prompts.template import get_prompt_registry
from src.reasoning.strategies.base import (
    ReasoningStrategy,
    ReasoningResult,
    ReasoningEvent,
    ReasoningEventType,
    ToolExecutor,
)


@dataclass
class ReActStep:
    """A single step in ReAct reasoning."""
    
    thought: str
    action: str
    action_input: str | dict[str, Any]
    observation: str = ""


class ReActStrategy(ReasoningStrategy):
    """
    ReAct (Reason + Act) strategy.
    
    The agent follows a thought-action-observation loop:
    1. Think about what to do next
    2. Take an action (use tool or answer)
    3. Observe the result
    4. Repeat until final answer
    
    This provides interpretable reasoning traces but
    may be slower than direct tool calling.
    """
    
    def __init__(
        self,
        llm: BaseLLMAdapter,
        tool_executor: ToolExecutor,
        max_iterations: int = 10,
    ):
        self._llm = llm
        self._tool_executor = tool_executor
        self._max_iterations = max_iterations
        self._registry = get_prompt_registry()
    
    @property
    def name(self) -> str:
        return "react"
    
    def _parse_react_output(self, text: str) -> ReActStep | None:
        """
        Parse LLM output into ReAct step.
        
        Expected format:
        Thought: ...
        Action: ...
        Action Input: ...
        """
        # Patterns for parsing
        thought_pattern = r"Thought:\s*(.+?)(?=Action:|$)"
        action_pattern = r"Action:\s*(.+?)(?=Action Input:|$)"
        action_input_pattern = r"Action Input:\s*(.+?)$"
        
        thought_match = re.search(thought_pattern, text, re.DOTALL | re.IGNORECASE)
        action_match = re.search(action_pattern, text, re.DOTALL | re.IGNORECASE)
        input_match = re.search(action_input_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if not thought_match or not action_match:
            return None
        
        return ReActStep(
            thought=thought_match.group(1).strip(),
            action=action_match.group(1).strip(),
            action_input=input_match.group(1).strip() if input_match else "",
        )
    
    async def reason(
        self,
        messages: list[Message],
        context: ExecutionContext,
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """Execute ReAct reasoning loop."""
        # Get available tools
        if tools is None:
            tools = self._tool_executor.get_tool_definitions(context)
        
        # Get the task from the last user message
        task = ""
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                task = msg.content
                break
        
        steps: list[ReActStep] = []
        tool_results: list[ToolResult] = []
        total_tokens = 0
        total_latency = 0.0
        
        for iteration in range(self._max_iterations):
            # Build reasoning prompt
            react_template = self._registry.get("react_reasoning")
            if not react_template:
                raise ValueError("ReAct template not found")
            
            reasoning_prompt = react_template.render(
                task=task,
                previous_steps=[
                    {
                        "thought": s.thought,
                        "action": s.action,
                        "observation": s.observation,
                    }
                    for s in steps
                ] if steps else None,
            )
            
            # Build messages for LLM
            reasoning_messages = list(messages)
            reasoning_messages.append(Message(
                role=MessageRole.USER,
                content=reasoning_prompt,
            ))
            
            # Get LLM response
            response = await self._llm.complete(reasoning_messages)
            total_tokens += response.total_tokens
            total_latency += response.latency_ms
            
            if not response.content:
                continue
            
            # Parse the response
            step = self._parse_react_output(response.content)
            if not step:
                # If parsing fails, treat as final answer
                return ReasoningResult(
                    response=response.content,
                    tool_results=tool_results,
                    iterations=iteration + 1,
                    total_tokens=total_tokens,
                    total_latency_ms=total_latency,
                    metadata={"steps": [s.__dict__ for s in steps]},
                )
            
            # Check for final answer
            if step.action.lower() in ("final answer", "finish", "done"):
                return ReasoningResult(
                    response=str(step.action_input),
                    tool_results=tool_results,
                    iterations=iteration + 1,
                    total_tokens=total_tokens,
                    total_latency_ms=total_latency,
                    metadata={"steps": [s.__dict__ for s in steps]},
                )
            
            # Execute tool
            tool_name = step.action
            try:
                # Parse action input as arguments
                if isinstance(step.action_input, str):
                    # Try to parse as JSON, fallback to raw string
                    import json
                    try:
                        args = json.loads(step.action_input)
                    except json.JSONDecodeError:
                        args = {"input": step.action_input}
                else:
                    args = step.action_input
                
                result = await self._tool_executor.execute(
                    tool_name,
                    args if isinstance(args, dict) else {"input": args},
                    context,
                )
                tool_results.append(result)
                
                step.observation = str(result.result) if not result.is_error else f"Error: {result.error}"
            except Exception as e:
                step.observation = f"Error executing {tool_name}: {str(e)}"
            
            steps.append(step)
        
        # Max iterations reached
        raise MaxIterationsExceededError(
            f"ReAct reasoning exceeded {self._max_iterations} iterations",
            context={"steps": len(steps)},
        )
    
    async def reason_stream(
        self,
        messages: list[Message],
        context: ExecutionContext,
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ReasoningEvent]:
        """Stream ReAct reasoning events."""
        yield ReasoningEvent(type=ReasoningEventType.STARTED)
        
        if tools is None:
            tools = self._tool_executor.get_tool_definitions(context)
        
        task = ""
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                task = msg.content
                break
        
        steps: list[ReActStep] = []
        tool_results: list[ToolResult] = []
        
        for iteration in range(self._max_iterations):
            react_template = self._registry.get("react_reasoning")
            if not react_template:
                raise ValueError("ReAct template not found")
            
            reasoning_prompt = react_template.render(
                task=task,
                previous_steps=[
                    {
                        "thought": s.thought,
                        "action": s.action,
                        "observation": s.observation,
                    }
                    for s in steps
                ] if steps else None,
            )
            
            reasoning_messages = list(messages)
            reasoning_messages.append(Message(
                role=MessageRole.USER,
                content=reasoning_prompt,
            ))
            
            # Stream LLM response
            content_buffer = ""
            async for chunk in self._llm.stream(reasoning_messages):
                if isinstance(chunk, str):
                    content_buffer += chunk
                    yield ReasoningEvent(
                        type=ReasoningEventType.CONTENT_CHUNK,
                        data=chunk,
                    )
            
            step = self._parse_react_output(content_buffer)
            if not step:
                yield ReasoningEvent(
                    type=ReasoningEventType.COMPLETED,
                    data=content_buffer,
                )
                return
            
            yield ReasoningEvent(
                type=ReasoningEventType.THINKING,
                data=step.thought,
            )
            
            if step.action.lower() in ("final answer", "finish", "done"):
                yield ReasoningEvent(
                    type=ReasoningEventType.COMPLETED,
                    data=str(step.action_input),
                    metadata={"steps": len(steps)},
                )
                return
            
            yield ReasoningEvent(
                type=ReasoningEventType.TOOL_CALL_REQUESTED,
                data={"name": step.action, "arguments": step.action_input},
            )
            
            # Execute tool
            try:
                import json
                if isinstance(step.action_input, str):
                    try:
                        args = json.loads(step.action_input)
                    except json.JSONDecodeError:
                        args = {"input": step.action_input}
                else:
                    args = step.action_input
                
                result = await self._tool_executor.execute(
                    step.action,
                    args if isinstance(args, dict) else {"input": args},
                    context,
                )
                tool_results.append(result)
                step.observation = str(result.result) if not result.is_error else f"Error: {result.error}"
            except Exception as e:
                step.observation = f"Error: {str(e)}"
            
            yield ReasoningEvent(
                type=ReasoningEventType.TOOL_CALL_COMPLETED,
                data={"name": step.action, "observation": step.observation},
            )
            
            steps.append(step)
        
        yield ReasoningEvent(
            type=ReasoningEventType.ERROR,
            data=f"Max iterations ({self._max_iterations}) exceeded",
        )
