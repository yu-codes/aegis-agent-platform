"""
Base Reasoning Strategy

Defines the interface for reasoning strategies.
Different strategies implement different approaches to
problem-solving (ReAct, tool-calling, chain-of-thought, etc).

Design decisions:
- Strategy pattern for swappable reasoning approaches
- Async execution for non-blocking I/O
- Context-aware: strategies receive full execution context
- Observable: strategies emit events for tracing
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from src.core.types import ExecutionContext, LLMResponse, Message, ToolResult


class ReasoningEventType(str, Enum):
    """Types of events emitted during reasoning."""
    
    STARTED = "started"
    THINKING = "thinking"
    TOOL_CALL_REQUESTED = "tool_call_requested"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    CONTENT_CHUNK = "content_chunk"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ReasoningEvent:
    """
    Event emitted during reasoning process.
    
    Used for observability and streaming responses.
    """
    
    type: ReasoningEventType
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """
    Final result of a reasoning process.
    
    Contains the response plus metadata for tracking.
    """
    
    response: str
    tool_results: list[ToolResult] = field(default_factory=list)
    iterations: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ReasoningStrategy(ABC):
    """
    Abstract base for reasoning strategies.
    
    Strategies define HOW the agent thinks and acts.
    They orchestrate LLM calls, tool execution, and
    iteration until a final response is produced.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        pass
    
    @abstractmethod
    async def reason(
        self,
        messages: list[Message],
        context: ExecutionContext,
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Execute reasoning and return final result.
        
        Args:
            messages: Conversation history
            context: Execution context with permissions/limits
            tools: Available tool definitions
            **kwargs: Strategy-specific parameters
            
        Returns:
            ReasoningResult with response and metadata
        """
        pass
    
    @abstractmethod
    async def reason_stream(
        self,
        messages: list[Message],
        context: ExecutionContext,
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ReasoningEvent]:
        """
        Execute reasoning with streaming events.
        
        Yields events as reasoning progresses, enabling
        real-time updates to clients.
        """
        pass


class ToolExecutor(ABC):
    """
    Interface for executing tools.
    
    Strategies delegate tool execution to this interface,
    allowing the tool system to handle permissions, tracing, etc.
    """
    
    @abstractmethod
    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute a tool and return the result."""
        pass
    
    @abstractmethod
    def get_tool_definitions(
        self,
        context: ExecutionContext,
    ) -> list[dict[str, Any]]:
        """Get tool definitions allowed for this context."""
        pass
