"""
Core Interfaces and Protocols

Defines the contracts between modules to prevent circular dependencies.
All cross-module interactions should use these interfaces.

Design decisions:
- Protocol-based for structural subtyping
- Minimal interface surface
- No implementation details leak through
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Protocol, runtime_checkable
from collections.abc import Awaitable
from dataclasses import dataclass

# Forward references to avoid circular imports
# Actual types are in core/types.py
from src.core.types import (
    ExecutionContext,
    Message,
    ToolCall,
    ToolResult,
    LLMResponse,
)


# =============================================================================
# LLM ADAPTER PROTOCOL
# =============================================================================

@runtime_checkable
class LLMAdapterProtocol(Protocol):
    """
    Interface for LLM providers.
    
    Implemented by: OpenAIAdapter, AnthropicAdapter, etc.
    Used by: ReasoningStrategy, AgentRuntime
    """
    
    @property
    def model(self) -> str:
        """Current model identifier."""
        ...
    
    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion."""
        ...
    
    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """Generate a streaming completion."""
        ...


# =============================================================================
# TOOL EXECUTOR PROTOCOL
# =============================================================================

@runtime_checkable
class ToolExecutorProtocol(Protocol):
    """
    Interface for tool execution.
    
    Implemented by: ToolExecutor, SecureToolExecutor
    Used by: ReasoningStrategy, AgentRuntime
    """
    
    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute a tool by name with arguments."""
        ...
    
    def get_tool_definitions(
        self,
        context: ExecutionContext,
    ) -> list[dict[str, Any]]:
        """Get available tool definitions for the context."""
        ...


# =============================================================================
# MEMORY PROTOCOL
# =============================================================================

@runtime_checkable
class MemoryProtocol(Protocol):
    """
    Interface for memory management.
    
    Implemented by: MemoryManager
    Used by: AgentRuntime
    """
    
    async def get_context(
        self,
        session_id: str,
        query: str | None = None,
        max_messages: int = 20,
    ) -> list[Message]:
        """Retrieve conversation context."""
        ...
    
    async def add_messages(
        self,
        session_id: str,
        messages: list[Message],
    ) -> None:
        """Add messages to memory."""
        ...
    
    async def summarize_if_needed(
        self,
        session_id: str,
        threshold: int = 50,
    ) -> bool:
        """Summarize old messages if threshold exceeded."""
        ...


# =============================================================================
# RETRIEVER PROTOCOL
# =============================================================================

@dataclass
class RetrievedContext:
    """Context retrieved from knowledge base."""
    
    content: str
    source: str | None = None
    score: float = 0.0
    metadata: dict[str, Any] | None = None


@runtime_checkable
class RetrieverProtocol(Protocol):
    """
    Interface for knowledge retrieval (RAG).
    
    Implemented by: Retriever
    Used by: AgentRuntime
    """
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievedContext]:
        """Retrieve relevant documents for query."""
        ...


# =============================================================================
# INPUT VALIDATOR PROTOCOL
# =============================================================================

@dataclass
class ValidationResult:
    """Result of input validation."""
    
    is_valid: bool
    issues: list[str] | None = None
    sanitized_input: str | None = None
    risk_score: float = 0.0


@runtime_checkable
class InputValidatorProtocol(Protocol):
    """
    Interface for input validation.
    
    Implemented by: InputValidator
    Used by: AgentRuntime
    """
    
    async def validate(
        self,
        text: str,
        context: ExecutionContext | None = None,
    ) -> ValidationResult:
        """Validate and sanitize input."""
        ...


# =============================================================================
# GUARDRAIL PROTOCOL
# =============================================================================

@dataclass
class GuardrailResult:
    """Result of guardrail check."""
    
    passed: bool
    blocked: bool = False
    reason: str | None = None
    modified_content: str | None = None


@runtime_checkable
class GuardrailProtocol(Protocol):
    """
    Interface for output guardrails.
    
    Implemented by: GuardrailChain
    Used by: AgentRuntime
    """
    
    async def check(
        self,
        content: str,
        context: ExecutionContext | None = None,
    ) -> GuardrailResult:
        """Check content against guardrails."""
        ...


# =============================================================================
# TRACER PROTOCOL
# =============================================================================

@runtime_checkable
class TracerProtocol(Protocol):
    """
    Interface for distributed tracing.
    
    Implemented by: Tracer
    Used by: AgentRuntime, middleware
    """
    
    def start_span(
        self,
        name: str,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> "SpanProtocol":
        """Start a new span."""
        ...


@runtime_checkable
class SpanProtocol(Protocol):
    """Interface for a trace span."""
    
    @property
    def span_id(self) -> str:
        """Unique span identifier."""
        ...
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        ...
    
    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add event to span."""
        ...
    
    def set_status(self, status: str, message: str | None = None) -> None:
        """Set span status."""
        ...
    
    def end(self) -> None:
        """End the span."""
        ...


# =============================================================================
# SESSION PROTOCOL
# =============================================================================

@runtime_checkable
class SessionProtocol(Protocol):
    """
    Interface for session management.
    
    Implemented by: SessionManager
    Used by: API routes
    """
    
    async def get(self, session_id: str) -> Any | None:
        """Get session by ID."""
        ...
    
    async def create(
        self,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create new session."""
        ...
    
    async def save(self, session: Any) -> None:
        """Save session state."""
        ...
    
    async def delete(self, session_id: str) -> bool:
        """Delete session."""
        ...
