"""
Core Types and Data Structures

Defines the fundamental types used throughout the Aegis platform.
These are intentionally simple, immutable where possible, and serializable.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"  # Legacy, maps to TOOL


class Message(BaseModel):
    """
    A single message in a conversation.

    Immutable by design. Create new messages rather than modifying.
    """

    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    name: str | None = None  # For tool/function messages
    tool_call_id: str | None = None  # Reference to tool invocation
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True


class ToolCall(BaseModel):
    """Represents a tool invocation request from the LLM."""

    id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:8]}")
    name: str
    arguments: dict[str, Any]

    class Config:
        frozen = True


class ToolResult(BaseModel):
    """Result from executing a tool."""

    tool_call_id: str
    name: str
    result: Any
    error: str | None = None
    duration_ms: float = 0.0

    @property
    def is_error(self) -> bool:
        return self.error is not None


class AgentState(str, Enum):
    """Lifecycle state of an agent execution."""

    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOL = "executing_tool"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Status of a task in the planning system."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class Task(BaseModel):
    """
    A discrete unit of work in a plan.

    Tasks are the building blocks of multi-step agent operations.
    They form a DAG (Directed Acyclic Graph) via dependencies.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[UUID] = Field(default_factory=list)
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.SKIPPED,
        }

    @property
    def duration_seconds(self) -> float | None:
        """Calculate task duration if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class ExecutionContext(BaseModel):
    """
    Context passed through an agent execution.

    Contains all the information needed for any component
    to understand the current execution state.
    """

    session_id: UUID
    execution_id: UUID = Field(default_factory=uuid4)
    user_id: str | None = None
    tenant_id: str | None = None

    # Permissions and scopes
    allowed_tools: set[str] | None = None  # None means all allowed
    denied_tools: set[str] = Field(default_factory=set)
    max_tool_calls: int = 50
    max_tokens: int = 100000

    # Execution limits
    timeout_seconds: float = 300.0
    max_retries: int = 3

    # Tracing
    trace_id: str | None = None
    parent_span_id: str | None = None

    # Feature flags
    enable_streaming: bool = True
    enable_memory: bool = True
    enable_rag: bool = True

    # Custom metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def with_span(self, span_id: str) -> "ExecutionContext":
        """Create a new context with updated span ID."""
        return self.model_copy(update={"parent_span_id": span_id})


class LLMResponse(BaseModel):
    """
    Response from an LLM call.

    Normalized across different providers.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0

    # Metadata
    model: str
    finish_reason: str | None = None
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class RetrievedDocument(BaseModel):
    """A document retrieved from the knowledge base."""

    id: str
    content: str
    score: float  # Similarity score
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    chunk_index: int | None = None
