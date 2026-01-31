"""
Core Module

Contains fundamental types, exceptions, and utilities used across
all other modules in the Aegis platform.
"""

from src.core.types import (
    AgentState,
    ExecutionContext,
    LLMResponse,
    Message,
    MessageRole,
    RetrievedDocument,
    Task,
    TaskStatus,
    ToolCall,
    ToolResult,
)
from src.core.exceptions import (
    AegisError,
    ConfigurationError,
    ExecutionError,
    LLMError,
    MemoryError,
    PlanningError,
    SafetyError,
    ToolError,
)

__all__ = [
    # Types
    "AgentState",
    "ExecutionContext",
    "LLMResponse",
    "Message",
    "MessageRole",
    "RetrievedDocument",
    "Task",
    "TaskStatus",
    "ToolCall",
    "ToolResult",
    # Exceptions
    "AegisError",
    "ConfigurationError",
    "ExecutionError",
    "LLMError",
    "MemoryError",
    "PlanningError",
    "SafetyError",
    "ToolError",
]
