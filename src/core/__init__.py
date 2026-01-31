"""
Core Module

Contains fundamental types, exceptions, interfaces, and utilities used across
all other modules in the Aegis platform.

The interfaces module defines protocols for cross-module communication,
preventing circular dependencies.
"""

from src.core.exceptions import (
    AegisError,
    ConfigurationError,
    ExecutionError,
    LLMError,
    MaxIterationsExceededError,
    MemoryError,
    PlanningError,
    SafetyError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolValidationError,
)
from src.core.interfaces import (
    GuardrailProtocol,
    GuardrailResult,
    InputValidatorProtocol,
    LLMAdapterProtocol,
    MemoryProtocol,
    RetrievedContext,
    RetrieverProtocol,
    SessionProtocol,
    SpanProtocol,
    ToolExecutorProtocol,
    TracerProtocol,
    ValidationResult,
)
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

__all__ = [
    # Exceptions
    "AegisError",
    # Types
    "AgentState",
    "ConfigurationError",
    "ExecutionContext",
    "ExecutionError",
    "GuardrailProtocol",
    "GuardrailResult",
    "InputValidatorProtocol",
    # Interfaces/Protocols
    "LLMAdapterProtocol",
    "LLMError",
    "LLMResponse",
    "MaxIterationsExceededError",
    "MemoryError",
    "MemoryProtocol",
    "Message",
    "MessageRole",
    "PlanningError",
    "RetrievedContext",
    "RetrievedDocument",
    "RetrieverProtocol",
    "SafetyError",
    "SessionProtocol",
    "SpanProtocol",
    "Task",
    "TaskStatus",
    "ToolCall",
    "ToolError",
    "ToolExecutionError",
    "ToolExecutorProtocol",
    "ToolNotFoundError",
    "ToolPermissionError",
    "ToolResult",
    "ToolValidationError",
    "TracerProtocol",
    "ValidationResult",
]
