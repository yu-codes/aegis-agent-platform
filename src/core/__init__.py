"""
Core Module

Contains fundamental types, exceptions, interfaces, and utilities used across
all other modules in the Aegis platform.

The interfaces module defines protocols for cross-module communication,
preventing circular dependencies.
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
    MaxIterationsExceededError,
    PlanningError,
    SafetyError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolValidationError,
)
from src.core.interfaces import (
    LLMAdapterProtocol,
    ToolExecutorProtocol,
    MemoryProtocol,
    RetrieverProtocol,
    InputValidatorProtocol,
    GuardrailProtocol,
    TracerProtocol,
    SpanProtocol,
    SessionProtocol,
    ValidationResult,
    GuardrailResult,
    RetrievedContext,
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
    "MaxIterationsExceededError",
    "PlanningError",
    "SafetyError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolPermissionError",
    "ToolValidationError",
    # Interfaces/Protocols
    "LLMAdapterProtocol",
    "ToolExecutorProtocol",
    "MemoryProtocol",
    "RetrieverProtocol",
    "InputValidatorProtocol",
    "GuardrailProtocol",
    "TracerProtocol",
    "SpanProtocol",
    "SessionProtocol",
    "ValidationResult",
    "GuardrailResult",
    "RetrievedContext",
]
