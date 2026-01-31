"""
Exception Hierarchy

Defines all exceptions used in the Aegis platform.
Exceptions are organized by domain and include context for debugging.

Design decisions:
- All exceptions inherit from AegisError for easy catching
- Exceptions carry structured context, not just messages
- Error codes enable programmatic handling
"""

from typing import Any


class AegisError(Exception):
    """
    Base exception for all Aegis errors.
    
    Provides structured error information including:
    - Human-readable message
    - Machine-readable error code
    - Additional context for debugging
    """
    
    error_code: str = "AEGIS_ERROR"
    
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.error_code
        self.context = context or {}
        self.__cause__ = cause
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize exception for API responses."""
        return {
            "error": self.code,
            "message": self.message,
            "context": self.context,
        }


# ============================================================
# Configuration Errors
# ============================================================

class ConfigurationError(AegisError):
    """Error in configuration or settings."""
    
    error_code = "CONFIGURATION_ERROR"


class SecretNotFoundError(ConfigurationError):
    """Required secret not found."""
    
    error_code = "SECRET_NOT_FOUND"


# ============================================================
# LLM / Reasoning Errors
# ============================================================

class LLMError(AegisError):
    """Base error for LLM-related issues."""
    
    error_code = "LLM_ERROR"


class LLMConnectionError(LLMError):
    """Failed to connect to LLM provider."""
    
    error_code = "LLM_CONNECTION_ERROR"


class LLMRateLimitError(LLMError):
    """Rate limit exceeded for LLM provider."""
    
    error_code = "LLM_RATE_LIMIT"
    
    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    """LLM request timed out."""
    
    error_code = "LLM_TIMEOUT"


class LLMResponseError(LLMError):
    """Invalid or unexpected response from LLM."""
    
    error_code = "LLM_RESPONSE_ERROR"


class ContextLengthExceededError(LLMError):
    """Input exceeds model's context length."""
    
    error_code = "CONTEXT_LENGTH_EXCEEDED"
    
    def __init__(
        self,
        message: str,
        *,
        max_tokens: int,
        actual_tokens: int,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens


# ============================================================
# Memory Errors
# ============================================================

class MemoryError(AegisError):
    """Base error for memory-related issues."""
    
    error_code = "MEMORY_ERROR"


class SessionNotFoundError(MemoryError):
    """Session does not exist."""
    
    error_code = "SESSION_NOT_FOUND"


class MemoryConnectionError(MemoryError):
    """Failed to connect to memory store."""
    
    error_code = "MEMORY_CONNECTION_ERROR"


# ============================================================
# Tool Errors
# ============================================================

class ToolError(AegisError):
    """Base error for tool-related issues."""
    
    error_code = "TOOL_ERROR"


class ToolNotFoundError(ToolError):
    """Requested tool does not exist."""
    
    error_code = "TOOL_NOT_FOUND"


class ToolExecutionError(ToolError):
    """Error during tool execution."""
    
    error_code = "TOOL_EXECUTION_ERROR"
    
    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class ToolPermissionError(ToolError):
    """User lacks permission to use tool."""
    
    error_code = "TOOL_PERMISSION_DENIED"


class ToolValidationError(ToolError):
    """Tool arguments failed validation."""
    
    error_code = "TOOL_VALIDATION_ERROR"


# ============================================================
# Knowledge / RAG Errors
# ============================================================

class KnowledgeError(AegisError):
    """Base error for knowledge/RAG issues."""
    
    error_code = "KNOWLEDGE_ERROR"


class DocumentIngestionError(KnowledgeError):
    """Error during document ingestion."""
    
    error_code = "DOCUMENT_INGESTION_ERROR"


class EmbeddingError(KnowledgeError):
    """Error generating embeddings."""
    
    error_code = "EMBEDDING_ERROR"


class VectorStoreError(KnowledgeError):
    """Error with vector store operations."""
    
    error_code = "VECTOR_STORE_ERROR"


# ============================================================
# Planning / Orchestration Errors
# ============================================================

class PlanningError(AegisError):
    """Base error for planning issues."""
    
    error_code = "PLANNING_ERROR"


class TaskDecompositionError(PlanningError):
    """Failed to decompose task."""
    
    error_code = "TASK_DECOMPOSITION_ERROR"


class CheckpointError(PlanningError):
    """Error with checkpoint operations."""
    
    error_code = "CHECKPOINT_ERROR"


class RollbackError(PlanningError):
    """Failed to rollback to checkpoint."""
    
    error_code = "ROLLBACK_ERROR"


# ============================================================
# Safety / Governance Errors
# ============================================================

class SafetyError(AegisError):
    """Base error for safety violations."""
    
    error_code = "SAFETY_ERROR"


class PromptInjectionError(SafetyError):
    """Potential prompt injection detected."""
    
    error_code = "PROMPT_INJECTION_DETECTED"


class ContentViolationError(SafetyError):
    """Content violates safety policies."""
    
    error_code = "CONTENT_VIOLATION"


class RateLimitError(SafetyError):
    """Rate limit exceeded."""
    
    error_code = "RATE_LIMIT_EXCEEDED"


class AccessDeniedError(SafetyError):
    """Access denied by RBAC."""
    
    error_code = "ACCESS_DENIED"


# ============================================================
# Agent Execution Errors
# ============================================================

class ExecutionError(AegisError):
    """Base error for agent execution issues."""
    
    error_code = "EXECUTION_ERROR"


class ExecutionTimeoutError(ExecutionError):
    """Agent execution timed out."""
    
    error_code = "EXECUTION_TIMEOUT"


class MaxIterationsExceededError(ExecutionError):
    """Agent exceeded maximum iterations."""
    
    error_code = "MAX_ITERATIONS_EXCEEDED"


class CancellationError(ExecutionError):
    """Execution was cancelled."""
    
    error_code = "EXECUTION_CANCELLED"
