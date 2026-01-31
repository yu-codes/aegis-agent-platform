"""
Tool Invocation Tracing

Records and tracks tool invocations for observability.

Design decisions:
- Full invocation recording
- Performance metrics
- Error tracking
- Integration with observability system
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from src.core.types import ExecutionContext


@dataclass
class ToolInvocation:
    """
    Record of a single tool invocation.
    
    Contains all information needed for debugging,
    auditing, and performance analysis.
    """
    
    id: UUID = field(default_factory=uuid4)
    
    # Tool information
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    
    # Result
    result: Any = None
    error: str | None = None
    is_success: bool = True
    
    # Performance
    duration_ms: float = 0.0
    
    # Context
    session_id: UUID | None = None
    execution_id: UUID | None = None
    user_id: str | None = None
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    
    # Tracing
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None


class ToolTracer:
    """
    Traces tool invocations.
    
    Records invocations and provides querying capabilities
    for debugging and observability.
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        enable_argument_logging: bool = True,
        enable_result_logging: bool = True,
    ):
        self._history: list[ToolInvocation] = []
        self._max_history = max_history
        self._log_args = enable_argument_logging
        self._log_results = enable_result_logging
        
        # Callbacks for external systems
        self._on_invocation: list[Any] = []
    
    def add_callback(self, callback) -> None:
        """Add a callback for invocation events."""
        self._on_invocation.append(callback)
    
    async def record_invocation(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        duration_ms: float,
        context: ExecutionContext,
    ) -> ToolInvocation:
        """Record a successful invocation."""
        invocation = ToolInvocation(
            tool_name=tool_name,
            arguments=arguments if self._log_args else {},
            result=result if self._log_results else "[logged]",
            error=None,
            is_success=True,
            duration_ms=duration_ms,
            session_id=context.session_id,
            execution_id=context.execution_id,
            user_id=context.user_id,
            completed_at=datetime.utcnow(),
            trace_id=context.trace_id,
            parent_span_id=context.parent_span_id,
        )
        
        self._add_to_history(invocation)
        await self._notify_callbacks(invocation)
        
        return invocation
    
    async def record_error(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        error: str,
        context: ExecutionContext,
    ) -> ToolInvocation:
        """Record a failed invocation."""
        invocation = ToolInvocation(
            tool_name=tool_name,
            arguments=arguments if self._log_args else {},
            result=None,
            error=error,
            is_success=False,
            duration_ms=0,
            session_id=context.session_id,
            execution_id=context.execution_id,
            user_id=context.user_id,
            completed_at=datetime.utcnow(),
            trace_id=context.trace_id,
            parent_span_id=context.parent_span_id,
        )
        
        self._add_to_history(invocation)
        await self._notify_callbacks(invocation)
        
        return invocation
    
    def _add_to_history(self, invocation: ToolInvocation) -> None:
        """Add invocation to history, maintaining size limit."""
        self._history.append(invocation)
        
        # Trim if over limit
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
    
    async def _notify_callbacks(self, invocation: ToolInvocation) -> None:
        """Notify registered callbacks."""
        for callback in self._on_invocation:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(invocation)
                else:
                    callback(invocation)
            except Exception:
                pass  # Don't let callback errors affect tracing
    
    def get_history(
        self,
        tool_name: str | None = None,
        session_id: UUID | None = None,
        user_id: str | None = None,
        success_only: bool = False,
        limit: int = 100,
    ) -> list[ToolInvocation]:
        """Query invocation history."""
        results = self._history
        
        if tool_name:
            results = [i for i in results if i.tool_name == tool_name]
        
        if session_id:
            results = [i for i in results if i.session_id == session_id]
        
        if user_id:
            results = [i for i in results if i.user_id == user_id]
        
        if success_only:
            results = [i for i in results if i.is_success]
        
        return results[-limit:]
    
    def get_statistics(
        self,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        """Get statistics for tool invocations."""
        invocations = self._history
        
        if tool_name:
            invocations = [i for i in invocations if i.tool_name == tool_name]
        
        if not invocations:
            return {}
        
        success_count = sum(1 for i in invocations if i.is_success)
        error_count = len(invocations) - success_count
        durations = [i.duration_ms for i in invocations if i.is_success]
        
        return {
            "total_invocations": len(invocations),
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / len(invocations) if invocations else 0,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
        }
    
    def get_error_summary(self) -> dict[str, list[str]]:
        """Get summary of errors by tool."""
        errors: dict[str, list[str]] = {}
        
        for inv in self._history:
            if inv.error:
                if inv.tool_name not in errors:
                    errors[inv.tool_name] = []
                errors[inv.tool_name].append(inv.error)
        
        return errors
    
    def clear_history(self) -> None:
        """Clear invocation history."""
        self._history.clear()


# Avoid import at top to prevent circular import
import asyncio
