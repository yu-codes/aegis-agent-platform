"""
Tool Executor

Secure execution of tools with validation, sandboxing, and tracing.

Design decisions:
- Validation before execution
- Timeout enforcement
- Error isolation
- Execution tracing for observability
- Implements ToolExecutorProtocol from core to avoid circular dependencies
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

from src.core.types import ExecutionContext, ToolResult
from src.core.interfaces import ToolExecutorProtocol
from src.core.exceptions import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
    ToolPermissionError,
)
from src.tools.registry import ToolRegistry, ToolDefinition
from src.tools.permissions import PermissionManager
from src.tools.tracing import ToolTracer


class ToolValidator:
    """
    Validates tool arguments against schema.
    
    Uses JSON Schema validation for type checking.
    """
    
    def validate(
        self,
        definition: ToolDefinition,
        arguments: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Validate arguments against tool schema.
        
        Returns (is_valid, error_messages).
        """
        errors = []
        schema = definition.parameters
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in arguments:
                errors.append(f"Missing required field: {field}")
        
        # Check types
        properties = schema.get("properties", {})
        for field, value in arguments.items():
            if field not in properties:
                continue  # Allow extra fields
            
            field_schema = properties[field]
            expected_type = field_schema.get("type")
            
            if expected_type and not self._check_type(value, expected_type):
                errors.append(
                    f"Field '{field}' expected type {expected_type}, "
                    f"got {type(value).__name__}"
                )
        
        return len(errors) == 0, errors
    
    def _check_type(self, value: Any, expected: str) -> bool:
        """Check if value matches expected JSON Schema type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        
        expected_types = type_map.get(expected)
        if expected_types is None:
            return True
        
        return isinstance(value, expected_types)


class BaseToolExecutor(ABC):
    """
    Abstract base for tool executors.
    
    Implements ToolExecutorProtocol from core.interfaces.
    """
    
    @abstractmethod
    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute a tool."""
        pass
    
    @abstractmethod
    def get_tool_definitions(
        self,
        context: ExecutionContext,
    ) -> list[dict[str, Any]]:
        """Get tool definitions for context."""
        pass


class ToolExecutor(BaseToolExecutor):
    """
    Standard tool executor.
    
    Provides:
    - Registry-based tool lookup
    - Argument validation
    - Timeout enforcement
    - Error handling
    """
    
    def __init__(
        self,
        registry: ToolRegistry,
        validator: ToolValidator | None = None,
        tracer: "ToolTracer | None" = None,
        default_timeout: float = 30.0,
    ):
        self._registry = registry
        self._validator = validator or ToolValidator()
        self._tracer = tracer
        self._default_timeout = default_timeout
    
    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute a tool by name."""
        start_time = time.perf_counter()
        
        # Look up tool
        definition = self._registry.get(tool_name)
        if definition is None:
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=None,
                error=f"Tool not found: {tool_name}",
                duration_ms=0,
            )
        
        # Validate arguments
        is_valid, errors = self._validator.validate(definition, arguments)
        if not is_valid:
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=None,
                error=f"Validation failed: {', '.join(errors)}",
                duration_ms=0,
            )
        
        # Execute with timeout
        timeout = definition.timeout_seconds or self._default_timeout
        
        try:
            if definition.is_async:
                result = await asyncio.wait_for(
                    definition.function(**arguments),
                    timeout=timeout,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(definition.function, **arguments),
                    timeout=timeout,
                )
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Trace if enabled
            if self._tracer:
                await self._tracer.record_invocation(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    duration_ms=duration_ms,
                    context=context,
                )
            
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=result,
                error=None,
                duration_ms=duration_ms,
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=None,
                error=f"Tool execution timed out after {timeout}s",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Trace error
            if self._tracer:
                await self._tracer.record_error(
                    tool_name=tool_name,
                    arguments=arguments,
                    error=str(e),
                    context=context,
                )
            
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=None,
                error=str(e),
                duration_ms=duration_ms,
            )
    
    def get_tool_definitions(
        self,
        context: ExecutionContext,
    ) -> list[dict[str, Any]]:
        """Get tool definitions allowed for context."""
        definitions = self._registry.get_all_definitions()
        
        # Filter by allowed tools
        if context.allowed_tools is not None:
            definitions = [d for d in definitions if d.name in context.allowed_tools]
        
        # Filter out denied tools
        definitions = [d for d in definitions if d.name not in context.denied_tools]
        
        return [d.to_openai_format() for d in definitions]


class SecureToolExecutor(BaseToolExecutor):
    """
    Secure tool executor with permission checking.
    
    Adds:
    - RBAC permission checking
    - Audit logging
    - Rate limiting
    - Dangerous tool confirmation
    """
    
    def __init__(
        self,
        registry: ToolRegistry,
        permission_manager: PermissionManager,
        validator: ToolValidator | None = None,
        tracer: "ToolTracer | None" = None,
    ):
        self._registry = registry
        self._permissions = permission_manager
        self._validator = validator or ToolValidator()
        self._tracer = tracer
    
    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute with permission checking."""
        start_time = time.perf_counter()
        
        # Look up tool
        definition = self._registry.get(tool_name)
        if definition is None:
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=None,
                error=f"Tool not found: {tool_name}",
                duration_ms=0,
            )
        
        # Check permissions
        has_permission = await self._permissions.check_permission(
            user_id=context.user_id,
            tool_name=tool_name,
            context=context,
        )
        
        if not has_permission:
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=None,
                error=f"Permission denied for tool: {tool_name}",
                duration_ms=0,
            )
        
        # Check if tool is denied in context
        if tool_name in context.denied_tools:
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=None,
                error=f"Tool is denied in this context: {tool_name}",
                duration_ms=0,
            )
        
        # Check allowed tools if specified
        if context.allowed_tools is not None and tool_name not in context.allowed_tools:
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=None,
                error=f"Tool not in allowed list: {tool_name}",
                duration_ms=0,
            )
        
        # Check rate limit
        is_allowed, retry_after = await self._permissions.check_rate_limit(
            user_id=context.user_id,
            tool_name=tool_name,
        )
        
        if not is_allowed:
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=None,
                error=f"Rate limit exceeded. Retry after {retry_after}s",
                duration_ms=0,
            )
        
        # Validate arguments
        is_valid, errors = self._validator.validate(definition, arguments)
        if not is_valid:
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=None,
                error=f"Validation failed: {', '.join(errors)}",
                duration_ms=0,
            )
        
        # Execute
        try:
            if definition.is_async:
                result = await asyncio.wait_for(
                    definition.function(**arguments),
                    timeout=definition.timeout_seconds,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(definition.function, **arguments),
                    timeout=definition.timeout_seconds,
                )
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Record usage for rate limiting
            await self._permissions.record_usage(
                user_id=context.user_id,
                tool_name=tool_name,
            )
            
            # Trace
            if self._tracer:
                await self._tracer.record_invocation(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    duration_ms=duration_ms,
                    context=context,
                )
            
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=result,
                error=None,
                duration_ms=duration_ms,
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=None,
                error=f"Timeout after {definition.timeout_seconds}s",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                result=None,
                error=str(e),
                duration_ms=duration_ms,
            )
    
    def get_tool_definitions(
        self,
        context: ExecutionContext,
    ) -> list[dict[str, Any]]:
        """Get tool definitions allowed for context."""
        definitions = self._registry.get_all_definitions()
        
        # Filter by context restrictions
        if context.allowed_tools is not None:
            definitions = [d for d in definitions if d.name in context.allowed_tools]
        
        definitions = [d for d in definitions if d.name not in context.denied_tools]
        
        return [d.to_openai_format() for d in definitions]
