"""
Tool Executor

Safe tool execution with timeout, retry, and error handling.

Based on: src/tools/executor.py
"""

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol
from uuid import UUID, uuid4


@dataclass
class ExecutionResult:
    """Result of a tool execution."""

    id: UUID = field(default_factory=uuid4)
    tool_name: str = ""
    success: bool = False
    result: Any = None
    error: str | None = None

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    duration_ms: float = 0.0

    # Retry info
    attempts: int = 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "attempts": self.attempts,
        }


class ToolRegistryProtocol(Protocol):
    """Protocol for tool registry."""

    def get(self, name: str) -> Any: ...


class ToolExecutor:
    """
    Executes tools safely.

    Features:
    - Timeout handling
    - Retry logic
    - Error capture
    - Rate limiting
    - Sandboxing
    """

    def __init__(
        self,
        registry: ToolRegistryProtocol | None = None,
        default_timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self._registry = registry
        self._default_timeout = default_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Rate limiting state
        self._call_times: dict[str, list[float]] = {}

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
        retry: bool = True,
    ) -> ExecutionResult:
        """
        Execute a tool.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            timeout: Execution timeout
            retry: Whether to retry on failure

        Returns:
            Execution result
        """
        result = ExecutionResult(tool_name=tool_name)
        timeout = timeout or self._default_timeout

        # Get tool definition
        tool_def = self._registry.get(tool_name) if self._registry else None

        if tool_def is None:
            result.error = f"Tool not found: {tool_name}"
            result.completed_at = datetime.utcnow()
            return result

        # Check rate limit
        if tool_def.rate_limit:
            if not self._check_rate_limit(tool_name, tool_def.rate_limit):
                result.error = f"Rate limit exceeded for tool: {tool_name}"
                result.completed_at = datetime.utcnow()
                return result

        # Execute with retries
        handler = tool_def.handler
        is_async = tool_def.is_async
        max_attempts = self._max_retries if retry else 1

        for attempt in range(1, max_attempts + 1):
            result.attempts = attempt

            try:
                start_time = time.time()

                if is_async:
                    output = await asyncio.wait_for(
                        handler(**arguments),
                        timeout=timeout,
                    )
                else:
                    output = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: handler(**arguments)
                        ),
                        timeout=timeout,
                    )

                result.success = True
                result.result = output
                result.duration_ms = (time.time() - start_time) * 1000
                result.completed_at = datetime.utcnow()

                # Record successful call for rate limiting
                self._record_call(tool_name)

                return result

            except asyncio.TimeoutError:
                result.error = f"Tool execution timed out after {timeout}s"
            except Exception as e:
                result.error = f"{type(e).__name__}: {str(e)}"
                if attempt < max_attempts:
                    await asyncio.sleep(self._retry_delay * attempt)

        result.completed_at = datetime.utcnow()
        result.duration_ms = (time.time() - start_time) * 1000

        return result

    async def execute_batch(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        parallel: bool = False,
    ) -> list[ExecutionResult]:
        """
        Execute multiple tools.

        Args:
            calls: List of (tool_name, arguments) tuples
            parallel: Execute in parallel

        Returns:
            List of execution results
        """
        if parallel:
            tasks = [self.execute(name, args) for name, args in calls]
            return await asyncio.gather(*tasks)

        results = []
        for name, args in calls:
            result = await self.execute(name, args)
            results.append(result)
        return results

    def _check_rate_limit(self, tool_name: str, limit: int) -> bool:
        """Check if tool is within rate limit."""
        now = time.time()
        window = 60.0  # 1 minute window

        if tool_name not in self._call_times:
            return True

        # Clean old entries
        self._call_times[tool_name] = [t for t in self._call_times[tool_name] if now - t < window]

        return len(self._call_times[tool_name]) < limit

    def _record_call(self, tool_name: str) -> None:
        """Record a tool call for rate limiting."""
        if tool_name not in self._call_times:
            self._call_times[tool_name] = []
        self._call_times[tool_name].append(time.time())

    async def validate_and_execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        validator: "ToolValidatorProtocol | None" = None,
    ) -> ExecutionResult:
        """
        Validate arguments then execute.

        Args:
            tool_name: Tool name
            arguments: Tool arguments
            validator: Optional validator

        Returns:
            Execution result
        """
        if validator:
            validation = await validator.validate(tool_name, arguments)
            if not validation.valid:
                result = ExecutionResult(tool_name=tool_name)
                result.error = f"Validation failed: {validation.error}"
                result.completed_at = datetime.utcnow()
                return result

        return await self.execute(tool_name, arguments)


class ToolValidatorProtocol(Protocol):
    """Protocol for tool validation."""

    async def validate(self, tool_name: str, arguments: dict) -> Any: ...
