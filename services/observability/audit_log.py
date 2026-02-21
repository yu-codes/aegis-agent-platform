"""
Audit Log

Audit logging for compliance and security.

Based on: src/safety/audit.py
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID, uuid4


class AuditEventType:
    """Audit event types."""

    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    AGENT_RUN = "agent.run"
    TOOL_CALL = "tool.call"
    POLICY_VIOLATION = "policy.violation"
    PERMISSION_DENIED = "permission.denied"
    DATA_ACCESS = "data.access"
    CONFIG_CHANGE = "config.change"
    ERROR = "error"


@dataclass
class AuditEntry:
    """An audit log entry."""

    id: UUID = field(default_factory=uuid4)
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Actor
    user_id: str | None = None
    session_id: str | None = None
    ip_address: str | None = None

    # Resource
    resource_type: str | None = None
    resource_id: str | None = None

    # Action details
    action: str = ""
    outcome: str = "success"  # success, failure, error

    # Context
    details: dict[str, Any] = field(default_factory=dict)

    # Request context
    trace_id: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "outcome": self.outcome,
            "details": self.details,
            "trace_id": self.trace_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AuditStorageProtocol(Protocol):
    """Protocol for audit storage backends."""

    async def store(self, entry: AuditEntry) -> None: ...
    async def query(self, filters: dict, limit: int) -> list[AuditEntry]: ...


class AuditLog:
    """
    Audit logging service.

    Records security and compliance events.
    """

    def __init__(
        self,
        storage: AuditStorageProtocol | None = None,
        enabled: bool = True,
    ):
        self._storage = storage
        self._enabled = enabled

        # In-memory buffer for testing
        self._buffer: list[AuditEntry] = []
        self._buffer_size = 1000

    async def log(
        self,
        event_type: str,
        action: str,
        user_id: str | None = None,
        session_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        outcome: str = "success",
        details: dict | None = None,
        ip_address: str | None = None,
    ) -> AuditEntry:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            action: Action description
            user_id: User ID
            session_id: Session ID
            resource_type: Type of resource accessed
            resource_id: Resource ID
            outcome: Outcome (success, failure, error)
            details: Additional details
            ip_address: Client IP address

        Returns:
            Created audit entry
        """
        if not self._enabled:
            return AuditEntry()

        # Get trace ID from current span
        trace_id = None
        try:
            from services.observability.tracing import _current_span

            span = _current_span.get()
            if span:
                trace_id = str(span.context.trace_id)
        except Exception:
            pass

        entry = AuditEntry(
            event_type=event_type,
            action=action,
            user_id=user_id,
            session_id=session_id,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            details=details or {},
            ip_address=ip_address,
            trace_id=trace_id,
        )

        # Store entry
        if self._storage:
            await self._storage.store(entry)
        else:
            self._buffer.append(entry)
            if len(self._buffer) > self._buffer_size:
                self._buffer = self._buffer[-self._buffer_size :]

        return entry

    async def log_agent_run(
        self,
        session_id: str,
        user_id: str | None,
        input_message: str,
        output: str,
        tools_used: list[str],
        duration_ms: float,
        success: bool = True,
    ) -> AuditEntry:
        """Log an agent run."""
        return await self.log(
            event_type=AuditEventType.AGENT_RUN,
            action="execute",
            user_id=user_id,
            session_id=session_id,
            outcome="success" if success else "failure",
            details={
                "input_length": len(input_message),
                "output_length": len(output),
                "tools_used": tools_used,
                "duration_ms": duration_ms,
            },
        )

    async def log_tool_call(
        self,
        tool_name: str,
        user_id: str | None,
        session_id: str | None,
        arguments: dict,
        result: Any,
        success: bool = True,
    ) -> AuditEntry:
        """Log a tool call."""
        return await self.log(
            event_type=AuditEventType.TOOL_CALL,
            action=f"call:{tool_name}",
            user_id=user_id,
            session_id=session_id,
            resource_type="tool",
            resource_id=tool_name,
            outcome="success" if success else "failure",
            details={
                "arguments": self._sanitize_arguments(arguments),
                "result_type": type(result).__name__,
            },
        )

    async def log_policy_violation(
        self,
        policy_id: str,
        policy_name: str,
        user_id: str | None,
        session_id: str | None,
        content_snippet: str | None = None,
    ) -> AuditEntry:
        """Log a policy violation."""
        return await self.log(
            event_type=AuditEventType.POLICY_VIOLATION,
            action=f"violation:{policy_id}",
            user_id=user_id,
            session_id=session_id,
            resource_type="policy",
            resource_id=policy_id,
            outcome="blocked",
            details={
                "policy_name": policy_name,
                "content_snippet": content_snippet[:100] if content_snippet else None,
            },
        )

    async def log_permission_denied(
        self,
        user_id: str,
        resource: str,
        action: str,
        required_permission: str,
    ) -> AuditEntry:
        """Log a permission denied event."""
        return await self.log(
            event_type=AuditEventType.PERMISSION_DENIED,
            action=action,
            user_id=user_id,
            resource_type="permission",
            resource_id=resource,
            outcome="denied",
            details={
                "required_permission": required_permission,
            },
        )

    async def query(
        self,
        event_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """
        Query audit logs.

        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            session_id: Filter by session ID
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum results

        Returns:
            List of matching audit entries
        """
        if self._storage:
            filters = {
                "event_type": event_type,
                "user_id": user_id,
                "session_id": session_id,
                "start_time": start_time,
                "end_time": end_time,
            }
            return await self._storage.query(filters, limit)

        # Filter in-memory buffer
        results = []
        for entry in reversed(self._buffer):
            if event_type and entry.event_type != event_type:
                continue
            if user_id and entry.user_id != user_id:
                continue
            if session_id and entry.session_id != session_id:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue

            results.append(entry)
            if len(results) >= limit:
                break

        return results

    def _sanitize_arguments(self, arguments: dict) -> dict:
        """Sanitize arguments for logging."""
        sanitized = {}
        sensitive_keys = {"password", "secret", "token", "api_key", "credentials"}

        for key, value in arguments.items():
            if any(s in key.lower() for s in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:1000] + "..."
            else:
                sanitized[key] = value

        return sanitized

    def enable(self) -> None:
        """Enable audit logging."""
        self._enabled = True

    def disable(self) -> None:
        """Disable audit logging."""
        self._enabled = False

    def get_buffer(self) -> list[AuditEntry]:
        """Get in-memory buffer (for testing)."""
        return list(self._buffer)

    def clear_buffer(self) -> None:
        """Clear in-memory buffer."""
        self._buffer.clear()
