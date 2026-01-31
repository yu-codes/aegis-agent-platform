"""
Audit Logging

Records all significant actions for compliance and debugging.

Design decisions:
- Structured log events
- Multiple storage backends
- Query support
- Retention policies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
import json


class AuditEventType(str, Enum):
    """Types of audit events."""
    
    # Authentication
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"
    
    # Session
    SESSION_CREATED = "session.created"
    SESSION_DELETED = "session.deleted"
    
    # Tool usage
    TOOL_INVOKED = "tool.invoked"
    TOOL_FAILED = "tool.failed"
    TOOL_BLOCKED = "tool.blocked"
    
    # Knowledge
    KNOWLEDGE_INDEXED = "knowledge.indexed"
    KNOWLEDGE_QUERIED = "knowledge.queried"
    KNOWLEDGE_DELETED = "knowledge.deleted"
    
    # Safety
    SAFETY_VIOLATION = "safety.violation"
    GUARDRAIL_TRIGGERED = "guardrail.triggered"
    INJECTION_DETECTED = "injection.detected"
    
    # Admin
    ROLE_ASSIGNED = "admin.role_assigned"
    ROLE_REVOKED = "admin.role_revoked"
    CONFIG_CHANGED = "admin.config_changed"
    
    # Model
    MODEL_CALLED = "model.called"
    MODEL_ERROR = "model.error"


@dataclass
class AuditEvent:
    """
    A single audit log entry.
    """
    
    id: UUID = field(default_factory=uuid4)
    
    # Event type
    event_type: AuditEventType = AuditEventType.TOOL_INVOKED
    
    # Actor
    user_id: str | None = None
    session_id: UUID | None = None
    tenant_id: str | None = None
    
    # Target
    resource_type: str | None = None
    resource_id: str | None = None
    
    # Details
    action: str = ""
    result: str = "success"  # success, failure, blocked
    details: dict[str, Any] = field(default_factory=dict)
    
    # Request context
    ip_address: str | None = None
    user_agent: str | None = None
    request_id: str | None = None
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float | None = None
    
    # Tracing
    trace_id: str | None = None
    span_id: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = {
            "id": str(self.id),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "session_id": str(self.session_id) if self.session_id else None,
            "tenant_id": self.tenant_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else uuid4(),
            event_type=AuditEventType(data["event_type"]),
            user_id=data.get("user_id"),
            session_id=UUID(data["session_id"]) if data.get("session_id") else None,
            tenant_id=data.get("tenant_id"),
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            action=data.get("action", ""),
            result=data.get("result", "success"),
            details=data.get("details", {}),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            request_id=data.get("request_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            duration_ms=data.get("duration_ms"),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
        )


class AuditStorage(ABC):
    """Abstract storage for audit logs."""
    
    @abstractmethod
    async def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        pass
    
    @abstractmethod
    async def query(
        self,
        event_type: AuditEventType | None = None,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events."""
        pass
    
    @abstractmethod
    async def cleanup(self, retention_days: int) -> int:
        """Delete events older than retention period."""
        pass


class InMemoryAuditStorage(AuditStorage):
    """In-memory audit storage for development/testing."""
    
    def __init__(self, max_events: int = 10000):
        self._events: list[AuditEvent] = []
        self._max_events = max_events
    
    async def store(self, event: AuditEvent) -> None:
        self._events.append(event)
        
        # Trim if over limit
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
    
    async def query(
        self,
        event_type: AuditEventType | None = None,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        results = self._events
        
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        
        return results[-limit:]
    
    async def cleanup(self, retention_days: int) -> int:
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        original_count = len(self._events)
        self._events = [e for e in self._events if e.timestamp >= cutoff]
        return original_count - len(self._events)


class RedisAuditStorage(AuditStorage):
    """Redis-based audit storage."""
    
    def __init__(
        self,
        redis_client,
        prefix: str = "aegis:audit:",
        max_events: int = 100000,
    ):
        self._redis = redis_client
        self._prefix = prefix
        self._max_events = max_events
    
    async def store(self, event: AuditEvent) -> None:
        # Store event by ID
        event_key = f"{self._prefix}event:{event.id}"
        await self._redis.set(event_key, json.dumps(event.to_dict()))
        
        # Add to sorted set for time-based queries
        await self._redis.zadd(
            f"{self._prefix}timeline",
            {str(event.id): event.timestamp.timestamp()},
        )
        
        # Add to type index
        await self._redis.zadd(
            f"{self._prefix}type:{event.event_type.value}",
            {str(event.id): event.timestamp.timestamp()},
        )
        
        # Add to user index if applicable
        if event.user_id:
            await self._redis.zadd(
                f"{self._prefix}user:{event.user_id}",
                {str(event.id): event.timestamp.timestamp()},
            )
        
        # Trim to max events
        count = await self._redis.zcard(f"{self._prefix}timeline")
        if count > self._max_events:
            # Remove oldest events
            to_remove = count - self._max_events
            old_ids = await self._redis.zrange(
                f"{self._prefix}timeline",
                0, to_remove - 1,
            )
            
            for event_id in old_ids:
                await self._delete_event(event_id)
    
    async def _delete_event(self, event_id: str) -> None:
        """Delete a single event from all indexes."""
        event_key = f"{self._prefix}event:{event_id}"
        data = await self._redis.get(event_key)
        
        if data:
            event_data = json.loads(data)
            
            # Remove from indexes
            await self._redis.zrem(f"{self._prefix}timeline", event_id)
            await self._redis.zrem(
                f"{self._prefix}type:{event_data['event_type']}",
                event_id,
            )
            
            if event_data.get("user_id"):
                await self._redis.zrem(
                    f"{self._prefix}user:{event_data['user_id']}",
                    event_id,
                )
            
            await self._redis.delete(event_key)
    
    async def query(
        self,
        event_type: AuditEventType | None = None,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        # Determine which index to use
        if user_id:
            index_key = f"{self._prefix}user:{user_id}"
        elif event_type:
            index_key = f"{self._prefix}type:{event_type.value}"
        else:
            index_key = f"{self._prefix}timeline"
        
        # Build range query
        min_score = start_time.timestamp() if start_time else "-inf"
        max_score = end_time.timestamp() if end_time else "+inf"
        
        # Get event IDs
        event_ids = await self._redis.zrangebyscore(
            index_key,
            min_score,
            max_score,
            start=0,
            num=limit,
        )
        
        # Fetch events
        events = []
        for event_id in event_ids:
            event_key = f"{self._prefix}event:{event_id}"
            data = await self._redis.get(event_key)
            if data:
                events.append(AuditEvent.from_dict(json.loads(data)))
        
        return events
    
    async def cleanup(self, retention_days: int) -> int:
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        cutoff_score = cutoff.timestamp()
        
        # Get old event IDs
        old_ids = await self._redis.zrangebyscore(
            f"{self._prefix}timeline",
            "-inf",
            cutoff_score,
        )
        
        for event_id in old_ids:
            await self._delete_event(event_id)
        
        return len(old_ids)


class AuditLogger:
    """
    High-level audit logging interface.
    
    Provides convenient methods for logging common events.
    """
    
    def __init__(self, storage: AuditStorage):
        self._storage = storage
    
    async def log(self, event: AuditEvent) -> None:
        """Log an audit event."""
        await self._storage.store(event)
    
    async def log_tool_invocation(
        self,
        tool_name: str,
        user_id: str | None,
        session_id: UUID | None,
        arguments: dict[str, Any],
        result: Any,
        duration_ms: float,
        success: bool = True,
    ) -> None:
        """Log a tool invocation."""
        await self.log(AuditEvent(
            event_type=AuditEventType.TOOL_INVOKED if success else AuditEventType.TOOL_FAILED,
            user_id=user_id,
            session_id=session_id,
            resource_type="tool",
            resource_id=tool_name,
            action="invoke",
            result="success" if success else "failure",
            details={
                "arguments": self._sanitize_details(arguments),
                "result": str(result)[:500] if result else None,
            },
            duration_ms=duration_ms,
        ))
    
    async def log_safety_violation(
        self,
        violation_type: str,
        user_id: str | None,
        content: str,
        details: dict[str, Any],
    ) -> None:
        """Log a safety violation."""
        await self.log(AuditEvent(
            event_type=AuditEventType.SAFETY_VIOLATION,
            user_id=user_id,
            resource_type="safety",
            resource_id=violation_type,
            action="violation",
            result="blocked",
            details={
                "content_preview": content[:200] if content else None,
                **details,
            },
        ))
    
    async def log_model_call(
        self,
        model: str,
        user_id: str | None,
        session_id: UUID | None,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        success: bool = True,
    ) -> None:
        """Log an LLM model call."""
        await self.log(AuditEvent(
            event_type=AuditEventType.MODEL_CALLED if success else AuditEventType.MODEL_ERROR,
            user_id=user_id,
            session_id=session_id,
            resource_type="model",
            resource_id=model,
            action="call",
            result="success" if success else "failure",
            details={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            duration_ms=duration_ms,
        ))
    
    async def log_auth(
        self,
        user_id: str,
        action: str,  # login, logout, failed
        ip_address: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log authentication events."""
        event_type = {
            "login": AuditEventType.AUTH_LOGIN,
            "logout": AuditEventType.AUTH_LOGOUT,
            "failed": AuditEventType.AUTH_FAILED,
        }.get(action, AuditEventType.AUTH_FAILED)
        
        await self.log(AuditEvent(
            event_type=event_type,
            user_id=user_id,
            resource_type="auth",
            action=action,
            result="success" if action != "failed" else "failure",
            details=details or {},
            ip_address=ip_address,
        ))
    
    async def query(
        self,
        event_type: AuditEventType | None = None,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit logs."""
        return await self._storage.query(
            event_type=event_type,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
    
    async def cleanup(self, retention_days: int = 90) -> int:
        """Clean up old audit logs."""
        return await self._storage.cleanup(retention_days)
    
    def _sanitize_details(self, details: dict[str, Any]) -> dict[str, Any]:
        """Remove sensitive data from details."""
        sensitive_keys = {"password", "secret", "token", "api_key", "apikey"}
        
        sanitized = {}
        for key, value in details.items():
            if any(s in key.lower() for s in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        
        return sanitized
