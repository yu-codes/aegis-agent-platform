"""
Tracer

Distributed tracing implementation.

Based on: src/observability/tracing.py
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict
from uuid import UUID, uuid4
from contextvars import ContextVar


# Context variable for current span
_current_span: ContextVar["Span | None"] = ContextVar("current_span", default=None)


@dataclass
class SpanContext:
    """Context for a span."""

    trace_id: UUID = field(default_factory=uuid4)
    span_id: UUID = field(default_factory=uuid4)
    parent_span_id: UUID | None = None

    # Baggage items (propagated across services)
    baggage: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for propagation."""
        return {
            "trace_id": str(self.trace_id),
            "span_id": str(self.span_id),
            "parent_span_id": str(self.parent_span_id) if self.parent_span_id else None,
            "baggage": self.baggage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpanContext":
        """Create from dictionary."""
        return cls(
            trace_id=UUID(data["trace_id"]) if data.get("trace_id") else uuid4(),
            span_id=UUID(data["span_id"]) if data.get("span_id") else uuid4(),
            parent_span_id=UUID(data["parent_span_id"]) if data.get("parent_span_id") else None,
            baggage=data.get("baggage", {}),
        )


@dataclass
class Span:
    """A tracing span."""

    name: str = ""
    context: SpanContext = field(default_factory=SpanContext)

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float = 0.0

    # Status
    status: str = "ok"  # ok, error
    error_message: str | None = None

    # Attributes and events
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    # Span kind
    kind: str = "internal"  # client, server, producer, consumer, internal

    def end(self, status: str = "ok", error_message: str | None = None) -> None:
        """End the span."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error_message = error_message

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        """Add an event to the span."""
        self.events.append(
            {
                "name": name,
                "timestamp": datetime.utcnow().isoformat(),
                "attributes": attributes or {},
            }
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "trace_id": str(self.context.trace_id),
            "span_id": str(self.context.span_id),
            "parent_span_id": (
                str(self.context.parent_span_id) if self.context.parent_span_id else None
            ),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "events": self.events,
            "kind": self.kind,
        }


class Tracer:
    """
    Distributed tracing service.

    Creates and manages spans for request tracing.
    """

    def __init__(
        self,
        service_name: str = "aegis-agent",
        exporter: Any | None = None,  # OpenTelemetry exporter
    ):
        self._service_name = service_name
        self._exporter = exporter
        self._spans: list[Span] = []

    def start_span(
        self,
        name: str,
        parent: Span | SpanContext | None = None,
        kind: str = "internal",
        attributes: dict | None = None,
    ) -> Span:
        """
        Start a new span.

        Args:
            name: Span name
            parent: Parent span or context
            kind: Span kind
            attributes: Initial attributes

        Returns:
            New span
        """
        # Get parent context
        if parent is None:
            parent = _current_span.get()

        if isinstance(parent, Span):
            parent_context = parent.context
        elif isinstance(parent, SpanContext):
            parent_context = parent
        else:
            parent_context = None

        # Create span context
        if parent_context:
            context = SpanContext(
                trace_id=parent_context.trace_id,
                parent_span_id=parent_context.span_id,
                baggage=dict(parent_context.baggage),
            )
        else:
            context = SpanContext()

        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes or {},
        )

        span.set_attribute("service.name", self._service_name)

        return span

    def end_span(self, span: Span, status: str = "ok", error: str | None = None) -> None:
        """End a span."""
        span.end(status, error)
        self._spans.append(span)

        # Export if exporter configured
        if self._exporter:
            self._export_span(span)

    def current_span(self) -> Span | None:
        """Get current span from context."""
        return _current_span.get()

    def __call__(self, name: str, kind: str = "internal"):
        """Decorator to trace a function."""

        def decorator(func):
            import functools

            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    span = self.start_span(name, kind=kind)
                    token = _current_span.set(span)
                    try:
                        result = await func(*args, **kwargs)
                        self.end_span(span)
                        return result
                    except Exception as e:
                        self.end_span(span, status="error", error=str(e))
                        raise
                    finally:
                        _current_span.reset(token)

                return async_wrapper
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    span = self.start_span(name, kind=kind)
                    token = _current_span.set(span)
                    try:
                        result = func(*args, **kwargs)
                        self.end_span(span)
                        return result
                    except Exception as e:
                        self.end_span(span, status="error", error=str(e))
                        raise
                    finally:
                        _current_span.reset(token)

                return sync_wrapper

        return decorator

    def _export_span(self, span: Span) -> None:
        """Export span to configured exporter."""
        try:
            if hasattr(self._exporter, "export"):
                self._exporter.export([span.to_dict()])
        except Exception:
            pass

    def get_spans(self, trace_id: UUID | None = None) -> list[Span]:
        """Get recorded spans."""
        if trace_id:
            return [s for s in self._spans if s.context.trace_id == trace_id]
        return list(self._spans)

    def clear_spans(self) -> None:
        """Clear recorded spans."""
        self._spans.clear()

    def inject_context(self, span: Span) -> dict:
        """Inject span context into headers for propagation."""
        return {
            "traceparent": f"00-{span.context.trace_id}-{span.context.span_id}-01",
            "tracestate": ",".join(f"{k}={v}" for k, v in span.context.baggage.items()),
        }

    def extract_context(self, headers: dict) -> SpanContext | None:
        """Extract span context from headers."""
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None

        parts = traceparent.split("-")
        if len(parts) < 3:
            return None

        try:
            return SpanContext(
                trace_id=UUID(parts[1]),
                span_id=UUID(parts[2]),
            )
        except (ValueError, IndexError):
            return None


# Import for decorator
import asyncio
