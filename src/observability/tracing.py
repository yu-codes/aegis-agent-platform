"""
Distributed Tracing

OpenTelemetry-compatible tracing for agent operations.

Design decisions:
- Span-based tracing
- Context propagation
- Multiple exporters
- Automatic instrumentation hooks
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class SpanStatus(str, Enum):
    """Status of a span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """
    Context for distributed tracing.

    Can be propagated across service boundaries.
    """

    trace_id: str = field(default_factory=lambda: uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid4().hex[:16])
    parent_span_id: str | None = None

    # Baggage for cross-cutting concerns
    baggage: dict[str, str] = field(default_factory=dict)

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP headers for propagation."""
        return {
            "X-Trace-Id": self.trace_id,
            "X-Span-Id": self.span_id,
            "X-Parent-Span-Id": self.parent_span_id or "",
        }

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> "SpanContext":
        """Create from HTTP headers."""
        return cls(
            trace_id=headers.get("X-Trace-Id", uuid4().hex),
            span_id=uuid4().hex[:16],
            parent_span_id=headers.get("X-Span-Id"),
        )


@dataclass
class SpanEvent:
    """An event within a span."""

    name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """
    A unit of work in a trace.

    Spans form a tree structure representing the call hierarchy.
    """

    name: str
    context: SpanContext = field(default_factory=SpanContext)

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None

    # Status
    status: SpanStatus = SpanStatus.UNSET
    status_message: str | None = None

    # Attributes
    attributes: dict[str, Any] = field(default_factory=dict)

    # Events within the span
    events: list[SpanEvent] = field(default_factory=list)

    # Links to related spans
    links: list[SpanContext] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        self.events.append(
            SpanEvent(
                name=name,
                attributes=attributes or {},
            )
        )

    def set_status(self, status: SpanStatus, message: str | None = None) -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.utcnow()
        if self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
        }


class SpanExporter:
    """Base class for span exporters."""

    async def export(self, spans: list[Span]) -> None:
        """Export spans."""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Exports spans to console."""

    async def export(self, spans: list[Span]) -> None:
        import json

        for span in spans:
            print(f"[TRACE] {json.dumps(span.to_dict(), indent=2)}")


class InMemorySpanExporter(SpanExporter):
    """Stores spans in memory for testing."""

    def __init__(self, max_spans: int = 1000):
        self.spans: list[Span] = []
        self._max_spans = max_spans

    async def export(self, spans: list[Span]) -> None:
        self.spans.extend(spans)
        if len(self.spans) > self._max_spans:
            self.spans = self.spans[-self._max_spans :]

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a trace."""
        return [s for s in self.spans if s.context.trace_id == trace_id]

    def clear(self) -> None:
        self.spans.clear()


class Tracer:
    """
    Main tracing interface.

    Creates and manages spans, handles context propagation.
    """

    def __init__(
        self,
        service_name: str = "aegis",
        exporters: list[SpanExporter] | None = None,
        sample_rate: float = 1.0,
    ):
        self._service_name = service_name
        self._exporters = exporters or []
        self._sample_rate = sample_rate

        # Current context storage (per-task in async)
        import contextvars

        self._current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
            "current_span", default=None
        )

        # Pending spans for batched export
        self._pending: list[Span] = []

    def start_span(
        self,
        name: str,
        parent_context: SpanContext | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        Start a new span.

        If no parent context is provided, uses current span as parent.
        """
        # Determine parent
        if parent_context is None:
            current = self._current_span.get()
            if current:
                parent_context = SpanContext(
                    trace_id=current.context.trace_id,
                    span_id=uuid4().hex[:16],
                    parent_span_id=current.context.span_id,
                )

        context = parent_context or SpanContext()

        span = Span(
            name=name,
            context=context,
            attributes=attributes or {},
        )

        span.set_attribute("service.name", self._service_name)

        return span

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        """
        Context manager for tracing a block.

        Usage:
            async with tracer.trace("operation") as span:
                span.set_attribute("key", "value")
                ...
        """
        span = self.start_span(name, attributes=attributes)
        token = self._current_span.set(span)

        try:
            yield span
            span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event(
                "exception",
                {
                    "exception.type": type(e).__name__,
                    "exception.message": str(e),
                },
            )
            raise
        finally:
            span.end()
            self._current_span.reset(token)
            await self._record_span(span)

    async def _record_span(self, span: Span) -> None:
        """Record a completed span."""
        import random

        # Apply sampling
        if random.random() > self._sample_rate:
            return

        self._pending.append(span)

        # Flush if batch is large enough
        if len(self._pending) >= 10:
            await self.flush()

    async def flush(self) -> None:
        """Flush pending spans to exporters."""
        if not self._pending:
            return

        spans = self._pending[:]
        self._pending.clear()

        for exporter in self._exporters:
            try:
                await exporter.export(spans)
            except Exception:
                pass  # Don't let export errors affect main flow

    def get_current_span(self) -> Span | None:
        """Get the current span in context."""
        return self._current_span.get()

    def get_current_context(self) -> SpanContext | None:
        """Get the current trace context."""
        span = self._current_span.get()
        return span.context if span else None


# Global tracer instance
_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get or create the global tracer."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def set_tracer(tracer: Tracer) -> None:
    """Set the global tracer."""
    global _tracer
    _tracer = tracer
