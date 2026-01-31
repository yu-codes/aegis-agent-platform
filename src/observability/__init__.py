"""
Observability & Evaluation Module

Tracing, metrics, logging, and evaluation harness.
"""

from src.observability.tracing import Tracer, Span, SpanContext
from src.observability.metrics import MetricsCollector, Counter, Histogram, Gauge
from src.observability.logging import StructuredLogger, LogLevel
from src.observability.evaluation import EvaluationHarness, EvalResult, EvalMetric

__all__ = [
    # Tracing
    "Tracer",
    "Span",
    "SpanContext",
    # Metrics
    "MetricsCollector",
    "Counter",
    "Histogram",
    "Gauge",
    # Logging
    "StructuredLogger",
    "LogLevel",
    # Evaluation
    "EvaluationHarness",
    "EvalResult",
    "EvalMetric",
]
