"""
Observability & Evaluation Module

Tracing, metrics, logging, and evaluation harness.
"""

from src.observability.evaluation import EvalMetric, EvalResult, EvaluationHarness
from src.observability.logging import LogLevel, StructuredLogger
from src.observability.metrics import Counter, Gauge, Histogram, MetricsCollector
from src.observability.tracing import Span, SpanContext, Tracer

__all__ = [
    "Counter",
    "EvalMetric",
    "EvalResult",
    # Evaluation
    "EvaluationHarness",
    "Gauge",
    "Histogram",
    "LogLevel",
    # Metrics
    "MetricsCollector",
    "Span",
    "SpanContext",
    # Logging
    "StructuredLogger",
    # Tracing
    "Tracer",
]
