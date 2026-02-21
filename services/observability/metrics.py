"""
Metrics

Metrics collection and export.

Based on: src/observability/metrics.py
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict


@dataclass
class MetricValue:
    """A metric value with timestamp."""

    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: dict[str, str] = field(default_factory=dict)


class Counter:
    """
    Counter metric.

    Monotonically increasing value.
    """

    def __init__(self, name: str, description: str = "", labels: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, float] = {}

    def inc(self, amount: float = 1, labels: dict[str, str] | None = None) -> None:
        """Increment counter."""
        key = self._labels_to_key(labels)
        self._values[key] = self._values.get(key, 0) + amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get counter value."""
        key = self._labels_to_key(labels)
        return self._values.get(key, 0)

    def _labels_to_key(self, labels: dict[str, str] | None) -> tuple:
        """Convert labels dict to hashable key."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))


class Histogram:
    """
    Histogram metric.

    Tracks value distribution.
    """

    DEFAULT_BUCKETS = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: tuple[float, ...] | None = None,
        labels: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self.label_names = labels or []

        self._counts: dict[tuple, list[int]] = {}
        self._sums: dict[tuple, float] = {}
        self._totals: dict[tuple, int] = {}

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record an observation."""
        key = self._labels_to_key(labels)

        if key not in self._counts:
            self._counts[key] = [0] * len(self.buckets)
            self._sums[key] = 0
            self._totals[key] = 0

        # Update buckets
        for i, bucket in enumerate(self.buckets):
            if value <= bucket:
                self._counts[key][i] += 1

        self._sums[key] += value
        self._totals[key] += 1

    def get_stats(self, labels: dict[str, str] | None = None) -> dict:
        """Get histogram statistics."""
        key = self._labels_to_key(labels)

        if key not in self._totals or self._totals[key] == 0:
            return {}

        return {
            "count": self._totals[key],
            "sum": self._sums[key],
            "avg": self._sums[key] / self._totals[key],
            "buckets": dict(zip(self.buckets, self._counts[key])),
        }

    def _labels_to_key(self, labels: dict[str, str] | None) -> tuple:
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def time(self, labels: dict[str, str] | None = None) -> "HistogramTimer":
        """Context manager for timing."""
        return HistogramTimer(self, labels)


class HistogramTimer:
    """Timer context manager for histogram."""

    def __init__(self, histogram: Histogram, labels: dict[str, str] | None = None):
        self._histogram = histogram
        self._labels = labels
        self._start: float = 0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        duration = time.time() - self._start
        self._histogram.observe(duration, self._labels)


class Gauge:
    """
    Gauge metric.

    Value that can go up and down.
    """

    def __init__(self, name: str, description: str = "", labels: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, float] = {}

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Set gauge value."""
        key = self._labels_to_key(labels)
        self._values[key] = value

    def inc(self, amount: float = 1, labels: dict[str, str] | None = None) -> None:
        """Increment gauge."""
        key = self._labels_to_key(labels)
        self._values[key] = self._values.get(key, 0) + amount

    def dec(self, amount: float = 1, labels: dict[str, str] | None = None) -> None:
        """Decrement gauge."""
        key = self._labels_to_key(labels)
        self._values[key] = self._values.get(key, 0) - amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get gauge value."""
        key = self._labels_to_key(labels)
        return self._values.get(key, 0)

    def _labels_to_key(self, labels: dict[str, str] | None) -> tuple:
        if not labels:
            return ()
        return tuple(sorted(labels.items()))


class MetricsCollector:
    """
    Metrics collection and registry.

    Central registry for all metrics.
    """

    def __init__(self, namespace: str = "aegis"):
        self._namespace = namespace
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._gauges: dict[str, Gauge] = {}

        # Register default metrics
        self._register_default_metrics()

    def counter(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> Counter:
        """Get or create a counter."""
        full_name = f"{self._namespace}_{name}"
        if full_name not in self._counters:
            self._counters[full_name] = Counter(full_name, description, labels)
        return self._counters[full_name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: tuple[float, ...] | None = None,
        labels: list[str] | None = None,
    ) -> Histogram:
        """Get or create a histogram."""
        full_name = f"{self._namespace}_{name}"
        if full_name not in self._histograms:
            self._histograms[full_name] = Histogram(full_name, description, buckets, labels)
        return self._histograms[full_name]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> Gauge:
        """Get or create a gauge."""
        full_name = f"{self._namespace}_{name}"
        if full_name not in self._gauges:
            self._gauges[full_name] = Gauge(full_name, description, labels)
        return self._gauges[full_name]

    def _register_default_metrics(self) -> None:
        """Register default metrics."""
        # Request metrics
        self.counter("requests_total", "Total requests", ["method", "endpoint", "status"])
        self.histogram("request_duration_seconds", "Request duration")

        # Agent metrics
        self.counter("agent_runs_total", "Total agent runs", ["status"])
        self.histogram("agent_run_duration_seconds", "Agent run duration")
        self.gauge("agent_active_sessions", "Active agent sessions")

        # Tool metrics
        self.counter("tool_calls_total", "Total tool calls", ["tool", "status"])
        self.histogram("tool_call_duration_seconds", "Tool call duration", labels=["tool"])

        # RAG metrics
        self.histogram("rag_retrieval_duration_seconds", "RAG retrieval duration")
        self.counter("rag_retrievals_total", "Total RAG retrievals", ["domain"])

        # LLM metrics
        self.counter("llm_tokens_total", "Total LLM tokens", ["model", "type"])
        self.histogram("llm_request_duration_seconds", "LLM request duration", labels=["model"])

    def collect(self) -> dict:
        """Collect all metrics."""
        metrics = {}

        for name, counter in self._counters.items():
            # Convert tuple keys to string representation
            values = {}
            for key, value in counter._values.items():
                str_key = str(dict(key)) if key else "default"
                values[str_key] = value
            metrics[name] = {"type": "counter", "values": values}

        for name, histogram in self._histograms.items():
            metrics[name] = {
                "type": "histogram",
                "stats": histogram.get_stats(),
            }

        for name, gauge in self._gauges.items():
            values = {}
            for key, value in gauge._values.items():
                str_key = str(dict(key)) if key else "default"
                values[str_key] = value
            metrics[name] = {"type": "gauge", "values": values}

        return metrics

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, counter in self._counters.items():
            lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            for labels, value in counter._values.items():
                label_str = self._format_labels(labels)
                lines.append(f"{name}{label_str} {value}")

        for name, gauge in self._gauges.items():
            lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")
            for labels, value in gauge._values.items():
                label_str = self._format_labels(labels)
                lines.append(f"{name}{label_str} {value}")

        return "\n".join(lines)

    def _format_labels(self, labels: tuple) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""
        label_pairs = [f'{k}="{v}"' for k, v in labels]
        return "{" + ",".join(label_pairs) + "}"
