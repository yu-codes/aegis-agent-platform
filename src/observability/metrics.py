"""
Metrics Collection

Prometheus-compatible metrics for monitoring.

Design decisions:
- Counter, Gauge, Histogram types
- Label support
- Multiple export formats
- Thread-safe updates
"""

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MetricValue:
    """A single metric observation."""

    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: dict[str, str] = field(default_factory=dict)


class Metric(ABC):
    """Base class for metrics."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._lock = threading.Lock()

    @abstractmethod
    def collect(self) -> list[MetricValue]:
        """Collect current metric values."""
        pass


class Counter(Metric):
    """
    A counter that only goes up.

    Use for: requests, errors, events.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ):
        super().__init__(name, description, labels)
        self._values: dict[tuple, float] = {}

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment the counter."""
        label_key = tuple(sorted(labels.items()))

        with self._lock:
            if label_key not in self._values:
                self._values[label_key] = 0.0
            self._values[label_key] += value

    def get(self, **labels: str) -> float:
        """Get current value."""
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            return self._values.get(label_key, 0.0)

    def collect(self) -> list[MetricValue]:
        with self._lock:
            return [
                MetricValue(
                    value=value,
                    labels=dict(label_key),
                )
                for label_key, value in self._values.items()
            ]


class Gauge(Metric):
    """
    A gauge that can go up and down.

    Use for: current connections, queue size, temperature.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ):
        super().__init__(name, description, labels)
        self._values: dict[tuple, float] = {}

    def set(self, value: float, **labels: str) -> None:
        """Set the gauge value."""
        label_key = tuple(sorted(labels.items()))

        with self._lock:
            self._values[label_key] = value

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment the gauge."""
        label_key = tuple(sorted(labels.items()))

        with self._lock:
            if label_key not in self._values:
                self._values[label_key] = 0.0
            self._values[label_key] += value

    def dec(self, value: float = 1.0, **labels: str) -> None:
        """Decrement the gauge."""
        self.inc(-value, **labels)

    def get(self, **labels: str) -> float:
        """Get current value."""
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            return self._values.get(label_key, 0.0)

    def collect(self) -> list[MetricValue]:
        with self._lock:
            return [
                MetricValue(
                    value=value,
                    labels=dict(label_key),
                )
                for label_key, value in self._values.items()
            ]


class Histogram(Metric):
    """
    A histogram for measuring distributions.

    Use for: request durations, response sizes.
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
        float("inf"),
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ):
        super().__init__(name, description, labels)
        self.buckets = buckets or self.DEFAULT_BUCKETS

        # Per-label-set data
        self._counts: dict[tuple, list[int]] = {}
        self._sums: dict[tuple, float] = {}

    def observe(self, value: float, **labels: str) -> None:
        """Record an observation."""
        label_key = tuple(sorted(labels.items()))

        with self._lock:
            if label_key not in self._counts:
                self._counts[label_key] = [0] * len(self.buckets)
                self._sums[label_key] = 0.0

            self._sums[label_key] += value

            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    self._counts[label_key][i] += 1

    def get_count(self, **labels: str) -> int:
        """Get total count of observations."""
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            counts = self._counts.get(label_key)
            if counts:
                return counts[-1]  # Last bucket is +Inf, contains all
            return 0

    def get_sum(self, **labels: str) -> float:
        """Get sum of observations."""
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            return self._sums.get(label_key, 0.0)

    def collect(self) -> list[MetricValue]:
        values = []

        with self._lock:
            for label_key, counts in self._counts.items():
                labels = dict(label_key)

                # Bucket values
                cumulative = 0
                for _i, (bound, count) in enumerate(zip(self.buckets, counts, strict=False)):
                    cumulative += count
                    bucket_labels = {**labels, "le": str(bound)}
                    values.append(
                        MetricValue(
                            value=cumulative,
                            labels=bucket_labels,
                        )
                    )

                # Sum
                values.append(
                    MetricValue(
                        value=self._sums[label_key],
                        labels={**labels, "type": "sum"},
                    )
                )

                # Count
                values.append(
                    MetricValue(
                        value=cumulative,
                        labels={**labels, "type": "count"},
                    )
                )

        return values


class Timer:
    """Context manager for timing operations."""

    def __init__(self, histogram: Histogram, **labels: str):
        self._histogram = histogram
        self._labels = labels
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        duration = time.perf_counter() - self._start
        self._histogram.observe(duration, **self._labels)


class MetricsCollector:
    """
    Central metrics registry and collector.

    Manages all metrics and provides export functionality.
    """

    def __init__(self, prefix: str = "aegis"):
        self._prefix = prefix
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()

        # Create default metrics
        self._setup_default_metrics()

    def _setup_default_metrics(self) -> None:
        """Setup default platform metrics."""
        # Request metrics
        self.register(
            Counter(
                "requests_total",
                "Total number of requests",
                labels=["endpoint", "method", "status"],
            )
        )

        self.register(
            Histogram(
                "request_duration_seconds",
                "Request duration in seconds",
                labels=["endpoint", "method"],
            )
        )

        # LLM metrics
        self.register(
            Counter(
                "llm_requests_total",
                "Total LLM API requests",
                labels=["provider", "model", "status"],
            )
        )

        self.register(
            Counter(
                "llm_tokens_total",
                "Total LLM tokens used",
                labels=["provider", "model", "direction"],
            )
        )

        self.register(
            Histogram(
                "llm_latency_seconds",
                "LLM request latency",
                labels=["provider", "model"],
            )
        )

        # Tool metrics
        self.register(
            Counter(
                "tool_invocations_total",
                "Total tool invocations",
                labels=["tool", "status"],
            )
        )

        self.register(
            Histogram(
                "tool_duration_seconds",
                "Tool execution duration",
                labels=["tool"],
            )
        )

        # Memory metrics
        self.register(
            Gauge(
                "active_sessions",
                "Number of active sessions",
            )
        )

        self.register(
            Gauge(
                "memory_entries",
                "Number of memory entries",
                labels=["type"],
            )
        )

    def register(self, metric: Metric) -> Metric:
        """Register a metric."""
        full_name = f"{self._prefix}_{metric.name}"
        metric.name = full_name

        with self._lock:
            self._metrics[full_name] = metric

        return metric

    def get(self, name: str) -> Metric | None:
        """Get a metric by name."""
        full_name = f"{self._prefix}_{name}"
        return self._metrics.get(full_name)

    def counter(self, name: str) -> Counter | None:
        """Get a counter by name."""
        metric = self.get(name)
        return metric if isinstance(metric, Counter) else None

    def gauge(self, name: str) -> Gauge | None:
        """Get a gauge by name."""
        metric = self.get(name)
        return metric if isinstance(metric, Gauge) else None

    def histogram(self, name: str) -> Histogram | None:
        """Get a histogram by name."""
        metric = self.get(name)
        return metric if isinstance(metric, Histogram) else None

    def collect_all(self) -> dict[str, list[MetricValue]]:
        """Collect all metric values."""
        with self._lock:
            return {name: metric.collect() for name, metric in self._metrics.items()}

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, values in self.collect_all().items():
            metric = self._metrics.get(name)
            if not metric:
                continue

            # HELP line
            lines.append(f"# HELP {name} {metric.description}")

            # TYPE line
            metric_type = {
                Counter: "counter",
                Gauge: "gauge",
                Histogram: "histogram",
            }.get(type(metric), "untyped")
            lines.append(f"# TYPE {name} {metric_type}")

            # Values
            for mv in values:
                label_str = ""
                if mv.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in mv.labels.items()]
                    label_str = "{" + ",".join(label_pairs) + "}"

                lines.append(f"{name}{label_str} {mv.value}")

        return "\n".join(lines)


# Global metrics collector
_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
