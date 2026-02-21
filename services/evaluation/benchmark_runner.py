"""
Benchmark Runner

Performance benchmarking for agent components.
"""

import asyncio
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from uuid import UUID, uuid4


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""

    iterations: int = 10
    warmup_iterations: int = 2
    timeout_seconds: float = 60.0
    parallel: bool = False
    concurrency: int = 5


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""

    # Timing (in milliseconds)
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    std_dev_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    # Counts
    iterations: int = 0
    successes: int = 0
    failures: int = 0

    # Throughput
    ops_per_second: float = 0.0

    # Metadata
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "iterations": self.iterations,
            "successes": self.successes,
            "failures": self.failures,
            "ops_per_second": round(self.ops_per_second, 2),
            "total_duration_ms": round(self.total_duration_ms, 2),
        }


class BenchmarkRunner:
    """
    Performance benchmarking framework.

    Measures latency, throughput, and resource usage.
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        self._config = config or BenchmarkConfig()
        self._benchmarks: dict[str, Callable] = {}
        self._results: list[BenchmarkResult] = []

    def register(
        self,
        name: str,
        func: Callable,
    ) -> None:
        """Register a benchmark function."""
        self._benchmarks[name] = func

    async def run(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """
        Run a single benchmark.

        Args:
            name: Benchmark name
            *args, **kwargs: Arguments to pass to benchmark function

        Returns:
            Benchmark result
        """
        func = self._benchmarks.get(name)
        if not func:
            return BenchmarkResult(name=name, failures=1)

        result = BenchmarkResult(name=name, started_at=datetime.utcnow())
        durations: list[float] = []

        start_total = time.time()

        # Warmup
        for _ in range(self._config.warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    func(*args, **kwargs)
            except Exception:
                pass

        # Actual benchmark
        if self._config.parallel:
            durations = await self._run_parallel(func, args, kwargs)
        else:
            durations = await self._run_sequential(func, args, kwargs)

        result.total_duration_ms = (time.time() - start_total) * 1000
        result.completed_at = datetime.utcnow()

        # Calculate statistics
        if durations:
            self._calculate_stats(result, durations)

        self._results.append(result)
        return result

    async def _run_sequential(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
    ) -> list[float]:
        """Run benchmark sequentially."""
        durations = []

        for _ in range(self._config.iterations):
            start = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self._config.timeout_seconds,
                    )
                else:
                    func(*args, **kwargs)

                duration = (time.time() - start) * 1000
                durations.append(duration)
            except Exception:
                pass

        return durations

    async def _run_parallel(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
    ) -> list[float]:
        """Run benchmark in parallel."""
        durations = []

        async def timed_call():
            start = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda: func(*args, **kwargs)
                    )
                return (time.time() - start) * 1000
            except Exception:
                return None

        # Run in batches
        remaining = self._config.iterations
        while remaining > 0:
            batch_size = min(remaining, self._config.concurrency)
            tasks = [timed_call() for _ in range(batch_size)]
            results = await asyncio.gather(*tasks)

            for r in results:
                if r is not None:
                    durations.append(r)

            remaining -= batch_size

        return durations

    def _calculate_stats(self, result: BenchmarkResult, durations: list[float]) -> None:
        """Calculate statistics from duration list."""
        result.iterations = self._config.iterations
        result.successes = len(durations)
        result.failures = self._config.iterations - len(durations)

        if not durations:
            return

        sorted_durations = sorted(durations)

        result.min_ms = min(durations)
        result.max_ms = max(durations)
        result.mean_ms = statistics.mean(durations)
        result.median_ms = statistics.median(durations)

        if len(durations) > 1:
            result.std_dev_ms = statistics.stdev(durations)

        # Percentiles
        n = len(sorted_durations)
        result.p95_ms = sorted_durations[int(n * 0.95)] if n > 0 else 0
        result.p99_ms = sorted_durations[int(n * 0.99)] if n > 0 else 0

        # Throughput
        if result.total_duration_ms > 0:
            result.ops_per_second = (result.successes / result.total_duration_ms) * 1000

    async def run_all(self) -> list[BenchmarkResult]:
        """Run all registered benchmarks."""
        results = []
        for name in self._benchmarks:
            result = await self.run(name)
            results.append(result)
        return results

    def compare(
        self,
        result1: BenchmarkResult,
        result2: BenchmarkResult,
    ) -> dict:
        """Compare two benchmark results."""
        return {
            "name_1": result1.name,
            "name_2": result2.name,
            "mean_diff_ms": result2.mean_ms - result1.mean_ms,
            "mean_diff_pct": (
                ((result2.mean_ms - result1.mean_ms) / result1.mean_ms * 100)
                if result1.mean_ms > 0
                else 0
            ),
            "throughput_diff_pct": (
                ((result2.ops_per_second - result1.ops_per_second) / result1.ops_per_second * 100)
                if result1.ops_per_second > 0
                else 0
            ),
            "faster": result1.name if result1.mean_ms < result2.mean_ms else result2.name,
        }

    def get_results(self) -> list[BenchmarkResult]:
        """Get all benchmark results."""
        return list(self._results)

    def clear_results(self) -> None:
        """Clear all results."""
        self._results.clear()
