"""
Worker

Background task worker.
"""

import asyncio
import signal
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from services.observability import Logger, Tracer, MetricsCollector


@dataclass
class WorkerConfig:
    """Worker configuration."""

    name: str = "aegis-worker"
    concurrency: int = 4
    poll_interval: float = 1.0
    max_retries: int = 3
    retry_delay: float = 5.0
    shutdown_timeout: float = 30.0


@dataclass
class Task:
    """A task to be processed."""

    id: str
    name: str
    payload: dict
    priority: int = 0
    retries: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


class Worker:
    """
    Background task worker.

    Processes tasks from a queue.
    """

    def __init__(self, config: WorkerConfig | None = None):
        self._config = config or WorkerConfig()
        self._handlers: dict[str, Callable] = {}
        self._running = False
        self._tasks: asyncio.Queue[Task] = asyncio.Queue()
        self._active_tasks: set[str] = set()

        self._logger = Logger.get_logger(f"worker.{self._config.name}")
        self._tracer = Tracer(service_name=self._config.name)
        self._metrics = MetricsCollector(namespace="aegis_worker")

    def register(self, task_name: str, handler: Callable) -> None:
        """Register a task handler."""
        self._handlers[task_name] = handler
        self._logger.info(f"Registered handler for task: {task_name}")

    async def enqueue(
        self,
        task_name: str,
        payload: dict,
        priority: int = 0,
    ) -> str:
        """
        Enqueue a task for processing.

        Args:
            task_name: Name of the task
            payload: Task payload
            priority: Task priority (higher = more important)

        Returns:
            Task ID
        """
        import uuid

        task = Task(
            id=str(uuid.uuid4()),
            name=task_name,
            payload=payload,
            priority=priority,
        )

        await self._tasks.put(task)
        self._metrics.counter("tasks_enqueued_total").inc(labels={"task": task_name})

        return task.id

    async def run(self) -> None:
        """Run the worker."""
        self._running = True
        self._logger.info(f"Starting worker with concurrency={self._config.concurrency}")

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        # Start worker tasks
        workers = [
            asyncio.create_task(self._worker_loop(i)) for i in range(self._config.concurrency)
        ]

        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            pass

        self._logger.info("Worker stopped")

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self._logger.info("Stopping worker...")
        self._running = False

        # Wait for active tasks
        start = datetime.utcnow()
        while self._active_tasks:
            elapsed = (datetime.utcnow() - start).total_seconds()
            if elapsed > self._config.shutdown_timeout:
                self._logger.warning(f"Timeout waiting for {len(self._active_tasks)} tasks")
                break
            await asyncio.sleep(0.5)

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop."""
        self._logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                # Get task with timeout
                try:
                    task = await asyncio.wait_for(
                        self._tasks.get(),
                        timeout=self._config.poll_interval,
                    )
                except asyncio.TimeoutError:
                    continue

                self._active_tasks.add(task.id)

                try:
                    await self._process_task(task)
                finally:
                    self._active_tasks.discard(task.id)
                    self._tasks.task_done()

            except Exception as e:
                self._logger.error(f"Worker {worker_id} error: {e}", error=e)

        self._logger.debug(f"Worker {worker_id} stopped")

    async def _process_task(self, task: Task) -> None:
        """Process a single task."""
        handler = self._handlers.get(task.name)
        if not handler:
            self._logger.warning(f"No handler for task: {task.name}")
            return

        span = self._tracer.start_span(
            name=f"task:{task.name}",
            attributes={
                "task.id": task.id,
                "task.name": task.name,
                "task.retries": task.retries,
            },
        )

        start_time = datetime.utcnow()

        try:
            self._logger.info(f"Processing task: {task.name} ({task.id})")

            result = await handler(task.payload)

            duration = (datetime.utcnow() - start_time).total_seconds()

            self._metrics.counter("tasks_completed_total").inc(
                labels={"task": task.name, "status": "success"}
            )
            self._metrics.histogram("task_duration_seconds").observe(
                duration, labels={"task": task.name}
            )

            self._tracer.end_span(span)
            self._logger.info(f"Task completed: {task.name} ({task.id}) in {duration:.2f}s")

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()

            self._logger.error(f"Task failed: {task.name} ({task.id}): {e}", error=e)
            self._metrics.counter("tasks_completed_total").inc(
                labels={"task": task.name, "status": "failure"}
            )

            self._tracer.end_span(span, status="error", error=str(e))

            # Retry if possible
            if task.retries < self._config.max_retries:
                task.retries += 1
                await asyncio.sleep(self._config.retry_delay)
                await self._tasks.put(task)
                self._logger.info(f"Retrying task: {task.name} ({task.id}), attempt {task.retries}")

    def get_stats(self) -> dict:
        """Get worker statistics."""
        return {
            "running": self._running,
            "active_tasks": len(self._active_tasks),
            "queued_tasks": self._tasks.qsize(),
            "handlers": list(self._handlers.keys()),
            "metrics": self._metrics.collect(),
        }
