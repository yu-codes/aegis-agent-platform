"""
Worker Tasks

Task handlers for background processing.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from services.observability import Logger

logger = Logger.get_logger("worker.tasks")


@dataclass
class ProcessDocumentTask:
    """
    Process and index a document.

    Chunks, embeds, and stores a document for RAG.
    """

    name: str = "process_document"

    async def __call__(self, payload: dict) -> dict:
        """
        Process document.

        Args:
            payload: {
                "document_id": str,
                "content": str,
                "metadata": dict,
                "domain": str | None
            }
        """
        from services.rag import IndexManager
        from services.rag.chunking import RecursiveChunker

        document_id = payload["document_id"]
        content = payload["content"]
        metadata = payload.get("metadata", {})
        domain = payload.get("domain")

        logger.info(f"Processing document: {document_id}")

        # Chunk document
        chunker = RecursiveChunker()
        chunks = chunker.chunk(content, metadata)

        logger.info(f"Created {len(chunks)} chunks for document {document_id}")

        # Index chunks
        index_manager = IndexManager()
        await index_manager.index_document(
            document_id=document_id,
            chunks=chunks,
            metadata={**metadata, "domain": domain},
        )

        return {
            "document_id": document_id,
            "chunks": len(chunks),
            "status": "indexed",
        }


@dataclass
class IndexUpdateTask:
    """
    Update search index.

    Rebuilds or updates the vector index.
    """

    name: str = "index_update"

    async def __call__(self, payload: dict) -> dict:
        """
        Update index.

        Args:
            payload: {
                "domain": str | None,
                "rebuild": bool (default: False)
            }
        """
        from services.rag import IndexManager

        domain = payload.get("domain")
        rebuild = payload.get("rebuild", False)

        logger.info(f"Updating index for domain: {domain or 'all'}, rebuild={rebuild}")

        index_manager = IndexManager()

        if rebuild:
            await index_manager.rebuild_index(domain=domain)
        else:
            await index_manager.refresh_index(domain=domain)

        return {
            "domain": domain,
            "rebuild": rebuild,
            "status": "updated",
        }


@dataclass
class MemoryCleanupTask:
    """
    Clean up old memory entries.

    Removes expired sessions and decayed memories.
    """

    name: str = "memory_cleanup"

    async def __call__(self, payload: dict) -> dict:
        """
        Cleanup memory.

        Args:
            payload: {
                "max_age_hours": int (default: 24),
                "decay_threshold": float (default: 0.1)
            }
        """
        from services.memory import SessionMemory, LongTermMemory

        max_age_hours = payload.get("max_age_hours", 24)
        decay_threshold = payload.get("decay_threshold", 0.1)

        logger.info(f"Cleaning up memory older than {max_age_hours}h")

        # Cleanup sessions
        session_memory = SessionMemory()
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        sessions_cleaned = 0
        sessions = await session_memory.list_sessions(limit=10000)

        for session in sessions:
            last_activity = session.get("last_activity")
            if last_activity and last_activity < cutoff:
                await session_memory.delete_session(session["session_id"])
                sessions_cleaned += 1

        # Cleanup long-term memory
        long_term = LongTermMemory()
        memories_cleaned = await long_term.cleanup(threshold=decay_threshold)

        logger.info(f"Cleaned {sessions_cleaned} sessions, {memories_cleaned} memories")

        return {
            "sessions_cleaned": sessions_cleaned,
            "memories_cleaned": memories_cleaned,
            "status": "completed",
        }


@dataclass
class EvaluationTask:
    """
    Run evaluation benchmarks.

    Executes evaluation suite and records metrics.
    """

    name: str = "evaluation"

    async def __call__(self, payload: dict) -> dict:
        """
        Run evaluation.

        Args:
            payload: {
                "test_suite": str,
                "sample_size": int | None,
                "parallel": bool (default: True)
            }
        """
        from services.evaluation import BenchmarkRunner, RegressionTests

        test_suite = payload.get("test_suite", "default")
        sample_size = payload.get("sample_size")
        parallel = payload.get("parallel", True)

        logger.info(f"Running evaluation: {test_suite}")

        # Load test cases
        regression = RegressionTests()
        test_cases = await regression.load_tests(test_suite)

        if sample_size:
            import random

            test_cases = random.sample(test_cases, min(sample_size, len(test_cases)))

        # Run benchmark
        async def test_func(input_data: str) -> str:
            # TODO: Implement actual agent call
            return f"Response to: {input_data}"

        runner = BenchmarkRunner()
        result = await runner.run(
            test_func,
            [tc.input for tc in test_cases],
            parallel=parallel,
        )

        # Run assertions
        test_results = await regression.run_all(test_func)

        passed = sum(1 for r in test_results if r.passed)
        failed = len(test_results) - passed

        logger.info(f"Evaluation complete: {passed}/{len(test_results)} passed")

        return {
            "test_suite": test_suite,
            "total_tests": len(test_results),
            "passed": passed,
            "failed": failed,
            "benchmark": {
                "mean_ms": result.mean_ms,
                "p95_ms": result.p95_ms,
                "throughput": result.throughput,
            },
            "status": "completed",
        }


# Task registry
TASKS = {
    "process_document": ProcessDocumentTask(),
    "index_update": IndexUpdateTask(),
    "memory_cleanup": MemoryCleanupTask(),
    "evaluation": EvaluationTask(),
}


def get_task_handler(task_name: str):
    """Get handler for a task."""
    return TASKS.get(task_name)


def register_all_tasks(worker):
    """Register all tasks with a worker."""
    for name, handler in TASKS.items():
        worker.register(name, handler)
