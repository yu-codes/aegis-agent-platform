"""
Evaluation Service

Agent evaluation and testing.

Components:
- RAGMetrics: RAG quality metrics
- HallucinationChecker: Hallucination detection
- RegressionTests: Regression testing
- BenchmarkRunner: Performance benchmarks
"""

from services.evaluation.rag_metrics import RAGMetrics, RAGScore
from services.evaluation.hallucination_check import HallucinationChecker, HallucinationResult
from services.evaluation.regression_tests import RegressionTests, TestResult
from services.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkResult

__all__ = [
    "RAGMetrics",
    "RAGScore",
    "HallucinationChecker",
    "HallucinationResult",
    "RegressionTests",
    "TestResult",
    "BenchmarkRunner",
    "BenchmarkResult",
]
