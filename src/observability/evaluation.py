"""
Evaluation Harness

Framework for evaluating agent performance.

Design decisions:
- Metric-based evaluation
- Multiple evaluator types
- Batch evaluation support
- Result aggregation
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from src.reasoning.llm import BaseLLMAdapter


class EvalMetricType(str, Enum):
    """Types of evaluation metrics."""

    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    GROUNDEDNESS = "groundedness"
    FAITHFULNESS = "faithfulness"
    TOXICITY = "toxicity"
    LATENCY = "latency"
    TOKEN_EFFICIENCY = "token_efficiency"
    TOOL_SUCCESS_RATE = "tool_success_rate"
    CUSTOM = "custom"


@dataclass
class EvalMetric:
    """
    A single evaluation metric result.
    """

    name: str
    metric_type: EvalMetricType
    value: float  # 0.0 - 1.0 for most metrics

    # Additional context
    threshold: float | None = None
    passed: bool | None = None

    # Details
    explanation: str | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """
    Complete evaluation result for a single example.
    """

    id: UUID = field(default_factory=uuid4)

    # Input/Output
    input_text: str = ""
    output_text: str = ""
    expected_output: str | None = None

    # Metrics
    metrics: list[EvalMetric] = field(default_factory=list)

    # Overall
    overall_score: float = 0.0
    passed: bool = True

    # Metadata
    model: str | None = None
    latency_ms: float | None = None
    token_count: int | None = None

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def get_metric(self, name: str) -> EvalMetric | None:
        """Get a specific metric by name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None


@dataclass
class EvalDataset:
    """A dataset for evaluation."""

    name: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    # Each example: {"input": str, "expected": str, "metadata": dict}

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class Evaluator(ABC):
    """Base class for evaluators."""

    name: str = "base_evaluator"

    @abstractmethod
    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[EvalMetric]:
        """Evaluate a single input/output pair."""
        pass


class ExactMatchEvaluator(Evaluator):
    """Evaluates exact match between output and expected."""

    name = "exact_match"

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[EvalMetric]:
        if expected is None:
            return []

        matches = output_text.strip() == expected.strip()

        return [
            EvalMetric(
                name="exact_match",
                metric_type=EvalMetricType.ACCURACY,
                value=1.0 if matches else 0.0,
                threshold=1.0,
                passed=matches,
            )
        ]


class ContainsEvaluator(Evaluator):
    """Evaluates if output contains expected substrings."""

    name = "contains"

    def __init__(self, substrings: list[str] | None = None):
        self._substrings = substrings or []

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[EvalMetric]:
        substrings = self._substrings

        if expected:
            substrings = [expected]

        if not substrings:
            return []

        found = sum(1 for s in substrings if s in output_text)
        score = found / len(substrings)

        return [
            EvalMetric(
                name="contains",
                metric_type=EvalMetricType.ACCURACY,
                value=score,
                threshold=0.8,
                passed=score >= 0.8,
                raw_data={"found": found, "total": len(substrings)},
            )
        ]


class LengthEvaluator(Evaluator):
    """Evaluates if output length is within expected range."""

    name = "length"

    def __init__(self, min_length: int = 10, max_length: int = 10000):
        self._min = min_length
        self._max = max_length

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[EvalMetric]:
        length = len(output_text)
        in_range = self._min <= length <= self._max

        # Score based on how close to ideal range
        if in_range:
            score = 1.0
        elif length < self._min:
            score = max(0, length / self._min)
        else:
            score = max(0, 1 - (length - self._max) / self._max)

        return [
            EvalMetric(
                name="length",
                metric_type=EvalMetricType.CUSTOM,
                value=score,
                passed=in_range,
                raw_data={"length": length, "min": self._min, "max": self._max},
            )
        ]


class LLMEvaluator(Evaluator):
    """
    Uses an LLM to evaluate quality.

    Good for subjective metrics like relevance, coherence.
    """

    name = "llm_judge"

    EVAL_PROMPT = """You are evaluating an AI assistant's response.

Question/Input: {input}

AI Response: {output}

{expected_section}

Evaluate the response on the following criteria and provide a score from 0-10 for each:

1. Relevance: Does the response address the question/input?
2. Coherence: Is the response well-structured and logical?
3. Accuracy: Is the information correct (if verifiable)?
4. Helpfulness: Would this response be helpful to the user?

Respond in JSON format:
{{
    "relevance": {{"score": 0-10, "explanation": "..."}},
    "coherence": {{"score": 0-10, "explanation": "..."}},
    "accuracy": {{"score": 0-10, "explanation": "..."}},
    "helpfulness": {{"score": 0-10, "explanation": "..."}}
}}"""

    def __init__(self, llm_adapter):
        self._llm: BaseLLMAdapter = llm_adapter

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[EvalMetric]:
        import json

        from src.core.types import Message

        expected_section = ""
        if expected:
            expected_section = f"Expected/Reference Output: {expected}\n"

        prompt = self.EVAL_PROMPT.format(
            input=input_text,
            output=output_text,
            expected_section=expected_section,
        )

        try:
            response = await self._llm.complete([Message(role="user", content=prompt)])

            # Parse JSON response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)

            metrics = []
            for criterion, result in data.items():
                score = result.get("score", 0) / 10.0
                metrics.append(
                    EvalMetric(
                        name=criterion,
                        metric_type=EvalMetricType.CUSTOM,
                        value=score,
                        threshold=0.6,
                        passed=score >= 0.6,
                        explanation=result.get("explanation"),
                    )
                )

            return metrics

        except Exception as e:
            return [
                EvalMetric(
                    name="llm_eval_error",
                    metric_type=EvalMetricType.CUSTOM,
                    value=0.0,
                    passed=False,
                    explanation=f"Evaluation failed: {e!s}",
                )
            ]


class EvaluationHarness:
    """
    Main evaluation framework.

    Features:
    - Multiple evaluators
    - Batch evaluation
    - Result aggregation
    - Threshold checking
    """

    def __init__(
        self,
        evaluators: list[Evaluator] | None = None,
        pass_threshold: float = 0.7,
    ):
        self._evaluators = evaluators or [
            ExactMatchEvaluator(),
            LengthEvaluator(),
        ]
        self._pass_threshold = pass_threshold

    def add_evaluator(self, evaluator: Evaluator) -> "EvaluationHarness":
        """Add an evaluator."""
        self._evaluators.append(evaluator)
        return self

    async def evaluate_single(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate a single example."""
        metrics = []

        for evaluator in self._evaluators:
            try:
                eval_metrics = await evaluator.evaluate(input_text, output_text, expected, context)
                metrics.extend(eval_metrics)
            except Exception as e:
                metrics.append(
                    EvalMetric(
                        name=f"{evaluator.name}_error",
                        metric_type=EvalMetricType.CUSTOM,
                        value=0.0,
                        passed=False,
                        explanation=str(e),
                    )
                )

        # Calculate overall score
        overall = sum(m.value for m in metrics) / len(metrics) if metrics else 0.0

        passed = overall >= self._pass_threshold

        return EvalResult(
            input_text=input_text,
            output_text=output_text,
            expected_output=expected,
            metrics=metrics,
            overall_score=overall,
            passed=passed,
        )

    async def evaluate_batch(
        self,
        examples: list[dict[str, Any]],
        generate_fn: Callable[[str], Awaitable[str]] | None = None,
    ) -> list[EvalResult]:
        """
        Evaluate a batch of examples.

        Args:
            examples: List of {"input": str, "output": str, "expected": str}
            generate_fn: Optional function to generate outputs from inputs
        """

        results = []

        for example in examples:
            input_text = example.get("input", "")
            output_text = example.get("output", "")
            expected = example.get("expected")

            # Generate output if function provided
            if generate_fn and not output_text:
                output_text = await generate_fn(input_text)

            result = await self.evaluate_single(
                input_text, output_text, expected, example.get("metadata")
            )
            results.append(result)

        return results

    async def evaluate_dataset(
        self,
        dataset: EvalDataset,
        generate_fn: Callable[[str], Awaitable[str]],
    ) -> dict[str, Any]:
        """
        Evaluate an entire dataset.

        Returns aggregated metrics.
        """
        results = await self.evaluate_batch(dataset.examples, generate_fn)

        return self.aggregate_results(results)

    def aggregate_results(self, results: list[EvalResult]) -> dict[str, Any]:
        """Aggregate evaluation results."""
        if not results:
            return {"total": 0}

        # Collect all metric values by name
        metric_values: dict[str, list[float]] = {}

        for result in results:
            for metric in result.metrics:
                if metric.name not in metric_values:
                    metric_values[metric.name] = []
                metric_values[metric.name].append(metric.value)

        # Calculate aggregates
        aggregates = {}
        for name, values in metric_values.items():
            aggregates[name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

        # Overall stats
        overall_scores = [r.overall_score for r in results]
        pass_rate = sum(1 for r in results if r.passed) / len(results)

        return {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "pass_rate": pass_rate,
            "overall_mean": sum(overall_scores) / len(overall_scores),
            "overall_min": min(overall_scores),
            "overall_max": max(overall_scores),
            "metrics": aggregates,
        }
