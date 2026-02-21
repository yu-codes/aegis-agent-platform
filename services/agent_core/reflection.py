"""
Reflection Engine

Self-improvement and error correction.

Design decisions:
- Post-execution analysis
- Error pattern detection
- Improvement suggestions
- Learning from failures
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4


class ReflectionType(str, Enum):
    """Types of reflection."""

    SUCCESS_ANALYSIS = "success_analysis"
    ERROR_ANALYSIS = "error_analysis"
    QUALITY_IMPROVEMENT = "quality_improvement"
    EFFICIENCY_REVIEW = "efficiency_review"


@dataclass
class ReflectionInput:
    """Input for reflection."""

    query: str = ""
    response: str = ""
    success: bool = True
    error: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    iterations: int = 1
    duration_ms: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Insight:
    """A single insight from reflection."""

    id: UUID = field(default_factory=uuid4)
    type: str = ""
    description: str = ""
    severity: str = "info"  # info, warning, error
    suggestion: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionResult:
    """Result of reflection analysis."""

    id: UUID = field(default_factory=uuid4)
    reflection_type: ReflectionType = ReflectionType.SUCCESS_ANALYSIS
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Analysis
    insights: list[Insight] = field(default_factory=list)
    overall_quality: float = 0.0  # 0-1

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # Patterns detected
    patterns: list[str] = field(default_factory=list)


class LLMProtocol(Protocol):
    """Protocol for LLM calls in reflection."""

    async def complete(self, messages: list[dict], **kwargs) -> dict: ...


class ReflectionEngine:
    """
    Self-reflection and improvement engine.

    Analyzes execution results and suggests improvements.
    """

    def __init__(self, llm: LLMProtocol | None = None):
        self._llm = llm
        self._history: list[ReflectionResult] = []
        self._error_patterns: dict[str, int] = {}

    async def reflect(
        self,
        input_data: ReflectionInput,
        reflection_type: ReflectionType | None = None,
    ) -> ReflectionResult:
        """
        Perform reflection on execution.

        Args:
            input_data: Data about the execution
            reflection_type: Type of reflection to perform

        Returns:
            Reflection result with insights
        """
        # Determine reflection type
        if reflection_type is None:
            reflection_type = (
                ReflectionType.ERROR_ANALYSIS
                if not input_data.success
                else ReflectionType.SUCCESS_ANALYSIS
            )

        # Perform analysis
        if self._llm:
            result = await self._llm_reflect(input_data, reflection_type)
        else:
            result = self._simple_reflect(input_data, reflection_type)

        # Track error patterns
        if not input_data.success and input_data.error:
            self._track_error_pattern(input_data.error)

        # Save to history
        self._history.append(result)

        return result

    def _simple_reflect(
        self,
        input_data: ReflectionInput,
        reflection_type: ReflectionType,
    ) -> ReflectionResult:
        """Simple rule-based reflection."""
        insights: list[Insight] = []
        recommendations: list[str] = []
        patterns: list[str] = []
        quality = 1.0

        # Analyze based on type
        if reflection_type == ReflectionType.ERROR_ANALYSIS:
            quality = 0.0

            if input_data.error:
                insights.append(
                    Insight(
                        type="error",
                        description=f"Execution failed: {input_data.error}",
                        severity="error",
                    )
                )

                # Check for common patterns
                if "timeout" in input_data.error.lower():
                    patterns.append("timeout_errors")
                    recommendations.append("Consider increasing timeout or optimizing operations")

                if "rate limit" in input_data.error.lower():
                    patterns.append("rate_limiting")
                    recommendations.append("Implement backoff or reduce request frequency")

        else:
            # Success analysis
            if input_data.iterations > 5:
                quality -= 0.2
                insights.append(
                    Insight(
                        type="efficiency",
                        description=f"High iteration count: {input_data.iterations}",
                        severity="warning",
                        suggestion="Consider optimizing the reasoning loop",
                    )
                )

            if input_data.duration_ms > 10000:
                quality -= 0.1
                insights.append(
                    Insight(
                        type="performance",
                        description=f"Slow execution: {input_data.duration_ms:.0f}ms",
                        severity="warning",
                        suggestion="Consider caching or parallel execution",
                    )
                )

            if len(input_data.tool_calls) > 10:
                quality -= 0.1
                insights.append(
                    Insight(
                        type="tool_usage",
                        description=f"High tool call count: {len(input_data.tool_calls)}",
                        severity="info",
                        suggestion="Review if all tool calls are necessary",
                    )
                )

        return ReflectionResult(
            reflection_type=reflection_type,
            insights=insights,
            overall_quality=max(0, quality),
            recommendations=recommendations,
            patterns=patterns,
        )

    async def _llm_reflect(
        self,
        input_data: ReflectionInput,
        reflection_type: ReflectionType,
    ) -> ReflectionResult:
        """LLM-based deep reflection."""
        if not self._llm:
            return self._simple_reflect(input_data, reflection_type)

        system_prompt = """You are an AI execution analyst.
Analyze the execution and provide insights.

Respond in JSON format:
{
    "overall_quality": 0.0 to 1.0,
    "insights": [
        {
            "type": "error|efficiency|quality|performance",
            "description": "...",
            "severity": "info|warning|error",
            "suggestion": "..."
        }
    ],
    "recommendations": ["..."],
    "patterns": ["..."]
}"""

        user_content = f"""Analyze this execution:

Query: {input_data.query}
Response: {input_data.response[:500]}
Success: {input_data.success}
Error: {input_data.error}
Iterations: {input_data.iterations}
Duration: {input_data.duration_ms:.0f}ms
Tool Calls: {len(input_data.tool_calls)}

Reflection Type: {reflection_type.value}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await self._llm.complete(messages, response_format={"type": "json_object"})
            content = response.get("content", "{}")

            import json

            data = json.loads(content)

            insights = [
                Insight(
                    type=i.get("type", ""),
                    description=i.get("description", ""),
                    severity=i.get("severity", "info"),
                    suggestion=i.get("suggestion"),
                )
                for i in data.get("insights", [])
            ]

            return ReflectionResult(
                reflection_type=reflection_type,
                insights=insights,
                overall_quality=data.get("overall_quality", 0.5),
                recommendations=data.get("recommendations", []),
                patterns=data.get("patterns", []),
            )

        except Exception:
            return self._simple_reflect(input_data, reflection_type)

    def _track_error_pattern(self, error: str) -> None:
        """Track error patterns for learning."""
        # Simple pattern extraction
        error_lower = error.lower()

        if "timeout" in error_lower:
            self._error_patterns["timeout"] = self._error_patterns.get("timeout", 0) + 1
        elif "rate" in error_lower:
            self._error_patterns["rate_limit"] = self._error_patterns.get("rate_limit", 0) + 1
        elif "permission" in error_lower or "auth" in error_lower:
            self._error_patterns["auth"] = self._error_patterns.get("auth", 0) + 1
        else:
            self._error_patterns["other"] = self._error_patterns.get("other", 0) + 1

    def get_error_patterns(self) -> dict[str, int]:
        """Get tracked error patterns."""
        return dict(self._error_patterns)

    def get_history(self, limit: int = 10) -> list[ReflectionResult]:
        """Get reflection history."""
        return self._history[-limit:]

    def get_average_quality(self) -> float:
        """Get average quality from recent reflections."""
        if not self._history:
            return 1.0

        recent = self._history[-10:]
        return sum(r.overall_quality for r in recent) / len(recent)
