"""
Regression Tests

Agent regression testing framework.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol
from uuid import UUID, uuid4
import json


@dataclass
class TestCase:
    """A test case definition."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""

    # Input
    input_message: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    # Expected output
    expected_output: str | None = None
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    expected_tool_calls: list[str] = field(default_factory=list)

    # Configuration
    domain: str = "default"
    timeout_seconds: float = 30.0

    # Metadata
    tags: list[str] = field(default_factory=list)
    priority: int = 1  # 1-5, 1 being highest


@dataclass
class TestResult:
    """Result of a test run."""

    test_id: UUID = field(default_factory=uuid4)
    test_name: str = ""
    passed: bool = False

    # Output
    actual_output: str = ""
    tool_calls: list[str] = field(default_factory=list)

    # Timing
    duration_ms: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    # Failures
    failure_reason: str | None = None
    assertions_passed: int = 0
    assertions_failed: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "test_id": str(self.test_id),
            "test_name": self.test_name,
            "passed": self.passed,
            "actual_output": self.actual_output,
            "tool_calls": self.tool_calls,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "failure_reason": self.failure_reason,
            "assertions_passed": self.assertions_passed,
            "assertions_failed": self.assertions_failed,
        }


class AgentProtocol(Protocol):
    """Protocol for agent under test."""

    async def run(self, message: str, context: dict | None = None) -> dict: ...


class RegressionTests:
    """
    Regression testing framework.

    Runs test suites against agent implementations.
    """

    def __init__(self, agent: AgentProtocol | None = None):
        self._agent = agent
        self._test_cases: dict[UUID, TestCase] = {}
        self._results: list[TestResult] = []

    def register_test(self, test_case: TestCase) -> None:
        """Register a test case."""
        self._test_cases[test_case.id] = test_case

    def register_tests_from_json(self, json_path: str) -> int:
        """
        Load test cases from JSON file.

        Returns number of tests loaded.
        """
        with open(json_path) as f:
            data = json.load(f)

        tests = data if isinstance(data, list) else data.get("tests", [])
        count = 0

        for t in tests:
            test_case = TestCase(
                id=UUID(t["id"]) if t.get("id") else uuid4(),
                name=t.get("name", f"Test {count + 1}"),
                description=t.get("description", ""),
                input_message=t.get("input", t.get("input_message", "")),
                context=t.get("context", {}),
                expected_output=t.get("expected_output"),
                expected_contains=t.get("expected_contains", []),
                expected_not_contains=t.get("expected_not_contains", []),
                expected_tool_calls=t.get("expected_tool_calls", []),
                domain=t.get("domain", "default"),
                timeout_seconds=t.get("timeout", 30.0),
                tags=t.get("tags", []),
                priority=t.get("priority", 1),
            )
            self.register_test(test_case)
            count += 1

        return count

    async def run_test(self, test_id: UUID) -> TestResult:
        """Run a single test case."""
        import time

        test_case = self._test_cases.get(test_id)
        if not test_case:
            return TestResult(
                test_id=test_id,
                failure_reason="Test case not found",
            )

        result = TestResult(
            test_id=test_id,
            test_name=test_case.name,
            started_at=datetime.utcnow(),
        )

        if not self._agent:
            result.failure_reason = "No agent configured"
            result.completed_at = datetime.utcnow()
            return result

        start_time = time.time()

        try:
            # Run agent
            response = await self._agent.run(
                test_case.input_message,
                context=test_case.context,
            )

            result.actual_output = response.get("output", response.get("response", ""))
            result.tool_calls = response.get("tool_calls", [])

            # Check assertions
            result.passed = True

            # Check expected output
            if test_case.expected_output:
                if result.actual_output != test_case.expected_output:
                    result.passed = False
                    result.failure_reason = "Output does not match expected"
                    result.assertions_failed += 1
                else:
                    result.assertions_passed += 1

            # Check expected contains
            for expected in test_case.expected_contains:
                if expected.lower() in result.actual_output.lower():
                    result.assertions_passed += 1
                else:
                    result.passed = False
                    result.failure_reason = f"Output missing expected: {expected}"
                    result.assertions_failed += 1

            # Check expected not contains
            for not_expected in test_case.expected_not_contains:
                if not_expected.lower() in result.actual_output.lower():
                    result.passed = False
                    result.failure_reason = f"Output contains unexpected: {not_expected}"
                    result.assertions_failed += 1
                else:
                    result.assertions_passed += 1

            # Check expected tool calls
            for tool in test_case.expected_tool_calls:
                if tool in result.tool_calls:
                    result.assertions_passed += 1
                else:
                    result.passed = False
                    result.failure_reason = f"Expected tool call not made: {tool}"
                    result.assertions_failed += 1

        except Exception as e:
            result.passed = False
            result.failure_reason = f"Exception: {type(e).__name__}: {str(e)}"

        result.duration_ms = (time.time() - start_time) * 1000
        result.completed_at = datetime.utcnow()

        self._results.append(result)
        return result

    async def run_all(
        self,
        tags: list[str] | None = None,
        priority: int | None = None,
    ) -> list[TestResult]:
        """
        Run all matching test cases.

        Args:
            tags: Filter by tags
            priority: Filter by max priority

        Returns:
            List of test results
        """
        results = []

        for test_id, test_case in self._test_cases.items():
            # Filter by tags
            if tags and not any(t in test_case.tags for t in tags):
                continue

            # Filter by priority
            if priority and test_case.priority > priority:
                continue

            result = await self.run_test(test_id)
            results.append(result)

        return results

    def get_summary(self) -> dict:
        """Get test run summary."""
        if not self._results:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}

        passed = sum(1 for r in self._results if r.passed)
        total = len(self._results)

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "avg_duration_ms": sum(r.duration_ms for r in self._results) / total,
        }

    def set_agent(self, agent: AgentProtocol) -> None:
        """Set the agent to test."""
        self._agent = agent

    def clear_results(self) -> None:
        """Clear test results."""
        self._results.clear()
