"""
Policy Engine

Policy evaluation and enforcement.

Based on: src/safety/guardrails.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from pathlib import Path
import re
import yaml


class PolicyAction(str, Enum):
    """Actions to take when policy is violated."""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REDACT = "redact"


@dataclass
class Policy:
    """A policy definition."""

    id: str = ""
    name: str = ""
    description: str = ""

    # Matching
    pattern: str | None = None
    patterns: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    # Action
    action: PolicyAction = PolicyAction.WARN
    severity: str = "medium"  # low, medium, high, critical

    # Custom handler
    handler: Callable | None = None

    # Metadata
    enabled: bool = True
    version: str = "1.0.0"

    def matches(self, text: str) -> bool:
        """Check if text matches policy pattern."""
        if not self.enabled:
            return False

        patterns = self.patterns.copy()
        if self.pattern:
            patterns.append(self.pattern)

        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return True

        return False


@dataclass
class PolicyResult:
    """Result of policy evaluation."""

    passed: bool = True
    action: PolicyAction = PolicyAction.ALLOW
    violations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    modified_content: str | None = None


class PolicyEngine:
    """
    Policy evaluation engine.

    Evaluates content against policies.
    """

    def __init__(self):
        self._policies: dict[str, Policy] = {}
        self._default_action = PolicyAction.WARN

        # Register built-in policies
        self._register_builtin_policies()

    def register(self, policy: Policy) -> None:
        """Register a policy."""
        self._policies[policy.id] = policy

    def unregister(self, policy_id: str) -> bool:
        """Unregister a policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            return True
        return False

    def evaluate(
        self,
        content: str,
        context: dict[str, Any] | None = None,
        categories: list[str] | None = None,
    ) -> PolicyResult:
        """
        Evaluate content against policies.

        Args:
            content: Content to evaluate
            context: Additional context
            categories: Policy categories to check

        Returns:
            Evaluation result
        """
        result = PolicyResult()

        for policy in self._policies.values():
            if not policy.enabled:
                continue

            # Filter by category
            if categories and not any(c in policy.categories for c in categories):
                continue

            # Check if policy matches
            if policy.matches(content):
                violation = {
                    "policy_id": policy.id,
                    "policy_name": policy.name,
                    "action": policy.action.value,
                    "severity": policy.severity,
                }

                if policy.action == PolicyAction.BLOCK:
                    result.passed = False
                    result.action = PolicyAction.BLOCK
                    result.violations.append(violation)
                elif policy.action == PolicyAction.WARN:
                    result.warnings.append(
                        f"Policy '{policy.name}' triggered: {policy.description}"
                    )
                    result.violations.append(violation)
                elif policy.action == PolicyAction.REDACT:
                    # Redact matching content
                    patterns = policy.patterns + ([policy.pattern] if policy.pattern else [])
                    modified = content
                    for p in patterns:
                        modified = re.sub(p, "[REDACTED]", modified, flags=re.IGNORECASE)
                    result.modified_content = modified
                    result.violations.append(violation)

                # Run custom handler if exists
                if policy.handler:
                    try:
                        handler_result = policy.handler(content, context)
                        if handler_result is False:
                            result.passed = False
                    except Exception:
                        pass

        return result

    async def evaluate_async(
        self,
        content: str,
        context: dict[str, Any] | None = None,
        categories: list[str] | None = None,
    ) -> PolicyResult:
        """Async evaluation."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, self.evaluate, content, context, categories
        )

    def load_from_yaml(self, path: Path | str) -> int:
        """
        Load policies from YAML file.

        Returns number of policies loaded.
        """
        path = Path(path)
        if not path.exists():
            return 0

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            return 0

        policies = data.get("policies", []) if isinstance(data, dict) else data
        count = 0

        for p in policies:
            policy = Policy(
                id=p.get("id", f"policy_{count}"),
                name=p.get("name", ""),
                description=p.get("description", ""),
                pattern=p.get("pattern"),
                patterns=p.get("patterns", []),
                categories=p.get("categories", []),
                action=PolicyAction(p.get("action", "warn")),
                severity=p.get("severity", "medium"),
                enabled=p.get("enabled", True),
            )
            self.register(policy)
            count += 1

        return count

    def _register_builtin_policies(self) -> None:
        """Register built-in safety policies."""
        # PII Detection
        self.register(
            Policy(
                id="pii_ssn",
                name="SSN Detection",
                description="Detects Social Security numbers",
                pattern=r"\b\d{3}-\d{2}-\d{4}\b",
                categories=["pii", "security"],
                action=PolicyAction.REDACT,
                severity="high",
            )
        )

        self.register(
            Policy(
                id="pii_credit_card",
                name="Credit Card Detection",
                description="Detects credit card numbers",
                pattern=r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                categories=["pii", "security"],
                action=PolicyAction.REDACT,
                severity="high",
            )
        )

        self.register(
            Policy(
                id="pii_email",
                name="Email Detection",
                description="Detects email addresses",
                pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                categories=["pii"],
                action=PolicyAction.WARN,
                severity="low",
            )
        )

        # Harmful Content
        self.register(
            Policy(
                id="harmful_instructions",
                name="Harmful Instructions",
                description="Detects instructions for harmful activities",
                patterns=[
                    r"how\s+to\s+(hack|break\s+into|steal)",
                    r"(make|build|create)\s+(bomb|weapon|explosive)",
                ],
                categories=["harmful", "security"],
                action=PolicyAction.BLOCK,
                severity="critical",
            )
        )

    def list_policies(self, category: str | None = None) -> list[Policy]:
        """List all policies."""
        policies = list(self._policies.values())
        if category:
            policies = [p for p in policies if category in p.categories]
        return policies

    def set_default_action(self, action: PolicyAction) -> None:
        """Set default action for new policies."""
        self._default_action = action
