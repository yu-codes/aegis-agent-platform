"""
Guardrails

Runtime safety checks for inputs and outputs.

Design decisions:
- Chainable guardrails
- Configurable actions (block, warn, modify)
- Both input and output guardrails
- Async support
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GuardrailAction(str, Enum):
    """Action to take when guardrail triggers."""

    ALLOW = "allow"  # Continue normally
    WARN = "warn"  # Log warning, continue
    MODIFY = "modify"  # Modify content, continue
    BLOCK = "block"  # Stop processing


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    action: GuardrailAction
    triggered: bool = False
    reason: str | None = None
    modified_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Guardrail(ABC):
    """Base class for input guardrails."""

    name: str = "base_guardrail"

    @abstractmethod
    async def check(self, content: str, context: dict[str, Any]) -> GuardrailResult:
        """
        Check content against guardrail.

        Args:
            content: The input to check
            context: Additional context (user_id, session, etc.)

        Returns:
            GuardrailResult with action to take
        """
        pass


class OutputGuardrail(ABC):
    """Base class for output guardrails."""

    name: str = "base_output_guardrail"

    @abstractmethod
    async def check(
        self,
        output: str,
        input_content: str,
        context: dict[str, Any],
    ) -> GuardrailResult:
        """
        Check output content.

        Args:
            output: The output to check
            input_content: Original input that generated this output
            context: Additional context

        Returns:
            GuardrailResult with action to take
        """
        pass


class LengthGuardrail(Guardrail):
    """Enforces maximum content length."""

    name = "length_guardrail"

    def __init__(self, max_length: int = 50000, truncate: bool = True):
        self._max_length = max_length
        self._truncate = truncate

    async def check(self, content: str, context: dict[str, Any]) -> GuardrailResult:
        if len(content) <= self._max_length:
            return GuardrailResult(action=GuardrailAction.ALLOW)

        if self._truncate:
            truncated = content[: self._max_length] + "... [truncated]"
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                triggered=True,
                reason=f"Content exceeded {self._max_length} characters, truncated",
                modified_content=truncated,
            )

        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            triggered=True,
            reason=f"Content exceeded maximum length of {self._max_length}",
        )


class ProfanityGuardrail(Guardrail):
    """Filters profanity from content."""

    name = "profanity_guardrail"

    def __init__(
        self, word_list: list[str] | None = None, action: GuardrailAction = GuardrailAction.MODIFY
    ):
        # Simple placeholder list - use a proper library in production
        self._words = set(word_list or ["badword1", "badword2"])
        self._action = action

    async def check(self, content: str, context: dict[str, Any]) -> GuardrailResult:
        import re

        found = []
        modified = content

        for word in self._words:
            pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
            if pattern.search(content):
                found.append(word)
                modified = pattern.sub("[filtered]", modified)

        if not found:
            return GuardrailResult(action=GuardrailAction.ALLOW)

        if self._action == GuardrailAction.MODIFY:
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                triggered=True,
                reason=f"Filtered {len(found)} profane words",
                modified_content=modified,
            )

        return GuardrailResult(
            action=self._action,
            triggered=True,
            reason=f"Content contains profanity: {', '.join(found)}",
        )


class TopicGuardrail(Guardrail):
    """Blocks content about restricted topics."""

    name = "topic_guardrail"

    def __init__(self, blocked_topics: list[str] | None = None):
        self._topics = blocked_topics or []
        self._patterns = [
            (topic, __import__("re").compile(rf"\b{topic}\b", __import__("re").IGNORECASE))
            for topic in self._topics
        ]

    async def check(self, content: str, context: dict[str, Any]) -> GuardrailResult:
        matched = []

        for topic, pattern in self._patterns:
            if pattern.search(content):
                matched.append(topic)

        if not matched:
            return GuardrailResult(action=GuardrailAction.ALLOW)

        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            triggered=True,
            reason=f"Content about restricted topics: {', '.join(matched)}",
            metadata={"blocked_topics": matched},
        )


class PIIGuardrail(OutputGuardrail):
    """Prevents PII in outputs."""

    name = "pii_guardrail"

    def __init__(self, redact: bool = True):
        self._redact = redact

        import re

        self._patterns = [
            ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
            ("phone", re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")),
            ("ssn", re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b")),
            ("credit_card", re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")),
        ]

    async def check(
        self,
        output: str,
        input_content: str,
        context: dict[str, Any],
    ) -> GuardrailResult:
        found = []
        modified = output

        for pii_type, pattern in self._patterns:
            matches = pattern.findall(output)
            if matches:
                found.extend([(pii_type, m) for m in matches])
                if self._redact:
                    modified = pattern.sub(f"[REDACTED_{pii_type.upper()}]", modified)

        if not found:
            return GuardrailResult(action=GuardrailAction.ALLOW)

        if self._redact:
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                triggered=True,
                reason=f"Redacted {len(found)} PII instances",
                modified_content=modified,
                metadata={"pii_types": list({t for t, _ in found})},
            )

        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            triggered=True,
            reason="Output contains PII",
            metadata={"pii_types": list({t for t, _ in found})},
        )


class HallucinationGuardrail(OutputGuardrail):
    """
    Checks for potential hallucinations.

    Note: This is a simplified implementation.
    Production systems should use more sophisticated methods.
    """

    name = "hallucination_guardrail"

    def __init__(self, confidence_threshold: float = 0.5):
        self._threshold = confidence_threshold

    async def check(
        self,
        output: str,
        input_content: str,
        context: dict[str, Any],
    ) -> GuardrailResult:
        # Simple heuristic checks
        issues = []

        # Check for definitive claims about uncertain topics
        uncertain_phrases = [
            "I'm not sure",
            "I believe",
            "probably",
            "might be",
            "I think",
        ]

        confident_claims = [
            "definitely",
            "certainly",
            "always",
            "never",
            "guaranteed",
        ]

        has_uncertainty = any(p.lower() in output.lower() for p in uncertain_phrases)
        has_confidence = any(p.lower() in output.lower() for p in confident_claims)

        # Flag if confident claims without supporting context
        if has_confidence and not has_uncertainty:
            issues.append("Contains definitive claims without hedging")

        # Check for fabricated citations
        import re

        citation_pattern = re.compile(r"\[\d+\]|\(\d{4}\)|\([A-Z][a-z]+,?\s+\d{4}\)")
        has_citations = bool(citation_pattern.search(output))

        if has_citations:
            issues.append("Contains citation-like patterns (verify sources)")

        if not issues:
            return GuardrailResult(action=GuardrailAction.ALLOW)

        return GuardrailResult(
            action=GuardrailAction.WARN,
            triggered=True,
            reason="; ".join(issues),
            metadata={"issues": issues},
        )


class GuardrailChain:
    """
    Chains multiple guardrails together.

    Guardrails are evaluated in order, and processing
    stops on first BLOCK action.
    """

    def __init__(
        self,
        input_guardrails: list[Guardrail] | None = None,
        output_guardrails: list[OutputGuardrail] | None = None,
    ):
        self._input_guardrails = input_guardrails or []
        self._output_guardrails = output_guardrails or []

    def add_input_guardrail(self, guardrail: Guardrail) -> "GuardrailChain":
        """Add an input guardrail."""
        self._input_guardrails.append(guardrail)
        return self

    def add_output_guardrail(self, guardrail: OutputGuardrail) -> "GuardrailChain":
        """Add an output guardrail."""
        self._output_guardrails.append(guardrail)
        return self

    async def check_input(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, list[GuardrailResult]]:
        """
        Run all input guardrails.

        Returns (possibly modified content, list of results).
        Raises if any guardrail returns BLOCK.
        """
        context = context or {}
        results = []
        current = content

        for guardrail in self._input_guardrails:
            result = await guardrail.check(current, context)
            results.append(result)

            if result.action == GuardrailAction.BLOCK:
                from src.core.exceptions import SafetyError

                raise SafetyError(f"Input blocked by {guardrail.name}: {result.reason}")

            if result.action == GuardrailAction.MODIFY and result.modified_content:
                current = result.modified_content

        return current, results

    async def check_output(
        self,
        output: str,
        input_content: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, list[GuardrailResult]]:
        """
        Run all output guardrails.

        Returns (possibly modified output, list of results).
        """
        context = context or {}
        results = []
        current = output

        for guardrail in self._output_guardrails:
            result = await guardrail.check(current, input_content, context)
            results.append(result)

            if result.action == GuardrailAction.BLOCK:
                from src.core.exceptions import SafetyError

                raise SafetyError(f"Output blocked by {guardrail.name}: {result.reason}")

            if result.action == GuardrailAction.MODIFY and result.modified_content:
                current = result.modified_content

        return current, results


def create_default_guardrail_chain() -> GuardrailChain:
    """Create a guardrail chain with sensible defaults."""
    return GuardrailChain(
        input_guardrails=[
            LengthGuardrail(max_length=100000),
        ],
        output_guardrails=[
            PIIGuardrail(redact=True),
            HallucinationGuardrail(),
        ],
    )
