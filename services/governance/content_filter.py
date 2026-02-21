"""
Content Filter

Content filtering and moderation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import re


class FilterLevel(str, Enum):
    """Content filter strictness levels."""

    STRICT = "strict"
    MODERATE = "moderate"
    LOW = "low"
    OFF = "off"


class ContentCategory(str, Enum):
    """Content categories for filtering."""

    HATE = "hate"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"
    DANGEROUS = "dangerous"


@dataclass
class FilterResult:
    """Result of content filtering."""

    passed: bool = True
    filtered_content: str = ""
    categories_triggered: list[str] = field(default_factory=list)
    confidence_scores: dict[str, float] = field(default_factory=dict)
    action_taken: str = "none"  # none, warn, block, redact


class ContentFilter:
    """
    Content filtering service.

    Filters content based on safety categories.
    """

    def __init__(
        self,
        level: FilterLevel = FilterLevel.MODERATE,
    ):
        self._level = level

        # Category patterns (simplified for offline mode)
        self._patterns: dict[ContentCategory, list[str]] = {
            ContentCategory.HATE: [
                r"\b(hate|hateful)\s+(speech|content)",
                r"\b(racist|racism|sexist|sexism)\b",
                r"discriminat(e|ion|ory)",
            ],
            ContentCategory.VIOLENCE: [
                r"\b(kill|murder|attack)\s+(people|someone)",
                r"(threat|threaten).*violence",
                r"\bloose\s+violence\b",
            ],
            ContentCategory.SEXUAL: [
                r"\bexplicit\s+sexual\b",
                r"\bpornograph(y|ic)\b",
            ],
            ContentCategory.HARASSMENT: [
                r"\b(harass|bully|stalk)\b",
                r"target(ing)?\s+(individual|person)",
            ],
            ContentCategory.SELF_HARM: [
                r"\bsuicid(e|al)\s+(method|instruction)",
                r"\bself[\s-]?harm\s+(instruction|how)",
            ],
            ContentCategory.DANGEROUS: [
                r"\b(make|build)\s+(bomb|explosive|weapon)",
                r"(illegal|illicit)\s+drug\s+(synth|mak|creat)",
            ],
        }

        # Thresholds by level
        self._thresholds = {
            FilterLevel.STRICT: 0.3,
            FilterLevel.MODERATE: 0.5,
            FilterLevel.LOW: 0.7,
            FilterLevel.OFF: 1.1,
        }

    def filter(
        self,
        content: str,
        categories: list[ContentCategory] | None = None,
    ) -> FilterResult:
        """
        Filter content.

        Args:
            content: Content to filter
            categories: Categories to check (all if None)

        Returns:
            Filter result
        """
        result = FilterResult(filtered_content=content)

        if self._level == FilterLevel.OFF:
            return result

        threshold = self._thresholds[self._level]
        categories_to_check = categories or list(ContentCategory)

        for category in categories_to_check:
            score = self._calculate_category_score(content, category)
            result.confidence_scores[category.value] = score

            if score >= threshold:
                result.categories_triggered.append(category.value)

        # Determine action
        if result.categories_triggered:
            if self._level == FilterLevel.STRICT:
                result.passed = False
                result.action_taken = "block"
            elif self._level == FilterLevel.MODERATE:
                # Block only dangerous content
                dangerous_categories = {
                    ContentCategory.VIOLENCE.value,
                    ContentCategory.SELF_HARM.value,
                    ContentCategory.DANGEROUS.value,
                }
                if any(c in dangerous_categories for c in result.categories_triggered):
                    result.passed = False
                    result.action_taken = "block"
                else:
                    result.action_taken = "warn"
            else:  # LOW
                result.action_taken = "warn"

        return result

    async def filter_async(
        self,
        content: str,
        categories: list[ContentCategory] | None = None,
    ) -> FilterResult:
        """Async filter."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, self.filter, content, categories
        )

    def _calculate_category_score(
        self,
        content: str,
        category: ContentCategory,
    ) -> float:
        """Calculate score for a category (0-1)."""
        patterns = self._patterns.get(category, [])
        if not patterns:
            return 0.0

        matches = 0
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                matches += 1

        # Normalize score
        return min(matches / len(patterns), 1.0)

    def set_level(self, level: FilterLevel) -> None:
        """Set filter level."""
        self._level = level

    def add_pattern(self, category: ContentCategory, pattern: str) -> None:
        """Add a pattern to a category."""
        if category not in self._patterns:
            self._patterns[category] = []
        self._patterns[category].append(pattern)

    def redact(
        self,
        content: str,
        categories: list[ContentCategory] | None = None,
    ) -> str:
        """Redact content matching categories."""
        categories_to_check = categories or list(ContentCategory)
        redacted = content

        for category in categories_to_check:
            patterns = self._patterns.get(category, [])
            for pattern in patterns:
                redacted = re.sub(
                    pattern,
                    "[CONTENT FILTERED]",
                    redacted,
                    flags=re.IGNORECASE,
                )

        return redacted
