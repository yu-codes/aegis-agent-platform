"""
Injection Guard

Prompt injection and jailbreak detection.

Based on: src/safety/input_validation.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import re


class ThreatType(str, Enum):
    """Types of injection threats."""

    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    CONTEXT_MANIPULATION = "context_manipulation"
    INSTRUCTION_OVERRIDE = "instruction_override"
    DATA_EXTRACTION = "data_extraction"


@dataclass
class ThreatResult:
    """Result of threat detection."""

    is_threat: bool = False
    threat_type: ThreatType | None = None
    confidence: float = 0.0
    details: str = ""
    matched_patterns: list[str] = field(default_factory=list)


class InjectionGuard:
    """
    Prompt injection and jailbreak detection.

    Detects various injection attempts:
    - Prompt injection
    - Jailbreak attempts
    - Context manipulation
    - Instruction override
    """

    def __init__(self, sensitivity: float = 0.5):
        self._sensitivity = sensitivity

        # Injection patterns
        self._injection_patterns = [
            # Direct instruction override
            r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
            r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
            r"forget\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
            r"override\s+(system|previous)\s+(prompt|instructions?)",
            # Role manipulation
            r"you\s+are\s+(now|actually)\s+a",
            r"act\s+as\s+if\s+you\s+have\s+no\s+(restrictions?|limitations?)",
            r"pretend\s+(you\s+are|to\s+be)\s+a",
            r"roleplay\s+as\s+a\s+(evil|unrestricted|unlimited)",
            # System prompt extraction
            r"(reveal|show|display|print)\s+(your|the)\s+(system\s+)?prompt",
            r"what\s+is\s+(your|the)\s+(system\s+)?prompt",
            r"(output|repeat)\s+(your|the)\s+(system\s+)?prompt",
        ]

        # Jailbreak patterns
        self._jailbreak_patterns = [
            r"DAN\s+(mode|prompt)",
            r"developer\s+mode",
            r"(enable|activate)\s+(unrestricted|unlimited)\s+mode",
            r"jailbreak",
            r"bypass\s+safety",
            r"remove\s+(all\s+)?restrictions",
            r"without\s+any\s+restrictions",
        ]

        # Context manipulation patterns
        self._context_patterns = [
            r"in\s+this\s+(hypothetical|fictional)\s+scenario",
            r"for\s+(educational|research)\s+purposes\s+only",
            r"imagine\s+you\s+are\s+in\s+a\s+world\s+where",
            r"this\s+is\s+(just\s+)?a\s+(test|experiment)",
            r"pretend\s+this\s+is\s+(legal|acceptable)",
        ]

        # Data extraction patterns
        self._extraction_patterns = [
            r"(list|show|reveal)\s+(all\s+)?(your|the)\s+(tools?|functions?|capabilities)",
            r"what\s+(tools?|functions?)\s+(do\s+you\s+have|are\s+available)",
            r"(dump|extract|output)\s+(your|the|all)\s+(data|information|memory)",
        ]

    def detect(self, text: str) -> ThreatResult:
        """
        Detect injection attempts.

        Args:
            text: Text to analyze

        Returns:
            Threat detection result
        """
        result = ThreatResult()
        text_lower = text.lower()

        # Check each threat type
        threat_checks = [
            (ThreatType.PROMPT_INJECTION, self._injection_patterns),
            (ThreatType.JAILBREAK, self._jailbreak_patterns),
            (ThreatType.CONTEXT_MANIPULATION, self._context_patterns),
            (ThreatType.DATA_EXTRACTION, self._extraction_patterns),
        ]

        max_confidence = 0.0
        max_threat = None
        all_matches = []

        for threat_type, patterns in threat_checks:
            matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches.append(pattern)

            if matches:
                # Calculate confidence based on number of matches
                confidence = min(len(matches) / len(patterns) + 0.3, 1.0)

                if confidence > max_confidence:
                    max_confidence = confidence
                    max_threat = threat_type

                all_matches.extend(matches)

        if max_confidence >= self._sensitivity:
            result.is_threat = True
            result.threat_type = max_threat
            result.confidence = max_confidence
            result.matched_patterns = all_matches
            result.details = (
                f"Detected {max_threat.value} attempt with {len(all_matches)} pattern matches"
            )

        return result

    async def detect_async(self, text: str) -> ThreatResult:
        """Async detection."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(None, self.detect, text)

    def sanitize(self, text: str) -> tuple[str, bool]:
        """
        Sanitize text by removing/escaping dangerous patterns.

        Returns:
            Tuple of (sanitized_text, was_modified)
        """
        sanitized = text
        modified = False

        # Remove common injection markers
        dangerous_markers = [
            r"\[SYSTEM\]",
            r"\[INST\]",
            r"\[/INST\]",
            r"<\|system\|>",
            r"<\|user\|>",
            r"<\|assistant\|>",
            r"###\s*System:",
            r"###\s*Human:",
            r"###\s*Assistant:",
        ]

        for marker in dangerous_markers:
            if re.search(marker, sanitized, re.IGNORECASE):
                sanitized = re.sub(marker, "", sanitized, flags=re.IGNORECASE)
                modified = True

        # Escape common delimiters
        delimiters = ["```", "---", "==="]
        for delimiter in delimiters:
            if delimiter in sanitized:
                sanitized = sanitized.replace(delimiter, " ")
                modified = True

        return sanitized, modified

    def set_sensitivity(self, sensitivity: float) -> None:
        """Set detection sensitivity (0-1)."""
        self._sensitivity = max(0.0, min(1.0, sensitivity))

    def add_pattern(self, threat_type: ThreatType, pattern: str) -> None:
        """Add a custom pattern."""
        if threat_type == ThreatType.PROMPT_INJECTION:
            self._injection_patterns.append(pattern)
        elif threat_type == ThreatType.JAILBREAK:
            self._jailbreak_patterns.append(pattern)
        elif threat_type == ThreatType.CONTEXT_MANIPULATION:
            self._context_patterns.append(pattern)
        elif threat_type == ThreatType.DATA_EXTRACTION:
            self._extraction_patterns.append(pattern)
