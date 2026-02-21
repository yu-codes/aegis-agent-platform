"""
Hallucination Checker

Detect hallucinations in generated content.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol
import re


@dataclass
class HallucinationResult:
    """Result of hallucination check."""

    has_hallucination: bool = False
    confidence: float = 0.0
    hallucinated_claims: list[str] = field(default_factory=list)
    verified_claims: list[str] = field(default_factory=list)
    unverifiable_claims: list[str] = field(default_factory=list)
    explanation: str = ""


class LLMProtocol(Protocol):
    """Protocol for LLM checking."""

    async def complete(self, prompt: str) -> str: ...


class HallucinationChecker:
    """
    Hallucination detection.

    Detects claims not grounded in provided context.
    """

    def __init__(self, llm: LLMProtocol | None = None):
        self._llm = llm

    async def check(
        self,
        generated_text: str,
        context: str | list[str],
    ) -> HallucinationResult:
        """
        Check for hallucinations.

        Args:
            generated_text: Text to check
            context: Source context(s)

        Returns:
            Hallucination check result
        """
        if isinstance(context, list):
            context = "\n\n".join(context)

        if self._llm:
            return await self._llm_check(generated_text, context)
        else:
            return self._heuristic_check(generated_text, context)

    async def _llm_check(
        self,
        generated_text: str,
        context: str,
    ) -> HallucinationResult:
        """Use LLM to check for hallucinations."""
        prompt = f"""Analyze the following generated text for hallucinations.
A hallucination is a claim that cannot be verified from the given context.

Context:
{context}

Generated Text:
{generated_text}

List any claims in the generated text that are NOT supported by the context.
For each claim, explain why it's a hallucination.

Format your response as:
HALLUCINATED_CLAIMS:
- [claim 1]
- [claim 2]

VERIFIED_CLAIMS:
- [verified claim 1]

CONFIDENCE: [0-100]%

If there are no hallucinations, respond with:
HALLUCINATED_CLAIMS: None
VERIFIED_CLAIMS: [list verified claims]
CONFIDENCE: [0-100]%"""

        response = await self._llm.complete(prompt)
        return self._parse_llm_response(response)

    def _heuristic_check(
        self,
        generated_text: str,
        context: str,
    ) -> HallucinationResult:
        """Heuristic-based hallucination check."""
        result = HallucinationResult()

        # Extract sentences/claims from generated text
        sentences = re.split(r"[.!?]", generated_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        context_lower = context.lower()
        context_terms = set(context_lower.split())

        for sentence in sentences:
            sentence_terms = set(sentence.lower().split())

            # Check overlap with context
            overlap = len(sentence_terms & context_terms)
            overlap_ratio = overlap / len(sentence_terms) if sentence_terms else 0

            if overlap_ratio > 0.6:
                result.verified_claims.append(sentence)
            elif overlap_ratio > 0.3:
                result.unverifiable_claims.append(sentence)
            else:
                result.hallucinated_claims.append(sentence)

        # Calculate confidence
        total = len(sentences)
        if total > 0:
            hallucination_ratio = len(result.hallucinated_claims) / total
            result.has_hallucination = len(result.hallucinated_claims) > 0
            result.confidence = hallucination_ratio

            if result.hallucinated_claims:
                result.explanation = f"Found {len(result.hallucinated_claims)} potential hallucinations out of {total} claims"
            else:
                result.explanation = f"All {total} claims appear to be grounded in context"

        return result

    def _parse_llm_response(self, response: str) -> HallucinationResult:
        """Parse LLM hallucination check response."""
        result = HallucinationResult()

        lines = response.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "HALLUCINATED_CLAIMS" in line.upper():
                current_section = "hallucinated"
                if "none" in line.lower():
                    current_section = None
            elif "VERIFIED_CLAIMS" in line.upper():
                current_section = "verified"
            elif "UNVERIFIABLE" in line.upper():
                current_section = "unverifiable"
            elif "CONFIDENCE" in line.upper():
                # Extract confidence percentage
                numbers = re.findall(r"\d+", line)
                if numbers:
                    result.confidence = int(numbers[0]) / 100
                current_section = None
            elif line.startswith("-") or line.startswith("*"):
                claim = line.lstrip("-* ").strip()
                if current_section == "hallucinated":
                    result.hallucinated_claims.append(claim)
                elif current_section == "verified":
                    result.verified_claims.append(claim)
                elif current_section == "unverifiable":
                    result.unverifiable_claims.append(claim)

        result.has_hallucination = len(result.hallucinated_claims) > 0

        if result.hallucinated_claims:
            result.explanation = f"Detected {len(result.hallucinated_claims)} hallucinated claims"
        else:
            result.explanation = "No hallucinations detected"

        return result

    def check_claims(
        self,
        claims: list[str],
        context: str,
    ) -> dict[str, bool]:
        """
        Check individual claims against context.

        Returns dict mapping claim to verified status.
        """
        context_lower = context.lower()
        context_terms = set(context_lower.split())

        results = {}

        for claim in claims:
            claim_terms = set(claim.lower().split())
            overlap = len(claim_terms & context_terms)
            overlap_ratio = overlap / len(claim_terms) if claim_terms else 0
            results[claim] = overlap_ratio > 0.5

        return results
