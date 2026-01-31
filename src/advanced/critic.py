"""
Critic Agent

Self-critique and quality improvement.

Design decisions:
- Multiple critique dimensions
- Iterative refinement
- Confidence scoring
- Action suggestions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.core.types import Message


class CritiqueDimension(str, Enum):
    """Dimensions for critique."""
    
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    COHERENCE = "coherence"
    SAFETY = "safety"
    HELPFULNESS = "helpfulness"


class CritiqueAction(str, Enum):
    """Suggested actions from critique."""
    
    ACCEPT = "accept"  # Response is good
    REVISE = "revise"  # Needs minor changes
    REGENERATE = "regenerate"  # Needs major rework
    ESCALATE = "escalate"  # Needs human review
    BLOCK = "block"  # Should not be sent


@dataclass
class CritiqueFinding:
    """A single finding from critique."""
    
    dimension: CritiqueDimension
    score: float  # 0.0 - 1.0
    issue: str | None = None
    suggestion: str | None = None
    severity: str = "low"  # low, medium, high


@dataclass
class Critique:
    """
    Complete critique of a response.
    """
    
    # Overall assessment
    overall_score: float = 0.0
    action: CritiqueAction = CritiqueAction.ACCEPT
    
    # Detailed findings
    findings: list[CritiqueFinding] = field(default_factory=list)
    
    # Suggestions
    improved_response: str | None = None
    improvement_notes: str | None = None
    
    # Confidence
    confidence: float = 1.0
    
    def get_dimension_score(self, dimension: CritiqueDimension) -> float | None:
        """Get score for a specific dimension."""
        for finding in self.findings:
            if finding.dimension == dimension:
                return finding.score
        return None


class CriticStrategy(ABC):
    """Base class for critique strategies."""
    
    @abstractmethod
    async def critique(
        self,
        input_text: str,
        output_text: str,
        context: dict[str, Any] | None = None,
    ) -> Critique:
        """Critique a response."""
        pass


class RuleBasedCritic(CriticStrategy):
    """
    Rule-based critique.
    
    Uses heuristics for fast, consistent critiques.
    """
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 10000,
        required_keywords: list[str] | None = None,
    ):
        self._min_length = min_length
        self._max_length = max_length
        self._required = required_keywords or []
    
    async def critique(
        self,
        input_text: str,
        output_text: str,
        context: dict[str, Any] | None = None,
    ) -> Critique:
        findings = []
        
        # Check length
        length = len(output_text)
        if length < self._min_length:
            findings.append(CritiqueFinding(
                dimension=CritiqueDimension.COMPLETENESS,
                score=0.3,
                issue="Response too short",
                suggestion="Provide more detailed response",
                severity="high",
            ))
        elif length > self._max_length:
            findings.append(CritiqueFinding(
                dimension=CritiqueDimension.CLARITY,
                score=0.5,
                issue="Response too long",
                suggestion="Make response more concise",
                severity="medium",
            ))
        else:
            findings.append(CritiqueFinding(
                dimension=CritiqueDimension.COMPLETENESS,
                score=0.9,
            ))
        
        # Check for required keywords
        for keyword in self._required:
            if keyword.lower() not in output_text.lower():
                findings.append(CritiqueFinding(
                    dimension=CritiqueDimension.RELEVANCE,
                    score=0.5,
                    issue=f"Missing expected term: {keyword}",
                    severity="medium",
                ))
        
        # Check for safety issues
        safety_issues = self._check_safety(output_text)
        if safety_issues:
            findings.append(CritiqueFinding(
                dimension=CritiqueDimension.SAFETY,
                score=0.0,
                issue="; ".join(safety_issues),
                severity="high",
            ))
        else:
            findings.append(CritiqueFinding(
                dimension=CritiqueDimension.SAFETY,
                score=1.0,
            ))
        
        # Calculate overall score
        if findings:
            overall = sum(f.score for f in findings) / len(findings)
        else:
            overall = 1.0
        
        # Determine action
        if any(f.severity == "high" for f in findings):
            if any(f.dimension == CritiqueDimension.SAFETY and f.score < 0.5 for f in findings):
                action = CritiqueAction.BLOCK
            else:
                action = CritiqueAction.REGENERATE
        elif overall < 0.7:
            action = CritiqueAction.REVISE
        else:
            action = CritiqueAction.ACCEPT
        
        return Critique(
            overall_score=overall,
            action=action,
            findings=findings,
            confidence=0.8,
        )
    
    def _check_safety(self, text: str) -> list[str]:
        """Check for safety issues."""
        issues = []
        
        # Simple pattern checks
        import re
        
        patterns = [
            (r"(?i)\b(password|secret|api.?key)\s*[:=]\s*\S+", "Contains potential credential"),
            (r"(?i)(fuck|shit|damn)", "Contains profanity"),
        ]
        
        for pattern, issue in patterns:
            if re.search(pattern, text):
                issues.append(issue)
        
        return issues


class LLMCritic(CriticStrategy):
    """
    LLM-powered critique.
    
    Uses an LLM to provide nuanced critique.
    """
    
    CRITIQUE_PROMPT = """You are a critical reviewer. Evaluate the following AI response.

User Input: {input}

AI Response: {output}

Evaluate the response on these dimensions (score 0-10):
1. Accuracy - Is the information correct?
2. Relevance - Does it address the user's request?
3. Completeness - Is the response thorough?
4. Clarity - Is it easy to understand?
5. Helpfulness - Would this help the user?
6. Safety - Is it appropriate and safe?

Also provide:
- Overall assessment
- Specific issues found
- Suggested improvements
- Recommended action: ACCEPT, REVISE, REGENERATE, or BLOCK

Respond in JSON format:
{{
    "dimensions": {{
        "accuracy": {{"score": 0-10, "issue": "..."}},
        "relevance": {{"score": 0-10, "issue": "..."}},
        "completeness": {{"score": 0-10, "issue": "..."}},
        "clarity": {{"score": 0-10, "issue": "..."}},
        "helpfulness": {{"score": 0-10, "issue": "..."}},
        "safety": {{"score": 0-10, "issue": "..."}}
    }},
    "overall_score": 0-10,
    "action": "ACCEPT|REVISE|REGENERATE|BLOCK",
    "improvements": ["improvement 1", "..."],
    "improved_response": "Optional improved version"
}}"""
    
    def __init__(self, llm_adapter):
        self._llm = llm_adapter
    
    async def critique(
        self,
        input_text: str,
        output_text: str,
        context: dict[str, Any] | None = None,
    ) -> Critique:
        import json
        
        prompt = self.CRITIQUE_PROMPT.format(
            input=input_text,
            output=output_text,
        )
        
        try:
            response = await self._llm.complete([Message(role="user", content=prompt)])
            
            # Parse JSON
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            # Build findings
            findings = []
            dimension_map = {
                "accuracy": CritiqueDimension.ACCURACY,
                "relevance": CritiqueDimension.RELEVANCE,
                "completeness": CritiqueDimension.COMPLETENESS,
                "clarity": CritiqueDimension.CLARITY,
                "helpfulness": CritiqueDimension.HELPFULNESS,
                "safety": CritiqueDimension.SAFETY,
            }
            
            for name, dim in dimension_map.items():
                dim_data = data.get("dimensions", {}).get(name, {})
                score = dim_data.get("score", 5) / 10.0
                issue = dim_data.get("issue")
                
                findings.append(CritiqueFinding(
                    dimension=dim,
                    score=score,
                    issue=issue if issue else None,
                    severity="high" if score < 0.3 else "medium" if score < 0.7 else "low",
                ))
            
            # Map action
            action_str = data.get("action", "ACCEPT").upper()
            action = {
                "ACCEPT": CritiqueAction.ACCEPT,
                "REVISE": CritiqueAction.REVISE,
                "REGENERATE": CritiqueAction.REGENERATE,
                "BLOCK": CritiqueAction.BLOCK,
            }.get(action_str, CritiqueAction.ACCEPT)
            
            return Critique(
                overall_score=data.get("overall_score", 5) / 10.0,
                action=action,
                findings=findings,
                improved_response=data.get("improved_response"),
                improvement_notes=", ".join(data.get("improvements", [])),
                confidence=0.9,
            )
            
        except Exception as e:
            # Fallback to accepting on error
            return Critique(
                overall_score=0.5,
                action=CritiqueAction.ACCEPT,
                confidence=0.1,
            )


class CriticAgent:
    """
    Main critic agent interface.
    
    Combines multiple strategies and provides
    iterative refinement.
    """
    
    def __init__(
        self,
        strategies: list[CriticStrategy] | None = None,
        threshold: float = 0.7,
        max_iterations: int = 3,
    ):
        self._strategies = strategies or [RuleBasedCritic()]
        self._threshold = threshold
        self._max_iterations = max_iterations
    
    async def critique(
        self,
        input_text: str,
        output_text: str,
        context: dict[str, Any] | None = None,
    ) -> Critique:
        """
        Run all critique strategies and combine results.
        """
        all_findings = []
        action_votes = []
        scores = []
        
        for strategy in self._strategies:
            critique = await strategy.critique(input_text, output_text, context)
            all_findings.extend(critique.findings)
            action_votes.append(critique.action)
            scores.append(critique.overall_score)
        
        # Combine results
        overall_score = sum(scores) / len(scores) if scores else 0.5
        
        # Vote on action (most restrictive wins)
        action_priority = {
            CritiqueAction.BLOCK: 0,
            CritiqueAction.REGENERATE: 1,
            CritiqueAction.REVISE: 2,
            CritiqueAction.ESCALATE: 3,
            CritiqueAction.ACCEPT: 4,
        }
        action = min(action_votes, key=lambda a: action_priority[a])
        
        return Critique(
            overall_score=overall_score,
            action=action,
            findings=all_findings,
            confidence=1 / len(self._strategies) if self._strategies else 0.5,
        )
    
    async def refine(
        self,
        input_text: str,
        output_text: str,
        generator,  # Callable to generate new response
        context: dict[str, Any] | None = None,
    ) -> tuple[str, Critique]:
        """
        Iteratively refine a response using critique.
        
        Returns (final_response, final_critique).
        """
        current_output = output_text
        
        for i in range(self._max_iterations):
            critique = await self.critique(input_text, current_output, context)
            
            if critique.action == CritiqueAction.ACCEPT:
                return current_output, critique
            
            if critique.action == CritiqueAction.BLOCK:
                return "", critique
            
            if critique.improved_response:
                current_output = critique.improved_response
            else:
                # Generate new response with feedback
                feedback = self._build_feedback(critique)
                current_output = await generator(input_text, feedback)
        
        # Return best effort after max iterations
        final_critique = await self.critique(input_text, current_output, context)
        return current_output, final_critique
    
    def _build_feedback(self, critique: Critique) -> str:
        """Build feedback string from critique."""
        issues = [f.issue for f in critique.findings if f.issue]
        suggestions = [f.suggestion for f in critique.findings if f.suggestion]
        
        feedback_parts = []
        
        if issues:
            feedback_parts.append("Issues: " + "; ".join(issues))
        
        if suggestions:
            feedback_parts.append("Suggestions: " + "; ".join(suggestions))
        
        if critique.improvement_notes:
            feedback_parts.append(f"Notes: {critique.improvement_notes}")
        
        return "\n".join(feedback_parts)
