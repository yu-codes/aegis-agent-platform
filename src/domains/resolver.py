"""
Domain Resolver

Determines which domain to use for a given request.

Resolution strategies:
1. EXPLICIT: Domain specified in request (highest priority)
2. INFERRED: Lightweight classification of input
3. CONTEXT: Based on session/user metadata
4. FALLBACK: Safe default domain (guaranteed)

Design decisions:
- Resolver is stateless - registry holds state
- Inference is optional and configurable
- Fallback is ALWAYS guaranteed
- Resolution is auditable (returns full result)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol
import logging
import re

from src.domains.profile import DomainProfile
from src.domains.registry import DomainRegistry


logger = logging.getLogger(__name__)


class ResolutionMethod(str, Enum):
    """How the domain was resolved."""
    
    EXPLICIT = "explicit"  # Specified in request
    INFERRED = "inferred"  # Classified from content
    CONTEXT = "context"  # From session/user metadata
    FALLBACK = "fallback"  # Default domain used


@dataclass
class ResolutionResult:
    """
    Result of domain resolution.
    
    Contains the resolved profile plus metadata about how
    it was resolved (for auditing and debugging).
    """
    
    profile: DomainProfile
    method: ResolutionMethod
    confidence: float = 1.0  # 0.0-1.0, only meaningful for INFERRED
    
    # For INFERRED: what triggered the match
    matched_patterns: list[str] = field(default_factory=list)
    
    # Alternative candidates considered (for debugging)
    candidates: list[tuple[str, float]] = field(default_factory=list)
    
    # Original request
    requested_domain: str | None = None
    
    def __repr__(self) -> str:
        return (
            f"ResolutionResult(domain={self.profile.name!r}, "
            f"method={self.method.value}, confidence={self.confidence:.2f})"
        )


class DomainClassifier(Protocol):
    """Protocol for custom domain classifiers."""
    
    async def classify(
        self,
        content: str,
        context: dict[str, Any],
    ) -> list[tuple[str, float]]:
        """
        Classify content to domain(s) with confidence scores.
        
        Args:
            content: Input text to classify
            context: Additional context (user, session, etc.)
            
        Returns:
            List of (domain_name, confidence) sorted by confidence desc
        """
        ...


@dataclass
class KeywordRule:
    """A keyword-based classification rule."""
    
    domain: str
    keywords: list[str]
    patterns: list[str] = field(default_factory=list)  # Regex patterns
    weight: float = 1.0
    
    # Compiled patterns (lazy)
    _compiled: list[re.Pattern] | None = field(default=None, repr=False)
    
    def get_patterns(self) -> list[re.Pattern]:
        """Get compiled regex patterns."""
        if self._compiled is None:
            self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]
        return self._compiled


class KeywordClassifier:
    """
    Simple keyword-based domain classifier.
    
    Uses keyword matching and regex patterns for classification.
    Suitable for deterministic, low-latency inference.
    """
    
    def __init__(self, rules: list[KeywordRule] | None = None):
        self._rules = rules or []
    
    def add_rule(self, rule: KeywordRule) -> None:
        """Add a classification rule."""
        self._rules.append(rule)
    
    async def classify(
        self,
        content: str,
        context: dict[str, Any],
    ) -> list[tuple[str, float]]:
        """Classify content using keyword/pattern matching."""
        content_lower = content.lower()
        scores: dict[str, float] = {}
        
        for rule in self._rules:
            score = 0.0
            
            # Keyword matching
            for keyword in rule.keywords:
                if keyword.lower() in content_lower:
                    score += rule.weight
            
            # Pattern matching
            for pattern in rule.get_patterns():
                if pattern.search(content):
                    score += rule.weight * 1.5  # Patterns slightly stronger
            
            if score > 0:
                current = scores.get(rule.domain, 0.0)
                scores[rule.domain] = max(current, score)
        
        # Normalize scores to 0-1
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: min(v / max_score, 1.0) for k, v in scores.items()}
        
        # Sort by score descending
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class DomainResolver:
    """
    Resolves which domain profile to use for a request.
    
    Resolution order:
    1. Explicit domain (if provided and exists)
    2. Inferred domain (if classifier enabled and confident)
    3. Context domain (from session/user metadata)
    4. Fallback domain (always succeeds)
    
    Thread-safe: Uses registry's locking.
    
    Usage:
        resolver = DomainResolver(registry)
        result = await resolver.resolve(
            explicit_domain="financial_analysis",
            content="What's my portfolio balance?",
            context={"user_id": "123"},
        )
        profile = result.profile
    """
    
    def __init__(
        self,
        registry: DomainRegistry,
        classifier: DomainClassifier | KeywordClassifier | None = None,
        inference_threshold: float = 0.6,
        enable_inference: bool = True,
    ):
        """
        Initialize resolver.
        
        Args:
            registry: Domain registry to resolve from
            classifier: Optional classifier for inference
            inference_threshold: Min confidence for inferred domain
            enable_inference: Whether to enable domain inference
        """
        self._registry = registry
        self._classifier = classifier or KeywordClassifier()
        self._threshold = inference_threshold
        self._enable_inference = enable_inference
    
    async def resolve(
        self,
        explicit_domain: str | None = None,
        content: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ResolutionResult:
        """
        Resolve domain for a request.
        
        Args:
            explicit_domain: Explicitly requested domain (highest priority)
            content: Input content for inference
            context: Request context (session, user, metadata)
            
        Returns:
            ResolutionResult with profile and resolution metadata
        """
        context = context or {}
        
        # 1. EXPLICIT: Use requested domain if valid
        if explicit_domain:
            if self._registry.exists(explicit_domain):
                profile = self._registry.get(explicit_domain)
                logger.debug(f"Domain resolved explicitly: {explicit_domain}")
                return ResolutionResult(
                    profile=profile,
                    method=ResolutionMethod.EXPLICIT,
                    confidence=1.0,
                    requested_domain=explicit_domain,
                )
            else:
                logger.warning(
                    f"Requested domain '{explicit_domain}' not found, "
                    "falling back to inference/default"
                )
        
        # 2. CONTEXT: Check for domain in session/user metadata
        context_domain = self._get_domain_from_context(context)
        if context_domain and self._registry.exists(context_domain):
            profile = self._registry.get(context_domain)
            logger.debug(f"Domain resolved from context: {context_domain}")
            return ResolutionResult(
                profile=profile,
                method=ResolutionMethod.CONTEXT,
                confidence=1.0,
                requested_domain=explicit_domain,
            )
        
        # 3. INFERRED: Classify content if enabled
        if self._enable_inference and content:
            result = await self._infer_domain(content, context)
            if result:
                logger.debug(
                    f"Domain inferred: {result.profile.name} "
                    f"(confidence={result.confidence:.2f})"
                )
                result.requested_domain = explicit_domain
                return result
        
        # 4. FALLBACK: Always succeeds
        logger.debug(f"Domain resolved to fallback: {self._registry.fallback.name}")
        return ResolutionResult(
            profile=self._registry.fallback,
            method=ResolutionMethod.FALLBACK,
            confidence=1.0,
            requested_domain=explicit_domain,
        )
    
    async def _infer_domain(
        self,
        content: str,
        context: dict[str, Any],
    ) -> ResolutionResult | None:
        """
        Infer domain from content using classifier.
        
        Returns None if no domain exceeds threshold.
        """
        try:
            candidates = await self._classifier.classify(content, context)
        except Exception as e:
            logger.warning(f"Domain classification failed: {e}")
            return None
        
        if not candidates:
            return None
        
        # Filter to valid domains and threshold
        valid_candidates = [
            (name, score)
            for name, score in candidates
            if self._registry.exists(name) and score >= self._threshold
        ]
        
        if not valid_candidates:
            return None
        
        # Take highest scoring
        best_name, best_score = valid_candidates[0]
        profile = self._registry.get(best_name)
        
        return ResolutionResult(
            profile=profile,
            method=ResolutionMethod.INFERRED,
            confidence=best_score,
            candidates=candidates,
        )
    
    def _get_domain_from_context(self, context: dict[str, Any]) -> str | None:
        """Extract domain from context metadata."""
        # Check various context locations
        locations = [
            ("domain",),  # Direct domain field
            ("session", "domain"),  # Session metadata
            ("user", "default_domain"),  # User preference
            ("metadata", "domain"),  # Generic metadata
        ]
        
        for path in locations:
            value = context
            for key in path:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    value = None
                    break
            if value and isinstance(value, str):
                return value
        
        return None
    
    def add_keyword_rule(
        self,
        domain: str,
        keywords: list[str],
        patterns: list[str] | None = None,
        weight: float = 1.0,
    ) -> None:
        """
        Add a keyword classification rule.
        
        Convenience method when using KeywordClassifier.
        """
        if isinstance(self._classifier, KeywordClassifier):
            self._classifier.add_rule(KeywordRule(
                domain=domain,
                keywords=keywords,
                patterns=patterns or [],
                weight=weight,
            ))
    
    @property
    def registry(self) -> DomainRegistry:
        """Access the underlying registry."""
        return self._registry


# ============================================================
# Factory functions
# ============================================================

def create_resolver_with_keywords(
    registry: DomainRegistry,
    rules: dict[str, list[str]],
    threshold: float = 0.6,
) -> DomainResolver:
    """
    Create resolver with keyword rules.
    
    Args:
        registry: Domain registry
        rules: Dict of domain_name -> keywords list
        threshold: Minimum confidence threshold
        
    Returns:
        Configured DomainResolver
    """
    classifier = KeywordClassifier()
    for domain, keywords in rules.items():
        classifier.add_rule(KeywordRule(domain=domain, keywords=keywords))
    
    return DomainResolver(
        registry=registry,
        classifier=classifier,
        inference_threshold=threshold,
    )


def create_default_resolver(registry: DomainRegistry) -> DomainResolver:
    """
    Create resolver with sensible default rules.
    
    Includes common patterns for standard domains.
    """
    classifier = KeywordClassifier()
    
    # Technical support patterns
    classifier.add_rule(KeywordRule(
        domain="technical_support",
        keywords=[
            "error", "bug", "crash", "not working", "broken",
            "help", "issue", "problem", "fix", "support",
            "install", "update", "upgrade", "password", "reset",
        ],
        patterns=[
            r"\berror\s+\d+\b",  # "error 404"
            r"\bfailed?\b",
            r"\bcrash(ed|ing)?\b",
        ],
    ))
    
    # Financial analysis patterns
    classifier.add_rule(KeywordRule(
        domain="financial_analysis",
        keywords=[
            "stock", "portfolio", "investment", "dividend", "earnings",
            "revenue", "profit", "loss", "market", "trading",
            "balance", "account", "transaction", "payment",
            "budget", "forecast", "valuation", "roi",
        ],
        patterns=[
            r"\$[\d,]+(?:\.\d{2})?",  # Dollar amounts
            r"\b\d+(?:\.\d+)?%\b",  # Percentages
            r"\b[A-Z]{1,5}\b",  # Stock tickers (rough)
        ],
    ))
    
    return DomainResolver(
        registry=registry,
        classifier=classifier,
        inference_threshold=0.6,
    )
