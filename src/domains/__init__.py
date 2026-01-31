"""
Domain-Aware Agent Configuration System

Enables dynamic agent behavior based on task domain WITHOUT code changes.

A Domain defines:
- Prompt templates (system prompt, persona)
- RAG configuration (index, filters, retrieval rules)
- Memory policies (scope, retention, retrieval)
- Tool availability (allowed/denied lists)
- Reasoning strategy (ReAct, ToolCalling, etc.)
- Safety constraints (guardrails, output limits)

Domain selection supports:
1. Explicit: API parameter specifies domain
2. Inferred: Lightweight classifier detects domain
3. Fallback: Safe default when uncertain

Usage:
    from src.domains import DomainRegistry, DomainProfile, DomainAwareRuntime

    registry = DomainRegistry.from_directory("config/domains")
    runtime = DomainAwareRuntime(registry=registry, llm=llm_adapter)

    # Explicit domain
    result = await runtime.run(message, context, domain="financial_analysis")

    # Auto-inferred domain
    result = await runtime.run(message, context)
"""

from src.domains.profile import (
    DomainProfile,
    MemoryConfig,
    MemoryScopeType,
    PromptConfig,
    RAGConfig,
    ReasoningConfig,
    ReasoningStrategyType,
    SafetyConfig,
    ToolsConfig,
)
from src.domains.registry import (
    DEFAULT_DOMAIN,
    DomainNotFoundError,
    DomainRegistry,
    DomainRegistryError,
    DomainValidationError,
)
from src.domains.resolver import (
    DomainResolver,
    KeywordClassifier,
    KeywordRule,
    ResolutionMethod,
    ResolutionResult,
    create_default_resolver,
    create_resolver_with_keywords,
)
from src.domains.runtime import (
    DomainAwareRuntime,
    DomainAwareToolExecutor,
    DomainEventType,
    DomainExecutionContext,
    create_domain_aware_runtime,
)

__all__ = [
    "DEFAULT_DOMAIN",
    # Runtime
    "DomainAwareRuntime",
    "DomainAwareToolExecutor",
    "DomainEventType",
    "DomainExecutionContext",
    "DomainNotFoundError",
    # Core profile types
    "DomainProfile",
    # Registry
    "DomainRegistry",
    "DomainRegistryError",
    # Resolution
    "DomainResolver",
    "DomainValidationError",
    "KeywordClassifier",
    "KeywordRule",
    "MemoryConfig",
    "MemoryScopeType",
    "PromptConfig",
    "RAGConfig",
    "ReasoningConfig",
    "ReasoningStrategyType",
    "ResolutionMethod",
    "ResolutionResult",
    "SafetyConfig",
    "ToolsConfig",
    "create_default_resolver",
    "create_domain_aware_runtime",
    "create_resolver_with_keywords",
]
