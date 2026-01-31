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
    PromptConfig,
    RAGConfig,
    MemoryConfig,
    ToolsConfig,
    ReasoningConfig,
    SafetyConfig,
    ReasoningStrategyType,
    MemoryScopeType,
)
from src.domains.registry import (
    DomainRegistry,
    DomainRegistryError,
    DomainNotFoundError,
    DomainValidationError,
    DEFAULT_DOMAIN,
)
from src.domains.resolver import (
    DomainResolver,
    ResolutionResult,
    ResolutionMethod,
    KeywordClassifier,
    KeywordRule,
    create_resolver_with_keywords,
    create_default_resolver,
)
from src.domains.runtime import (
    DomainAwareRuntime,
    DomainAwareToolExecutor,
    DomainExecutionContext,
    DomainEventType,
    create_domain_aware_runtime,
)

__all__ = [
    # Core profile types
    "DomainProfile",
    "PromptConfig",
    "RAGConfig",
    "MemoryConfig",
    "ToolsConfig",
    "ReasoningConfig",
    "SafetyConfig",
    "ReasoningStrategyType",
    "MemoryScopeType",
    # Registry
    "DomainRegistry",
    "DomainRegistryError",
    "DomainNotFoundError",
    "DomainValidationError",
    "DEFAULT_DOMAIN",
    # Resolution
    "DomainResolver",
    "ResolutionResult",
    "ResolutionMethod",
    "KeywordClassifier",
    "KeywordRule",
    "create_resolver_with_keywords",
    "create_default_resolver",
    # Runtime
    "DomainAwareRuntime",
    "DomainAwareToolExecutor",
    "DomainExecutionContext",
    "DomainEventType",
    "create_domain_aware_runtime",
]
