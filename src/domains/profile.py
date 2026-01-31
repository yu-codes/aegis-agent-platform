"""
Domain Profile Schema

Defines the complete configuration surface for a domain.

Design principles:
- Declarative: All configuration is data, not code
- Composable: Profiles can extend base profiles
- Versionable: Each profile has explicit version
- Validatable: Pydantic ensures correctness at load time
- Auditable: All fields are traceable

A DomainProfile is READ-ONLY at runtime. It is resolved once
before execution begins and never modified during execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ReasoningStrategyType(str, Enum):
    """Available reasoning strategies."""
    
    REACT = "react"
    TOOL_CALLING = "tool_calling"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REFLECTION = "reflection"
    AUTO = "auto"  # Let runtime decide based on model


class MemoryScopeType(str, Enum):
    """Scope of memory retrieval."""
    
    SESSION = "session"  # Current session only
    USER = "user"  # All sessions for current user
    DOMAIN = "domain"  # All sessions in this domain
    GLOBAL = "global"  # Everything (use with caution)


class PromptConfig(BaseModel):
    """
    Prompt configuration for a domain.
    
    Defines system prompt, persona, and optional prompt templates.
    """
    
    # Core prompt
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="Main system prompt defining agent behavior",
    )
    
    # Optional persona for more personality
    persona: str | None = Field(
        default=None,
        description="Character/persona to adopt (appended to system prompt)",
    )
    
    # Template references (for prompt registry integration)
    template_id: str | None = Field(
        default=None,
        description="ID of prompt template in PromptRegistry",
    )
    
    # Context injection templates
    rag_context_template: str = Field(
        default="Relevant context:\n{context}\n\nUser question: {query}",
        description="Template for injecting RAG context",
    )
    
    memory_context_template: str = Field(
        default="Previous relevant information:\n{memory}",
        description="Template for injecting memory context",
    )
    
    # Response guidelines
    response_format: Literal["text", "markdown", "json", "structured"] = "text"
    response_language: str = Field(
        default="en",
        description="Preferred response language (ISO 639-1)",
    )
    
    class Config:
        frozen = True


class RAGConfig(BaseModel):
    """
    RAG (Retrieval-Augmented Generation) configuration.
    
    Defines how documents are retrieved and assembled for context.
    """
    
    # Enable/disable RAG
    enabled: bool = True
    
    # Index selection
    index_name: str | None = Field(
        default=None,
        description="Specific vector index to use (None = default)",
    )
    collection: str | None = Field(
        default=None,
        description="Collection/namespace within index",
    )
    
    # Retrieval parameters
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Filtering
    metadata_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Static metadata filters to apply",
    )
    
    # Dynamic filter fields (extracted from context)
    filter_from_context: list[str] = Field(
        default_factory=list,
        description="Context fields to use as filters (e.g., ['user_id', 'department'])",
    )
    
    # Reranking
    rerank: bool = False
    rerank_model: str | None = None
    
    # Context budget
    max_context_tokens: int = Field(default=4000, ge=100)
    max_documents: int = Field(default=10, ge=1)
    
    # Source requirements
    require_sources: bool = Field(
        default=False,
        description="Require RAG context (fail if no results)",
    )
    
    class Config:
        frozen = True


class MemoryConfig(BaseModel):
    """
    Memory configuration for a domain.
    
    Defines how memory is stored, scoped, and retrieved.
    """
    
    # Enable/disable memory
    enabled: bool = True
    
    # Memory scope
    scope: MemoryScopeType = MemoryScopeType.SESSION
    
    # Short-term memory (conversation history)
    short_term_enabled: bool = True
    max_turns: int = Field(default=20, ge=1, le=100)
    summarize_after: int | None = Field(
        default=None,
        description="Summarize history after N turns",
    )
    
    # Long-term memory
    long_term_enabled: bool = False
    long_term_retrieval_k: int = Field(default=5, ge=1)
    
    # What to remember
    store_user_messages: bool = True
    store_assistant_messages: bool = True
    store_tool_results: bool = True
    
    # Retention
    retention_days: int | None = Field(
        default=None,
        description="Auto-expire after N days (None = forever)",
    )
    
    class Config:
        frozen = True


class ToolsConfig(BaseModel):
    """
    Tool availability configuration.
    
    Controls which tools are available in this domain.
    Uses allowlist/denylist pattern for flexibility.
    """
    
    # Enable/disable tools entirely
    enabled: bool = True
    
    # Tool selection (allowlist takes precedence if both specified)
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Explicit list of allowed tool names (None = all)",
    )
    denied_tools: list[str] = Field(
        default_factory=list,
        description="Tools to deny (applied after allowlist)",
    )
    
    # Categories (alternative to individual tools)
    allowed_categories: list[str] | None = Field(
        default=None,
        description="Allowed tool categories",
    )
    denied_categories: list[str] = Field(
        default_factory=list,
        description="Denied tool categories",
    )
    
    # Tool-specific overrides
    tool_configs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-tool configuration overrides",
    )
    
    # Execution limits
    max_tool_calls: int = Field(default=20, ge=0)
    require_confirmation: list[str] = Field(
        default_factory=list,
        description="Tools requiring user confirmation",
    )
    
    class Config:
        frozen = True


class ReasoningConfig(BaseModel):
    """
    Reasoning strategy configuration.
    
    Defines how the agent reasons about tasks.
    """
    
    # Strategy selection
    strategy: ReasoningStrategyType = ReasoningStrategyType.AUTO
    
    # Iteration limits
    max_iterations: int = Field(default=10, ge=1, le=50)
    
    # Model selection (override global default)
    model: str | None = Field(
        default=None,
        description="LLM model to use (None = use default)",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature override",
    )
    
    # Token budget
    max_tokens_per_call: int = Field(default=4096, ge=100)
    max_total_tokens: int = Field(default=100000, ge=1000)
    
    # Thinking/reasoning visibility
    show_thinking: bool = Field(
        default=False,
        description="Include reasoning steps in output",
    )
    
    # Planning integration
    enable_planning: bool = Field(
        default=False,
        description="Use planning module for complex tasks",
    )
    planning_threshold: int = Field(
        default=3,
        description="Steps before triggering planning mode",
    )
    
    class Config:
        frozen = True


class SafetyConfig(BaseModel):
    """
    Safety and guardrail configuration.
    
    Defines constraints on inputs and outputs.
    """
    
    # Input guardrails
    input_guardrails_enabled: bool = True
    blocked_topics: list[str] = Field(
        default_factory=list,
        description="Topics to refuse (e.g., ['violence', 'illegal_activities'])",
    )
    blocked_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns to block in input",
    )
    
    # Output guardrails
    output_guardrails_enabled: bool = True
    output_blocked_patterns: list[str] = Field(
        default_factory=list,
        description="Patterns to never include in output",
    )
    
    # Content limits
    max_input_length: int = Field(default=32000, ge=100)
    max_output_length: int = Field(default=16000, ge=100)
    
    # PII handling
    redact_pii_input: bool = False
    redact_pii_output: bool = False
    pii_categories: list[str] = Field(
        default_factory=lambda: ["email", "phone", "ssn", "credit_card"],
    )
    
    # Compliance
    require_citations: bool = Field(
        default=False,
        description="Require source citations in responses",
    )
    disclaimer: str | None = Field(
        default=None,
        description="Disclaimer to append to responses",
    )
    
    # Audit
    log_all_interactions: bool = True
    flag_for_review: list[str] = Field(
        default_factory=list,
        description="Keywords triggering human review flag",
    )
    
    class Config:
        frozen = True


class DomainProfile(BaseModel):
    """
    Complete domain configuration profile.
    
    This is the FIRST-CLASS ABSTRACTION for domain-aware behavior.
    A DomainProfile completely describes how the agent should behave
    for a particular task domain.
    
    Key properties:
    - Declarative: All configuration is data
    - Versionable: Has version field
    - Immutable: Frozen at runtime
    - Auditable: Includes metadata for tracking
    
    Example:
        profile = DomainProfile(
            name="technical_support",
            version="1.2.0",
            description="IT support agent",
            prompt=PromptConfig(
                system_prompt="You are an IT support specialist..."
            ),
            tools=ToolsConfig(
                allowed_tools=["search_docs", "create_ticket"]
            ),
        )
    """
    
    # Identity
    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique domain identifier (lowercase, underscores)",
    )
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Semantic version",
    )
    
    # Metadata
    display_name: str | None = Field(
        default=None,
        description="Human-readable name",
    )
    description: str = Field(
        default="",
        description="What this domain is for",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )
    
    # Inheritance
    extends: str | None = Field(
        default=None,
        description="Base profile to extend",
    )
    
    # Configuration sections
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    
    # Custom domain-specific data
    custom: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific custom configuration",
    )
    
    # Audit fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str | None = None
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is valid identifier."""
        reserved = {"default", "none", "all", "system", "internal"}
        if v.lower() in reserved:
            raise ValueError(f"'{v}' is a reserved domain name")
        return v
    
    def effective_system_prompt(self) -> str:
        """Build complete system prompt with persona if defined."""
        prompt = self.prompt.system_prompt
        if self.prompt.persona:
            prompt = f"{prompt}\n\nPersona: {self.prompt.persona}"
        return prompt
    
    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in this domain."""
        if not self.tools.enabled:
            return False
        
        # Explicit deny takes precedence
        if tool_name in self.tools.denied_tools:
            return False
        
        # If allowlist exists, tool must be in it
        if self.tools.allowed_tools is not None:
            return tool_name in self.tools.allowed_tools
        
        # Not in denylist and no allowlist = allowed
        return True
    
    def get_effective_model(self, default: str) -> str:
        """Get model to use, with fallback to default."""
        return self.reasoning.model or default
    
    def get_effective_temperature(self, default: float) -> float:
        """Get temperature to use, with fallback."""
        return self.reasoning.temperature if self.reasoning.temperature is not None else default
    
    class Config:
        frozen = True
        
    def __hash__(self) -> int:
        """Allow DomainProfile to be used in sets/dicts."""
        return hash((self.name, self.version))
    
    def __repr__(self) -> str:
        return f"DomainProfile(name={self.name!r}, version={self.version!r})"


# ============================================================
# Factory functions for common patterns
# ============================================================

def create_minimal_profile(name: str, system_prompt: str) -> DomainProfile:
    """Create a minimal domain profile with just name and prompt."""
    return DomainProfile(
        name=name,
        prompt=PromptConfig(system_prompt=system_prompt),
    )


def create_readonly_profile(name: str, system_prompt: str) -> DomainProfile:
    """Create a profile with no tools (read-only agent)."""
    return DomainProfile(
        name=name,
        prompt=PromptConfig(system_prompt=system_prompt),
        tools=ToolsConfig(enabled=False),
    )


def create_safe_profile(name: str, system_prompt: str) -> DomainProfile:
    """Create a profile with strict safety settings."""
    return DomainProfile(
        name=name,
        prompt=PromptConfig(system_prompt=system_prompt),
        safety=SafetyConfig(
            input_guardrails_enabled=True,
            output_guardrails_enabled=True,
            redact_pii_input=True,
            redact_pii_output=True,
            log_all_interactions=True,
        ),
    )
