"""
Domain API Routes

Endpoints for domain discovery and information.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_domain_registry
from src.domains import DomainRegistry

router = APIRouter()


class DomainSummary(BaseModel):
    """Summary of a domain profile."""

    name: str
    version: str
    display_name: str | None
    description: str
    tags: list[str]

    # Feature flags
    rag_enabled: bool
    memory_enabled: bool
    tools_enabled: bool

    # Tool counts
    allowed_tools_count: int | None
    denied_tools_count: int


class DomainDetail(BaseModel):
    """Detailed domain profile information."""

    name: str
    version: str
    display_name: str | None
    description: str
    tags: list[str]
    extends: str | None

    # Configuration summaries
    prompt_preview: str
    reasoning_strategy: str
    memory_scope: str

    # RAG config
    rag_enabled: bool
    rag_collection: str | None
    rag_top_k: int

    # Tools config
    tools_enabled: bool
    allowed_tools: list[str] | None
    denied_tools: list[str]

    # Safety config
    input_guardrails: bool
    output_guardrails: bool
    require_citations: bool
    has_disclaimer: bool


class DomainListResponse(BaseModel):
    """Response for domain list endpoint."""

    domains: list[DomainSummary]
    total: int
    default: str


@router.get("/domains")
async def list_domains(
    registry: DomainRegistry = Depends(get_domain_registry),
) -> DomainListResponse:
    """
    List all available domains.

    Returns summaries of all registered domain profiles.
    """
    summaries = []

    for profile in registry:
        summaries.append(
            DomainSummary(
                name=profile.name,
                version=profile.version,
                display_name=profile.display_name,
                description=profile.description[:200] if profile.description else "",
                tags=profile.tags,
                rag_enabled=profile.rag.enabled,
                memory_enabled=profile.memory.enabled,
                tools_enabled=profile.tools.enabled,
                allowed_tools_count=(
                    len(profile.tools.allowed_tools) if profile.tools.allowed_tools else None
                ),
                denied_tools_count=len(profile.tools.denied_tools),
            )
        )

    return DomainListResponse(
        domains=summaries,
        total=len(summaries),
        default=registry.fallback.name,
    )


@router.get("/domains/{name}")
async def get_domain(
    name: str,
    registry: DomainRegistry = Depends(get_domain_registry),
) -> DomainDetail:
    """
    Get detailed information about a specific domain.
    """
    if not registry.exists(name):
        raise HTTPException(status_code=404, detail=f"Domain not found: {name}")

    profile = registry.get(name)

    return DomainDetail(
        name=profile.name,
        version=profile.version,
        display_name=profile.display_name,
        description=profile.description,
        tags=profile.tags,
        extends=profile.extends,
        prompt_preview=(
            profile.prompt.system_prompt[:300] + "..."
            if len(profile.prompt.system_prompt) > 300
            else profile.prompt.system_prompt
        ),
        reasoning_strategy=profile.reasoning.strategy.value,
        memory_scope=profile.memory.scope.value,
        rag_enabled=profile.rag.enabled,
        rag_collection=profile.rag.collection,
        rag_top_k=profile.rag.top_k,
        tools_enabled=profile.tools.enabled,
        allowed_tools=profile.tools.allowed_tools,
        denied_tools=profile.tools.denied_tools,
        input_guardrails=profile.safety.input_guardrails_enabled,
        output_guardrails=profile.safety.output_guardrails_enabled,
        require_citations=profile.safety.require_citations,
        has_disclaimer=profile.safety.disclaimer is not None,
    )


@router.get("/domains/{name}/tools")
async def get_domain_tools(
    name: str,
    registry: DomainRegistry = Depends(get_domain_registry),
) -> dict[str, Any]:
    """
    Get tool configuration for a domain.
    """
    if not registry.exists(name):
        raise HTTPException(status_code=404, detail=f"Domain not found: {name}")

    profile = registry.get(name)

    return {
        "domain": name,
        "tools_enabled": profile.tools.enabled,
        "allowed_tools": profile.tools.allowed_tools,
        "denied_tools": profile.tools.denied_tools,
        "allowed_categories": profile.tools.allowed_categories,
        "denied_categories": profile.tools.denied_categories,
        "max_tool_calls": profile.tools.max_tool_calls,
        "require_confirmation": profile.tools.require_confirmation,
    }


@router.get("/domains/{name}/safety")
async def get_domain_safety(
    name: str,
    registry: DomainRegistry = Depends(get_domain_registry),
) -> dict[str, Any]:
    """
    Get safety configuration for a domain.
    """
    if not registry.exists(name):
        raise HTTPException(status_code=404, detail=f"Domain not found: {name}")

    profile = registry.get(name)

    return {
        "domain": name,
        "input_guardrails_enabled": profile.safety.input_guardrails_enabled,
        "output_guardrails_enabled": profile.safety.output_guardrails_enabled,
        "blocked_topics": profile.safety.blocked_topics,
        "blocked_patterns_count": len(profile.safety.blocked_patterns),
        "max_input_length": profile.safety.max_input_length,
        "max_output_length": profile.safety.max_output_length,
        "redact_pii_input": profile.safety.redact_pii_input,
        "redact_pii_output": profile.safety.redact_pii_output,
        "require_citations": profile.safety.require_citations,
        "has_disclaimer": profile.safety.disclaimer is not None,
        "log_all_interactions": profile.safety.log_all_interactions,
    }
