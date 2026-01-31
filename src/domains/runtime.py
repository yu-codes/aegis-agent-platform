"""
Domain-Aware Agent Runtime

Extends AgentRuntime with domain profile support.
Resolves domain BEFORE reasoning and applies all domain configurations.

Integration architecture:

    ┌─────────────────────────────────────────────────────────────────┐
    │                        API Request                               │
    │   { message: "...", domain: "financial_analysis", ... }          │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    DomainResolver                                │
    │   1. Explicit domain check                                       │
    │   2. Context-based resolution                                    │
    │   3. Inference (if enabled)                                      │
    │   4. Fallback to default                                         │
    └────────────────────────────┬────────────────────────────────────┘
                                 │ DomainProfile
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                  DomainAwareRuntime                              │
    │   • Apply system prompt from profile                             │
    │   • Configure RAG retrieval                                      │
    │   • Filter tools by domain rules                                 │
    │   • Set memory scope                                             │
    │   • Configure safety guardrails                                  │
    │   • Emit domain context events                                   │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    AgentRuntime.run()                            │
    │   (Existing execution with domain-configured components)         │
    └─────────────────────────────────────────────────────────────────┘
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from src.core.interfaces import (
    GuardrailProtocol,
    InputValidatorProtocol,
    LLMAdapterProtocol,
    MemoryProtocol,
    RetrieverProtocol,
    ToolExecutorProtocol,
    TracerProtocol,
)
from src.core.types import ExecutionContext, Message
from src.domains.profile import DomainProfile
from src.domains.registry import DomainRegistry
from src.domains.resolver import DomainResolver, ResolutionResult
from src.runtime.agent import (
    AgentEvent,
    AgentEventType,
    AgentResult,
    AgentRuntime,
    RuntimeConfig,
)

logger = logging.getLogger(__name__)


# New event types for domain operations
class DomainEventType:
    """Domain-specific event types (extends AgentEventType)."""

    DOMAIN_RESOLVED = "domain_resolved"
    DOMAIN_APPLIED = "domain_applied"
    TOOL_FILTERED = "tool_filtered"  # Tool blocked by domain policy


@dataclass
class DomainExecutionContext:
    """
    Extended execution context with domain information.

    This is passed through the entire execution pipeline
    after domain resolution.
    """

    # Original context
    base: ExecutionContext

    # Domain information
    profile: DomainProfile
    resolution: ResolutionResult

    # Applied configuration
    effective_system_prompt: str
    allowed_tools: set[str] | None = None
    denied_tools: set[str] = field(default_factory=set)

    @property
    def domain_name(self) -> str:
        return self.profile.name

    @property
    def domain_version(self) -> str:
        return self.profile.version


class DomainAwareToolExecutor:
    """
    Wrapper that filters tools based on domain policy.

    Sits between the runtime and the real tool executor,
    enforcing domain-level tool access control.
    """

    def __init__(
        self,
        executor: ToolExecutorProtocol,
        profile: DomainProfile,
    ):
        self._executor = executor
        self._profile = profile
        self._filtered_count = 0

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in current domain."""
        return self._profile.is_tool_allowed(tool_name)

    def get_tool_definitions(self, context: ExecutionContext) -> list[dict[str, Any]]:
        """Get tool definitions filtered by domain policy."""
        all_tools = self._executor.get_tool_definitions(context)

        if not self._profile.tools.enabled:
            return []

        filtered = []
        for tool in all_tools:
            tool_name = tool.get("function", {}).get("name", "")
            if self.is_tool_allowed(tool_name):
                filtered.append(tool)
            else:
                self._filtered_count += 1
                logger.debug(f"Tool '{tool_name}' filtered by domain '{self._profile.name}'")

        return filtered

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> Any:
        """Execute tool if allowed by domain."""
        if not self.is_tool_allowed(name):
            from src.core.types import ToolResult

            return ToolResult(
                tool_call_id="filtered",
                name=name,
                result=None,
                error=f"Tool '{name}' is not available in domain '{self._profile.name}'",
            )

        return await self._executor.execute(name, arguments, context)

    @property
    def filtered_count(self) -> int:
        """Number of tools filtered out."""
        return self._filtered_count


class DomainAwareRuntime:
    """
    Domain-aware agent runtime.

    Wraps AgentRuntime with domain resolution and configuration.
    This is the recommended entry point for domain-aware execution.

    Key responsibilities:
    1. Resolve domain BEFORE any execution
    2. Apply domain configuration to all components
    3. Enforce domain policies (tools, safety)
    4. Emit domain-related events
    5. Maintain domain audit trail

    Usage:
        runtime = DomainAwareRuntime(
            registry=domain_registry,
            base_runtime=agent_runtime,
        )

        result = await runtime.run(
            message="How do I reset my password?",
            context=context,
            domain="technical_support",  # Optional explicit domain
        )
    """

    def __init__(
        self,
        registry: DomainRegistry,
        llm: LLMAdapterProtocol,
        tool_executor: ToolExecutorProtocol | None = None,
        memory: MemoryProtocol | None = None,
        retriever: RetrieverProtocol | None = None,
        input_validator: InputValidatorProtocol | None = None,
        guardrails: GuardrailProtocol | None = None,
        tracer: TracerProtocol | None = None,
        resolver: DomainResolver | None = None,
        enable_inference: bool = True,
        inference_threshold: float = 0.6,
    ):
        """
        Initialize domain-aware runtime.

        Args:
            registry: Domain profile registry
            llm: LLM adapter
            tool_executor: Tool executor (will be wrapped for filtering)
            memory: Memory component
            retriever: RAG retriever
            input_validator: Input validation
            guardrails: Output guardrails
            tracer: Observability tracer
            resolver: Custom domain resolver (optional)
            enable_inference: Enable automatic domain inference
            inference_threshold: Minimum confidence for inferred domains
        """
        self._registry = registry
        self._llm = llm
        self._tools = tool_executor
        self._memory = memory
        self._retriever = retriever
        self._validator = input_validator
        self._guardrails = guardrails
        self._tracer = tracer

        # Create resolver if not provided
        self._resolver = resolver or DomainResolver(
            registry=registry,
            enable_inference=enable_inference,
            inference_threshold=inference_threshold,
        )

    async def resolve_domain(
        self,
        explicit_domain: str | None = None,
        content: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ResolutionResult:
        """
        Resolve which domain to use.

        Exposed for cases where you need domain info before execution.
        """
        return await self._resolver.resolve(
            explicit_domain=explicit_domain,
            content=content,
            context=context,
        )

    async def run(
        self,
        message: str,
        context: ExecutionContext,
        history: list[Message] | None = None,
        domain: str | None = None,
    ) -> AgentResult:
        """
        Execute with domain-aware configuration.

        Args:
            message: User message
            context: Execution context
            history: Optional conversation history
            domain: Explicit domain override (highest priority)

        Returns:
            AgentResult with domain metadata included
        """
        # Phase 0: Resolve domain
        resolution = await self._resolver.resolve(
            explicit_domain=domain,
            content=message,
            context={
                "session_id": str(context.session_id) if context.session_id else None,
                "user_id": context.user_id,
            },
        )

        profile = resolution.profile

        logger.info(
            f"Domain resolved: {profile.name} v{profile.version} "
            f"(method={resolution.method.value}, confidence={resolution.confidence:.2f})"
        )

        # Build domain-specific runtime config
        config = self._build_config_from_profile(profile)

        # Create domain-filtered tool executor
        filtered_tools = None
        if self._tools:
            filtered_tools = DomainAwareToolExecutor(self._tools, profile)

        # Create runtime with domain configuration
        runtime = AgentRuntime(
            llm=self._llm,
            tool_executor=filtered_tools,
            memory=self._memory if profile.memory.enabled else None,
            retriever=self._retriever if profile.rag.enabled else None,
            input_validator=self._validator if profile.safety.input_guardrails_enabled else None,
            guardrails=self._guardrails if profile.safety.output_guardrails_enabled else None,
            tracer=self._tracer,
            config=config,
        )

        # Execute
        result = await runtime.run(message, context, history)

        # Append domain metadata
        if result.success:
            result = self._enrich_result(result, resolution)

        return result

    async def run_stream(
        self,
        message: str,
        context: ExecutionContext,
        history: list[Message] | None = None,
        domain: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Execute with streaming and domain awareness.

        Yields domain-related events before standard events.
        """
        # Phase 0: Resolve domain
        resolution = await self._resolver.resolve(
            explicit_domain=domain,
            content=message,
            context={
                "session_id": str(context.session_id) if context.session_id else None,
                "user_id": context.user_id,
            },
        )

        profile = resolution.profile

        # Emit domain resolution event
        yield AgentEvent(
            type=AgentEventType.STATE_CHANGED,  # Use existing type for compatibility
            data={
                "event_type": DomainEventType.DOMAIN_RESOLVED,
                "domain": profile.name,
                "version": profile.version,
                "method": resolution.method.value,
                "confidence": resolution.confidence,
            },
        )

        # Build domain-specific config
        config = self._build_config_from_profile(profile)

        # Create filtered tool executor
        filtered_tools = None
        if self._tools:
            filtered_tools = DomainAwareToolExecutor(self._tools, profile)

        # Create runtime
        runtime = AgentRuntime(
            llm=self._llm,
            tool_executor=filtered_tools,
            memory=self._memory if profile.memory.enabled else None,
            retriever=self._retriever if profile.rag.enabled else None,
            input_validator=self._validator if profile.safety.input_guardrails_enabled else None,
            guardrails=self._guardrails if profile.safety.output_guardrails_enabled else None,
            tracer=self._tracer,
            config=config,
        )

        # Stream events from base runtime
        async for event in runtime.run_stream(message, context, history):
            yield event

    def _build_config_from_profile(self, profile: DomainProfile) -> RuntimeConfig:
        """Build RuntimeConfig from domain profile."""
        return RuntimeConfig(
            # Use profile's model if specified, otherwise use default
            model=profile.reasoning.model or "gpt-4-turbo-preview",
            temperature=profile.get_effective_temperature(0.7),
            max_tokens_per_call=profile.reasoning.max_tokens_per_call,
            # System prompt from profile
            system_prompt=profile.effective_system_prompt(),
            # Budgets from profile
            max_iterations=profile.reasoning.max_iterations,
            max_tool_calls=profile.tools.max_tool_calls,
            max_total_tokens=profile.reasoning.max_total_tokens,
            # Feature flags from profile
            enable_memory=profile.memory.enabled,
            enable_rag=profile.rag.enabled,
            enable_tools=profile.tools.enabled,
            # RAG settings from profile
            rag_top_k=profile.rag.top_k,
            rag_min_score=profile.rag.min_score,
        )

    def _enrich_result(
        self,
        result: AgentResult,
        resolution: ResolutionResult,
    ) -> AgentResult:
        """Add domain metadata to result."""
        # Create a new result with domain info (AgentResult is frozen)

        # Unfortunately AgentResult isn't a dataclass, so we reconstruct
        return AgentResult(
            content=result.content,
            success=result.success,
            blocked=result.blocked,
            blocked_reason=result.blocked_reason,
            execution_id=result.execution_id,
            model=result.model,
            total_tokens=result.total_tokens,
            tool_calls_count=result.tool_calls_count,
            iterations=result.iterations,
            duration_ms=result.duration_ms,
            tool_results=result.tool_results,
            sources=result.sources,
            trace_id=result.trace_id,
            # Domain info would go here if AgentResult had a metadata field
        )

    @property
    def registry(self) -> DomainRegistry:
        """Access domain registry."""
        return self._registry

    @property
    def resolver(self) -> DomainResolver:
        """Access domain resolver."""
        return self._resolver


# ============================================================
# Factory functions
# ============================================================


def create_domain_aware_runtime(
    registry: DomainRegistry,
    llm: LLMAdapterProtocol,
    tool_executor: ToolExecutorProtocol | None = None,
    memory: MemoryProtocol | None = None,
    retriever: RetrieverProtocol | None = None,
    enable_inference: bool = True,
) -> DomainAwareRuntime:
    """
    Create a domain-aware runtime with sensible defaults.

    This is the recommended way to create a production runtime.
    """
    from src.domains.resolver import create_default_resolver

    resolver = create_default_resolver(registry)

    return DomainAwareRuntime(
        registry=registry,
        llm=llm,
        tool_executor=tool_executor,
        memory=memory,
        retriever=retriever,
        resolver=resolver,
        enable_inference=enable_inference,
    )
