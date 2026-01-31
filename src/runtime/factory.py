"""
Runtime Factory

Factory and builder for creating AgentRuntime instances.
Provides a clean API for assembling runtimes with various configurations.
"""

from src.core.interfaces import (
    GuardrailProtocol,
    InputValidatorProtocol,
    LLMAdapterProtocol,
    MemoryProtocol,
    RetrieverProtocol,
    ToolExecutorProtocol,
    TracerProtocol,
)
from src.runtime.agent import AgentRuntime, RuntimeConfig


def create_runtime(
    llm: LLMAdapterProtocol,
    tool_executor: ToolExecutorProtocol | None = None,
    memory: MemoryProtocol | None = None,
    retriever: RetrieverProtocol | None = None,
    input_validator: InputValidatorProtocol | None = None,
    guardrails: GuardrailProtocol | None = None,
    tracer: TracerProtocol | None = None,
    config: RuntimeConfig | None = None,
) -> AgentRuntime:
    """
    Create an AgentRuntime with the specified components.

    This is the simplest way to create a runtime. For more control,
    use RuntimeBuilder.

    Args:
        llm: Required LLM adapter
        tool_executor: Optional tool executor
        memory: Optional memory manager
        retriever: Optional RAG retriever
        input_validator: Optional input validator
        guardrails: Optional output guardrails
        tracer: Optional tracer
        config: Optional runtime configuration

    Returns:
        Configured AgentRuntime
    """
    return AgentRuntime(
        llm=llm,
        tool_executor=tool_executor,
        memory=memory,
        retriever=retriever,
        input_validator=input_validator,
        guardrails=guardrails,
        tracer=tracer,
        config=config,
    )


class RuntimeBuilder:
    """
    Builder pattern for AgentRuntime.

    Provides a fluent API for constructing runtimes:

        runtime = (
            RuntimeBuilder()
            .with_llm(openai_adapter)
            .with_tools(tool_executor)
            .with_memory(memory_manager)
            .with_config(RuntimeConfig(max_iterations=20))
            .build()
        )
    """

    def __init__(self):
        self._llm: LLMAdapterProtocol | None = None
        self._tool_executor: ToolExecutorProtocol | None = None
        self._memory: MemoryProtocol | None = None
        self._retriever: RetrieverProtocol | None = None
        self._input_validator: InputValidatorProtocol | None = None
        self._guardrails: GuardrailProtocol | None = None
        self._tracer: TracerProtocol | None = None
        self._config = RuntimeConfig()

    def with_llm(self, llm: LLMAdapterProtocol) -> "RuntimeBuilder":
        """Set the LLM adapter (required)."""
        self._llm = llm
        return self

    def with_tools(self, executor: ToolExecutorProtocol) -> "RuntimeBuilder":
        """Set the tool executor."""
        self._tool_executor = executor
        return self

    def with_memory(self, memory: MemoryProtocol) -> "RuntimeBuilder":
        """Set the memory manager."""
        self._memory = memory
        return self

    def with_retriever(self, retriever: RetrieverProtocol) -> "RuntimeBuilder":
        """Set the RAG retriever."""
        self._retriever = retriever
        return self

    def with_input_validator(self, validator: InputValidatorProtocol) -> "RuntimeBuilder":
        """Set the input validator."""
        self._input_validator = validator
        return self

    def with_guardrails(self, guardrails: GuardrailProtocol) -> "RuntimeBuilder":
        """Set the output guardrails."""
        self._guardrails = guardrails
        return self

    def with_tracer(self, tracer: TracerProtocol) -> "RuntimeBuilder":
        """Set the tracer."""
        self._tracer = tracer
        return self

    def with_config(self, config: RuntimeConfig) -> "RuntimeBuilder":
        """Set the runtime configuration."""
        self._config = config
        return self

    def with_system_prompt(self, prompt: str) -> "RuntimeBuilder":
        """Set the system prompt."""
        self._config.system_prompt = prompt
        return self

    def with_model(self, model: str) -> "RuntimeBuilder":
        """Set the model."""
        self._config.model = model
        return self

    def with_temperature(self, temperature: float) -> "RuntimeBuilder":
        """Set the temperature."""
        self._config.temperature = temperature
        return self

    def with_max_iterations(self, max_iterations: int) -> "RuntimeBuilder":
        """Set the maximum iterations."""
        self._config.max_iterations = max_iterations
        return self

    def with_max_tool_calls(self, max_tool_calls: int) -> "RuntimeBuilder":
        """Set the maximum tool calls."""
        self._config.max_tool_calls = max_tool_calls
        return self

    def with_timeout(self, timeout_seconds: float) -> "RuntimeBuilder":
        """Set the execution timeout."""
        self._config.timeout_seconds = timeout_seconds
        return self

    def disable_memory(self) -> "RuntimeBuilder":
        """Disable memory."""
        self._config.enable_memory = False
        return self

    def disable_rag(self) -> "RuntimeBuilder":
        """Disable RAG."""
        self._config.enable_rag = False
        return self

    def disable_tools(self) -> "RuntimeBuilder":
        """Disable tools."""
        self._config.enable_tools = False
        return self

    def build(self) -> AgentRuntime:
        """
        Build the AgentRuntime.

        Raises:
            ValueError: If LLM is not set
        """
        if self._llm is None:
            raise ValueError("LLM adapter is required. Use .with_llm() to set it.")

        return AgentRuntime(
            llm=self._llm,
            tool_executor=self._tool_executor,
            memory=self._memory,
            retriever=self._retriever,
            input_validator=self._input_validator,
            guardrails=self._guardrails,
            tracer=self._tracer,
            config=self._config,
        )


async def create_default_runtime_from_settings() -> AgentRuntime:
    """
    Create a runtime using default settings.

    This is a convenience function for getting started quickly.
    For production, use create_runtime() or RuntimeBuilder.
    """
    from src.config import get_settings

    settings = get_settings()

    # Create LLM adapter based on provider
    llm: LLMAdapterProtocol

    if settings.llm.default_provider == "openai":
        from src.reasoning.llm.openai_adapter import OpenAIAdapter

        llm = OpenAIAdapter(
            model=settings.llm.default_model,
            api_key=settings.llm.openai_api_key,
        )
    elif settings.llm.default_provider == "anthropic":
        from src.reasoning.llm.anthropic_adapter import AnthropicAdapter

        llm = AnthropicAdapter(
            model=settings.llm.default_model,
            api_key=settings.llm.anthropic_api_key,
        )
    else:
        raise ValueError(f"Unknown provider: {settings.llm.default_provider}")

    # Create tool executor
    from src.tools import ToolExecutor, ToolRegistry
    from src.tools.builtin import register_builtin_tools

    registry = ToolRegistry()
    register_builtin_tools(registry)
    tool_executor = ToolExecutor(registry)

    # Create config
    config = RuntimeConfig(
        model=settings.llm.default_model,
        temperature=settings.llm.default_temperature,
        max_iterations=10,
    )

    return create_runtime(
        llm=llm,
        tool_executor=tool_executor,
        config=config,
    )
