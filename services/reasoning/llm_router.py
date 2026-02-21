"""
LLM Router

Intelligent model selection and load balancing.

Design decisions:
- Route based on task complexity
- Cost optimization
- Fallback chains
- Rate limit management
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class ModelProvider(str, Enum):
    """Supported model providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    STUB = "stub"


class TaskComplexity(str, Enum):
    """Task complexity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModelConfig:
    """Configuration for a model."""

    name: str
    provider: ModelProvider
    model_id: str
    max_tokens: int = 4096
    temperature: float = 0.7

    # Capabilities
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True

    # Cost (per 1k tokens)
    input_cost: float = 0.0
    output_cost: float = 0.0

    # Limits
    rate_limit_rpm: int = 60
    context_window: int = 100000

    # Metadata
    complexity_level: TaskComplexity = TaskComplexity.MEDIUM

    def estimated_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token counts."""
        return input_tokens / 1000 * self.input_cost + output_tokens / 1000 * self.output_cost


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    model: ModelConfig
    reason: str
    alternatives: list[ModelConfig] = field(default_factory=list)
    estimated_cost: float = 0.0


@dataclass
class ModelUsage:
    """Track model usage for rate limiting."""

    requests: int = 0
    tokens: int = 0
    last_request: datetime | None = None
    window_start: datetime = field(default_factory=datetime.utcnow)


class LLMRouter:
    """
    Intelligent LLM routing.

    Selects the best model based on task requirements.
    """

    def __init__(self):
        self._models: dict[str, ModelConfig] = {}
        self._usage: dict[str, ModelUsage] = {}
        self._fallback_chain: list[str] = []

        # Register default models
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default model configurations."""
        self.register(
            ModelConfig(
                name="claude-3-5-sonnet",
                provider=ModelProvider.ANTHROPIC,
                model_id="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                supports_tools=True,
                supports_streaming=True,
                input_cost=0.003,
                output_cost=0.015,
                rate_limit_rpm=60,
                context_window=200000,
                complexity_level=TaskComplexity.HIGH,
            )
        )

        self.register(
            ModelConfig(
                name="claude-3-haiku",
                provider=ModelProvider.ANTHROPIC,
                model_id="claude-3-haiku-20240307",
                max_tokens=4096,
                supports_tools=True,
                supports_streaming=True,
                input_cost=0.00025,
                output_cost=0.00125,
                rate_limit_rpm=100,
                context_window=200000,
                complexity_level=TaskComplexity.LOW,
            )
        )

        self.register(
            ModelConfig(
                name="gpt-4o",
                provider=ModelProvider.OPENAI,
                model_id="gpt-4o",
                max_tokens=4096,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True,
                input_cost=0.005,
                output_cost=0.015,
                rate_limit_rpm=120,
                context_window=128000,
                complexity_level=TaskComplexity.HIGH,
            )
        )

        self.register(
            ModelConfig(
                name="gpt-4o-mini",
                provider=ModelProvider.OPENAI,
                model_id="gpt-4o-mini",
                max_tokens=4096,
                supports_tools=True,
                supports_streaming=True,
                input_cost=0.00015,
                output_cost=0.0006,
                rate_limit_rpm=200,
                context_window=128000,
                complexity_level=TaskComplexity.LOW,
            )
        )

        self.register(
            ModelConfig(
                name="stub",
                provider=ModelProvider.STUB,
                model_id="stub-v1",
                max_tokens=4096,
                supports_tools=True,
                supports_streaming=True,
                input_cost=0.0,
                output_cost=0.0,
                rate_limit_rpm=1000,
                context_window=100000,
                complexity_level=TaskComplexity.LOW,
            )
        )

        # Default fallback chain
        self._fallback_chain = [
            "claude-3-5-sonnet",
            "gpt-4o",
            "claude-3-haiku",
            "gpt-4o-mini",
            "stub",
        ]

    def register(self, config: ModelConfig) -> None:
        """Register a model configuration."""
        self._models[config.name] = config
        self._usage[config.name] = ModelUsage()

    def route(
        self,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        requires_tools: bool = False,
        requires_vision: bool = False,
        requires_streaming: bool = False,
        max_tokens: int | None = None,
        prefer_cost: bool = False,
        preferred_provider: ModelProvider | None = None,
    ) -> RoutingDecision:
        """
        Route to the best model for the task.

        Args:
            complexity: Task complexity level
            requires_tools: Whether tools are needed
            requires_vision: Whether vision is needed
            requires_streaming: Whether streaming is needed
            max_tokens: Required max output tokens
            prefer_cost: Optimize for cost
            preferred_provider: Prefer a specific provider

        Returns:
            Routing decision with selected model
        """
        candidates = self._filter_candidates(
            requires_tools=requires_tools,
            requires_vision=requires_vision,
            requires_streaming=requires_streaming,
            max_tokens=max_tokens,
            preferred_provider=preferred_provider,
        )

        if not candidates:
            # Use first available from fallback chain
            for name in self._fallback_chain:
                if name in self._models:
                    return RoutingDecision(
                        model=self._models[name],
                        reason="No suitable model found, using fallback",
                    )

        # Score candidates
        scored = []
        for model in candidates:
            score = self._score_model(model, complexity, prefer_cost)
            scored.append((score, model))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = scored[0][1]
        alternatives = [m for _, m in scored[1:3]]

        return RoutingDecision(
            model=selected,
            reason=f"Best match for {complexity.value} complexity task",
            alternatives=alternatives,
        )

    def _filter_candidates(
        self,
        requires_tools: bool,
        requires_vision: bool,
        requires_streaming: bool,
        max_tokens: int | None,
        preferred_provider: ModelProvider | None,
    ) -> list[ModelConfig]:
        """Filter models by requirements."""
        candidates = []

        for model in self._models.values():
            # Check requirements
            if requires_tools and not model.supports_tools:
                continue
            if requires_vision and not model.supports_vision:
                continue
            if requires_streaming and not model.supports_streaming:
                continue
            if max_tokens and model.max_tokens < max_tokens:
                continue

            # Check rate limit
            if not self._check_rate_limit(model.name):
                continue

            # Prefer provider if specified
            if preferred_provider and model.provider != preferred_provider:
                continue

            candidates.append(model)

        # If preferred provider filter resulted in no candidates, remove filter
        if not candidates and preferred_provider:
            return self._filter_candidates(
                requires_tools=requires_tools,
                requires_vision=requires_vision,
                requires_streaming=requires_streaming,
                max_tokens=max_tokens,
                preferred_provider=None,
            )

        return candidates

    def _score_model(
        self,
        model: ModelConfig,
        complexity: TaskComplexity,
        prefer_cost: bool,
    ) -> float:
        """Score a model for selection."""
        score = 0.0

        # Complexity match
        complexity_order = [
            TaskComplexity.LOW,
            TaskComplexity.MEDIUM,
            TaskComplexity.HIGH,
            TaskComplexity.CRITICAL,
        ]
        model_level = complexity_order.index(model.complexity_level)
        task_level = complexity_order.index(complexity)

        if model_level >= task_level:
            score += 10 - abs(model_level - task_level) * 2
        else:
            score -= 5  # Penalize under-powered models

        # Cost optimization
        if prefer_cost:
            avg_cost = (model.input_cost + model.output_cost) / 2
            score -= avg_cost * 100  # Lower cost = higher score

        # Rate limit headroom
        usage = self._usage.get(model.name, ModelUsage())
        headroom = 1 - (usage.requests / model.rate_limit_rpm if model.rate_limit_rpm else 0)
        score += headroom * 5

        return score

    def _check_rate_limit(self, model_name: str) -> bool:
        """Check if model is within rate limit."""
        usage = self._usage.get(model_name)
        if not usage:
            return True

        model = self._models.get(model_name)
        if not model:
            return False

        # Reset window if needed
        now = datetime.utcnow()
        if now - usage.window_start > timedelta(minutes=1):
            usage.requests = 0
            usage.window_start = now

        return usage.requests < model.rate_limit_rpm

    def record_usage(
        self,
        model_name: str,
        tokens: int = 0,
    ) -> None:
        """Record model usage."""
        if model_name not in self._usage:
            self._usage[model_name] = ModelUsage()

        usage = self._usage[model_name]
        usage.requests += 1
        usage.tokens += tokens
        usage.last_request = datetime.utcnow()

    def get_model(self, name: str) -> ModelConfig | None:
        """Get model by name."""
        return self._models.get(name)

    def list_models(self) -> list[ModelConfig]:
        """List all registered models."""
        return list(self._models.values())

    def set_fallback_chain(self, chain: list[str]) -> None:
        """Set the fallback chain."""
        self._fallback_chain = chain
