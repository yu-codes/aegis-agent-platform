"""
Model Routing

Intelligent routing of requests to appropriate LLM models.
Supports load balancing, fallback chains, and capability-based routing.

Design decisions:
- Models are registered with capabilities and constraints
- Router selects based on task requirements, not just availability
- Fallback chains ensure resilience
- Cost-aware routing for budget management
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelCapability(str, Enum):
    """Capabilities that models may support."""
    
    CHAT = "chat"
    COMPLETION = "completion"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    CODE = "code"
    REASONING = "reasoning"
    LONG_CONTEXT = "long_context"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for a specific model.
    
    Immutable to prevent accidental modification.
    Contains all metadata needed for routing decisions.
    """
    
    name: str
    provider: ModelProvider
    capabilities: frozenset[ModelCapability]
    
    # Context limits
    max_input_tokens: int
    max_output_tokens: int
    
    # Cost (USD per 1M tokens)
    input_cost_per_million: float
    output_cost_per_million: float
    
    # Performance characteristics
    avg_latency_ms: float = 1000.0
    
    # Constraints
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000
    
    # Fallback model (if this one fails)
    fallback_model: str | None = None
    
    # Additional provider-specific settings
    extra: dict[str, Any] = field(default_factory=dict)
    
    @property
    def supports_function_calling(self) -> bool:
        return ModelCapability.FUNCTION_CALLING in self.capabilities
    
    @property
    def supports_vision(self) -> bool:
        return ModelCapability.VISION in self.capabilities
    
    @property
    def context_window(self) -> int:
        return self.max_input_tokens + self.max_output_tokens


# Pre-defined model configurations
PREDEFINED_MODELS: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider=ModelProvider.OPENAI,
        capabilities=frozenset({
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.CODE,
            ModelCapability.STREAMING,
            ModelCapability.JSON_MODE,
            ModelCapability.LONG_CONTEXT,
        }),
        max_input_tokens=128000,
        max_output_tokens=16384,
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
        avg_latency_ms=800,
        fallback_model="gpt-4o-mini",
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        capabilities=frozenset({
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.CODE,
            ModelCapability.STREAMING,
            ModelCapability.JSON_MODE,
        }),
        max_input_tokens=128000,
        max_output_tokens=16384,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
        avg_latency_ms=500,
    ),
    "claude-sonnet-4-20250514": ModelConfig(
        name="claude-sonnet-4-20250514",
        provider=ModelProvider.ANTHROPIC,
        capabilities=frozenset({
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT,
        }),
        max_input_tokens=200000,
        max_output_tokens=8192,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
        avg_latency_ms=1200,
        fallback_model="claude-3-5-haiku-20241022",
    ),
    "claude-3-5-haiku-20241022": ModelConfig(
        name="claude-3-5-haiku-20241022",
        provider=ModelProvider.ANTHROPIC,
        capabilities=frozenset({
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.CODE,
            ModelCapability.STREAMING,
        }),
        max_input_tokens=200000,
        max_output_tokens=8192,
        input_cost_per_million=0.80,
        output_cost_per_million=4.00,
        avg_latency_ms=400,
    ),
}


@dataclass
class RoutingContext:
    """Context for making routing decisions."""
    
    required_capabilities: set[ModelCapability] = field(default_factory=set)
    preferred_provider: ModelProvider | None = None
    max_cost_per_million: float | None = None
    min_context_window: int | None = None
    prefer_speed: bool = False
    prefer_quality: bool = False


class ModelRouter:
    """
    Intelligent model router.
    
    Selects the best model based on:
    - Required capabilities
    - Cost constraints
    - Performance requirements
    - Provider preferences
    - Fallback availability
    """
    
    def __init__(self):
        self._models: dict[str, ModelConfig] = dict(PREDEFINED_MODELS)
        self._disabled_models: set[str] = set()
    
    def register_model(self, config: ModelConfig) -> None:
        """Register a new model configuration."""
        self._models[config.name] = config
    
    def disable_model(self, model_name: str) -> None:
        """Temporarily disable a model (e.g., due to errors)."""
        self._disabled_models.add(model_name)
    
    def enable_model(self, model_name: str) -> None:
        """Re-enable a disabled model."""
        self._disabled_models.discard(model_name)
    
    def get_model(self, name: str) -> ModelConfig | None:
        """Get a specific model configuration."""
        if name in self._disabled_models:
            return None
        return self._models.get(name)
    
    def select_model(self, context: RoutingContext) -> ModelConfig | None:
        """
        Select the best model for the given context.
        
        Selection priority:
        1. Filter by required capabilities
        2. Filter by cost constraints
        3. Filter by context window
        4. Rank by preference (speed vs quality)
        5. Return best match
        """
        candidates = []
        
        for name, config in self._models.items():
            # Skip disabled models
            if name in self._disabled_models:
                continue
            
            # Check capabilities
            if context.required_capabilities:
                if not context.required_capabilities.issubset(config.capabilities):
                    continue
            
            # Check provider preference
            if context.preferred_provider:
                if config.provider != context.preferred_provider:
                    continue
            
            # Check cost constraint
            if context.max_cost_per_million is not None:
                avg_cost = (config.input_cost_per_million + config.output_cost_per_million) / 2
                if avg_cost > context.max_cost_per_million:
                    continue
            
            # Check context window
            if context.min_context_window is not None:
                if config.context_window < context.min_context_window:
                    continue
            
            candidates.append(config)
        
        if not candidates:
            return None
        
        # Sort by preference
        if context.prefer_speed:
            candidates.sort(key=lambda m: m.avg_latency_ms)
        elif context.prefer_quality:
            # Higher cost often correlates with quality
            candidates.sort(
                key=lambda m: m.input_cost_per_million + m.output_cost_per_million,
                reverse=True,
            )
        else:
            # Default: balance of cost and capability
            candidates.sort(
                key=lambda m: (
                    len(m.capabilities),  # More capabilities
                    -m.avg_latency_ms / 1000,  # Faster
                ),
                reverse=True,
            )
        
        return candidates[0]
    
    def get_fallback_chain(self, model_name: str, max_depth: int = 3) -> list[ModelConfig]:
        """
        Get the fallback chain for a model.
        
        Returns ordered list of models to try if primary fails.
        Limited depth to prevent infinite loops.
        """
        chain = []
        current = model_name
        depth = 0
        seen = set()
        
        while current and depth < max_depth:
            if current in seen:
                break  # Prevent cycles
            seen.add(current)
            
            config = self.get_model(current)
            if config:
                chain.append(config)
                current = config.fallback_model
            else:
                break
            depth += 1
        
        return chain
    
    def list_models(
        self,
        provider: ModelProvider | None = None,
        capability: ModelCapability | None = None,
    ) -> list[ModelConfig]:
        """List available models, optionally filtered."""
        models = []
        
        for name, config in self._models.items():
            if name in self._disabled_models:
                continue
            if provider and config.provider != provider:
                continue
            if capability and capability not in config.capabilities:
                continue
            models.append(config)
        
        return models


# Module-level singleton
_router: ModelRouter | None = None


def get_model_router() -> ModelRouter:
    """Get or create the global model router."""
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
