"""
Settings Management

Provides centralized, type-safe configuration using Pydantic.
Supports environment variables, .env files, and hierarchical config.

Design decisions:
- Using pydantic-settings for validation and type coercion
- Immutable settings after initialization (frozen model)
- Separate concerns: infra settings vs. model settings vs. feature flags
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    """Redis connection configuration."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    enabled: bool = Field(default=True, description="Enable Redis (false uses in-memory)")
    host: str = Field(default="localhost", description="Redis server hostname")
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: SecretStr | None = Field(default=None)
    ssl: bool = Field(default=False)
    max_connections: int = Field(default=10, ge=1)
    socket_timeout: float = Field(default=5.0, gt=0)

    @property
    def url(self) -> str:
        """Construct Redis URL for connection."""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    # Provider selection (stub = offline mode, no API key required)
    default_provider: Literal["openai", "anthropic", "local", "stub"] = "openai"

    # Offline mode - forces stub adapter regardless of provider setting
    offline_mode: bool = Field(
        default=False,
        description="Force offline mode using stub adapter (no API calls)",
    )

    # OpenAI settings
    openai_api_key: SecretStr | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)
    openai_default_model: str = Field(default="gpt-4o")
    openai_org_id: str | None = Field(default=None)

    # Anthropic settings
    anthropic_api_key: SecretStr | None = Field(default=None)
    anthropic_default_model: str = Field(default="claude-sonnet-4-20250514")

    # Local/Custom endpoint
    local_base_url: str = Field(default="http://localhost:8080/v1")
    local_default_model: str = Field(default="local-model")

    # Stub adapter settings
    stub_model_name: str = Field(default="stub-model-v1")
    stub_stream_delay_ms: int = Field(default=20, ge=0)

    # Shared settings
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=4096, ge=1)
    request_timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0)

    @property
    def default_model(self) -> str:
        """Get default model based on provider."""
        if self.offline_mode or self.default_provider == "stub":
            return self.stub_model_name
        elif self.default_provider == "openai":
            return self.openai_default_model
        elif self.default_provider == "anthropic":
            return self.anthropic_default_model
        else:
            return self.local_default_model

    @property
    def effective_provider(self) -> str:
        """Get effective provider (stub if offline)."""
        if self.offline_mode:
            return "stub"
        return self.default_provider


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_")

    provider: Literal["faiss", "milvus", "memory"] = "faiss"

    # FAISS settings
    faiss_index_path: str = Field(default="./data/faiss_index")

    # Milvus settings
    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)
    milvus_collection: str = Field(default="aegis_documents")

    # Embedding settings
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=1536)

    # Retrieval settings
    default_top_k: int = Field(default=5, ge=1)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class SafetySettings(BaseSettings):
    """Safety and governance configuration."""

    model_config = SettingsConfigDict(env_prefix="SAFETY_")

    enable_input_guardrails: bool = Field(default=True)
    enable_output_guardrails: bool = Field(default=True)
    enable_tool_access_control: bool = Field(default=True)
    enable_audit_logging: bool = Field(default=True)

    # Rate limiting
    max_requests_per_minute: int = Field(default=60, ge=1)
    max_tokens_per_minute: int = Field(default=100000, ge=1)

    # Content filtering
    blocked_patterns_file: str | None = Field(default=None)
    max_input_length: int = Field(default=32000, ge=1)
    max_output_length: int = Field(default=16000, ge=1)


class ObservabilitySettings(BaseSettings):
    """Observability configuration."""

    model_config = SettingsConfigDict(env_prefix="OBS_")

    enable_tracing: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)

    # Tracing
    trace_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    trace_export_endpoint: str | None = Field(default=None)

    # Metrics
    metrics_port: int = Field(default=9090)
    metrics_path: str = Field(default="/metrics")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    # Cost tracking
    enable_cost_tracking: bool = Field(default=True)

    # Rate limiting
    rate_limit_enabled: bool = Field(default=False)
    rate_limit_rpm: int = Field(default=60, ge=1)


class Settings(BaseSettings):
    """
    Master settings aggregator.

    This is the single source of truth for all configuration.
    Sub-settings are composed here to maintain clear boundaries.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,  # Immutable after creation
    )

    # Application metadata
    app_name: str = Field(default="Aegis")
    app_version: str = Field(default="0.1.0")
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = Field(default=False)

    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_prefix: str = Field(default="/api/v1")
    cors_origins: list[str] = Field(default=["*"])

    # Component settings (composed)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    # Feature flags
    enable_multi_agent: bool = Field(default=False)
    enable_reflection: bool = Field(default=False)
    enable_plugins: bool = Field(default=True)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to ensure only one settings instance exists.
    This is safe because settings are frozen/immutable.
    """
    return Settings()
