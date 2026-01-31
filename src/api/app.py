"""
FastAPI Application Factory

Creates and configures the main application.

Design decisions:
- Factory pattern for testability
- Middleware composition
- Lifespan management for component lifecycle
- DomainAwareRuntime as the domain-aware orchestration point
- CORS configuration
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[dict[str, Any]]:
    """
    Application lifespan manager.

    Initializes all components on startup:
    - Redis connection (if configured)
    - Session manager
    - Tool registry and executor
    - Domain registry (loads domain profiles)
    - DomainAwareRuntime (the domain-aware orchestration point)

    All components are stored in app.state.components for DI.
    """
    # Startup
    settings = get_settings()

    # Initialize components
    state: dict[str, Any] = {}

    # Initialize Redis if configured
    if settings.redis.enabled:
        try:
            import redis.asyncio as redis

            redis_password = settings.redis.password.get_secret_value() if settings.redis.password else None
            redis_client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                password=redis_password,
                db=settings.redis.db,
                decode_responses=True,
            )
            state["redis"] = redis_client
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")

    # Initialize session manager
    from src.memory import InMemorySessionBackend, RedisSessionBackend, SessionManager

    if "redis" in state:
        redis_url = f"redis://{settings.redis.host}:{settings.redis.port}/{settings.redis.db}"
        session_backend: RedisSessionBackend | InMemorySessionBackend = RedisSessionBackend(redis_url)
    else:
        session_backend = InMemorySessionBackend()

    state["session_manager"] = SessionManager(session_backend)

    # Initialize tool registry and executor
    from src.tools import ToolExecutor, ToolRegistry
    from src.tools.builtin import register_builtin_tools

    registry = ToolRegistry()  # type: ignore
    register_builtin_tools(registry)
    state["tool_registry"] = registry

    tool_executor = ToolExecutor(registry)
    state["tool_executor"] = tool_executor

    # ====================================================================
    # Initialize Domain System
    # ====================================================================
    from src.domains import (
        DomainAwareRuntime,
        DomainRegistry,
        create_default_resolver,
    )

    # Load domain profiles from config directory
    domains_path = Path(__file__).parent.parent.parent / "config" / "domains"

    if domains_path.exists():
        domain_registry = DomainRegistry.from_directory(domains_path)
        print(f"Loaded {len(domain_registry)} domain profiles from {domains_path}")
        for profile in domain_registry:
            print(f"  - {profile.name} v{profile.version}: {profile.description[:50]}...")
    else:
        # Create empty registry with default domain
        domain_registry = DomainRegistry()
        print("No domain profiles found, using default domain only")

    state["domain_registry"] = domain_registry

    # Create domain resolver with default keyword rules
    domain_resolver = create_default_resolver(domain_registry)
    state["domain_resolver"] = domain_resolver

    # ====================================================================
    # Initialize AgentRuntime (base) and DomainAwareRuntime
    # ====================================================================
    from src.runtime import RuntimeConfig, create_runtime

    # Create LLM adapter based on settings
    llm_adapter = await _create_llm_adapter(settings)

    if llm_adapter:
        # Create base runtime config (will be overridden by domain profiles)
        runtime_config = RuntimeConfig(
            model=settings.llm.default_model,
            temperature=settings.llm.default_temperature,
            system_prompt="You are a helpful AI assistant powered by the Aegis platform.",
            max_iterations=settings.agent.max_iterations if hasattr(settings, "agent") else 10,
            max_tool_calls=settings.agent.max_tool_calls if hasattr(settings, "agent") else 20,
            enable_memory=True,
            enable_rag=True,
            enable_tools=True,
        )

        # Create base AgentRuntime (for non-domain-aware access)
        base_runtime = create_runtime(
            llm=llm_adapter,
            tool_executor=tool_executor,
            config=runtime_config,
        )
        state["agent_runtime"] = base_runtime

        # Create DomainAwareRuntime (the recommended runtime)
        domain_aware_runtime = DomainAwareRuntime(
            registry=domain_registry,
            llm=llm_adapter,
            tool_executor=tool_executor,
            resolver=domain_resolver,
            enable_inference=True,
            inference_threshold=0.6,
        )
        state["domain_aware_runtime"] = domain_aware_runtime

        print("Domain-aware runtime initialized successfully")
    else:
        # Create a placeholder runtime for development/testing
        state["agent_runtime"] = None
        state["domain_aware_runtime"] = None
        print("Warning: No LLM adapter configured. Chat endpoints will not work.")

    # Store in app state
    app.state.components = state

    yield state

    # Shutdown
    if "redis" in state:
        await state["redis"].close()


async def _create_llm_adapter(settings: Any) -> Any:
    """
    Create the appropriate LLM adapter based on settings.

    Priority order:
    1. If offline_mode=True or provider="stub": Use StubLLMAdapter
    2. If OpenAI API key provided and provider="openai": Use OpenAI
    3. If Anthropic API key provided and provider="anthropic": Use Anthropic
    4. Fallback: Use StubLLMAdapter (ensures system always works)

    Returns:
        An LLM adapter (never None - stub is always available)
    """
    try:
        # Get effective provider (respects offline_mode flag)
        provider = getattr(
            settings.llm, "effective_provider", getattr(settings.llm, "default_provider", "openai")
        )

        # Check for offline/stub mode first
        if provider == "stub" or getattr(settings.llm, "offline_mode", False):
            from src.reasoning.llm.stub_adapter import StubLLMAdapter

            print("🔌 Using STUB LLM adapter (offline mode)")
            return StubLLMAdapter(
                model=getattr(settings.llm, "stub_model_name", "stub-model-v1"),
                stream_delay_ms=getattr(settings.llm, "stub_stream_delay_ms", 20),
            )

        if provider == "openai":
            api_key = getattr(settings.llm, "openai_api_key", None)
            if api_key:
                from src.reasoning.llm.openai_adapter import OpenAIAdapter

                print("🔌 Using OpenAI LLM adapter")
                return OpenAIAdapter(settings=settings.llm)

        elif provider == "anthropic":
            api_key = getattr(settings.llm, "anthropic_api_key", None)
            if api_key:
                from src.reasoning.llm.anthropic_adapter import AnthropicAdapter

                print("🔌 Using Anthropic LLM adapter")
                return AnthropicAdapter(settings=settings.llm)

        # Fallback to stub if no valid provider configured
        print("⚠️  No LLM API key configured - falling back to STUB adapter")
        from src.reasoning.llm.stub_adapter import StubLLMAdapter

        return StubLLMAdapter()

    except Exception as e:
        print(f"⚠️  Failed to create LLM adapter ({e}) - using STUB adapter")
        from src.reasoning.llm.stub_adapter import StubLLMAdapter

        return StubLLMAdapter()


def create_app(
    title: str = "Aegis Agent Platform",
    version: str = "1.0.0",
    debug: bool = False,
    **kwargs: Any,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        title: API title
        version: API version
        debug: Enable debug mode
        **kwargs: Additional FastAPI arguments
    """
    settings = get_settings()

    app = FastAPI(
        title=title,
        version=version,
        description="Enterprise-Grade Modular AI Agent Platform",
        debug=debug,
        lifespan=lifespan,
        **kwargs,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    from src.api.middleware import (
        ErrorHandlingMiddleware,
        RateLimitMiddleware,
        TracingMiddleware,
    )

    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(TracingMiddleware)

    if settings.observability.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.observability.rate_limit_rpm,
        )

    # Include routers
    from src.api.routes import admin, chat, domains, health, sessions, tools

    app.include_router(health.router, tags=["health"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    app.include_router(domains.router, prefix="/api/v1", tags=["domains"])
    app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
    app.include_router(tools.router, prefix="/api/v1", tags=["tools"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])

    return app


# Default app instance for running directly
app = create_app()
