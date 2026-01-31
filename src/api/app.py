"""
FastAPI Application Factory

Creates and configures the main application.

Design decisions:
- Factory pattern for testability
- Middleware composition
- Lifespan management for component lifecycle
- AgentRuntime as the single orchestration point
- CORS configuration
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

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
    - AgentRuntime (the orchestration point)
    
    All components are stored in app.state.components for DI.
    """
    # Startup
    settings = get_settings()
    
    # Initialize components
    state = {}
    
    # Initialize Redis if configured
    if settings.redis.enabled:
        try:
            import redis.asyncio as redis
            redis_client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                password=settings.redis.password,
                db=settings.redis.db,
                decode_responses=True,
            )
            state["redis"] = redis_client
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
    
    # Initialize session manager
    from src.memory import SessionManager, InMemorySessionBackend, RedisSessionBackend
    
    if "redis" in state:
        session_backend = RedisSessionBackend(state["redis"])
    else:
        session_backend = InMemorySessionBackend()
    
    state["session_manager"] = SessionManager(session_backend)
    
    # Initialize tool registry and executor
    from src.tools import ToolRegistry, ToolExecutor
    from src.tools.builtin import register_builtin_tools
    
    registry = ToolRegistry()
    register_builtin_tools(registry)
    state["tool_registry"] = registry
    
    tool_executor = ToolExecutor(registry)
    state["tool_executor"] = tool_executor
    
    # Initialize AgentRuntime - the SINGLE orchestration point
    # This is where all agent execution flows through
    from src.runtime import AgentRuntime, RuntimeConfig, create_runtime
    
    # Create LLM adapter based on settings
    llm_adapter = await _create_llm_adapter(settings)
    
    if llm_adapter:
        # Create runtime with all components
        runtime_config = RuntimeConfig(
            model=settings.llm.default_model,
            temperature=settings.llm.default_temperature,
            system_prompt="You are a helpful AI assistant powered by the Aegis platform.",
            max_iterations=settings.agent.max_iterations if hasattr(settings, 'agent') else 10,
            max_tool_calls=settings.agent.max_tool_calls if hasattr(settings, 'agent') else 20,
            enable_memory=True,
            enable_rag=True,
            enable_tools=True,
        )
        
        state["agent_runtime"] = create_runtime(
            llm=llm_adapter,
            tool_executor=tool_executor,
            config=runtime_config,
        )
    else:
        # Create a placeholder runtime for development/testing
        state["agent_runtime"] = None
        print("Warning: No LLM adapter configured. Chat endpoints will not work.")
    
    # Store in app state
    app.state.components = state
    
    yield state
    
    # Shutdown
    if "redis" in state:
        await state["redis"].close()


async def _create_llm_adapter(settings):
    """
    Create the appropriate LLM adapter based on settings.
    
    Returns None if no valid configuration is found.
    """
    try:
        provider = getattr(settings.llm, 'default_provider', 'openai')
        
        if provider == "openai":
            api_key = getattr(settings.llm, 'openai_api_key', None)
            if api_key:
                from src.reasoning.llm.openai_adapter import OpenAIAdapter
                return OpenAIAdapter(
                    model=settings.llm.default_model,
                    api_key=api_key,
                )
        
        elif provider == "anthropic":
            api_key = getattr(settings.llm, 'anthropic_api_key', None)
            if api_key:
                from src.reasoning.llm.anthropic_adapter import AnthropicAdapter
                return AnthropicAdapter(
                    model=settings.llm.default_model,
                    api_key=api_key,
                )
    except Exception as e:
        print(f"Failed to create LLM adapter: {e}")
    
    return None


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
        TracingMiddleware,
        RateLimitMiddleware,
        ErrorHandlingMiddleware,
    )
    
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(TracingMiddleware)
    
    if settings.observability.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.observability.rate_limit_rpm,
        )
    
    # Include routers
    from src.api.routes import chat, sessions, tools, admin, health
    
    app.include_router(health.router, tags=["health"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
    app.include_router(tools.router, prefix="/api/v1", tags=["tools"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
    
    return app


# Default app instance for running directly
app = create_app()
