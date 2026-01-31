"""
FastAPI Application Factory

Creates and configures the main application.

Design decisions:
- Factory pattern for testability
- Middleware composition
- Lifespan management
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
    
    Handles startup and shutdown.
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
    
    # Initialize tool registry
    from src.tools import ToolRegistry
    from src.tools.builtin import register_builtin_tools
    
    registry = ToolRegistry()
    register_builtin_tools(registry)
    state["tool_registry"] = registry
    
    # Store in app state
    app.state.components = state
    
    yield state
    
    # Shutdown
    if "redis" in state:
        await state["redis"].close()


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
