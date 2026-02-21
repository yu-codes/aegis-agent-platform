"""
Application Factory

FastAPI application creation.

Based on: src/api/app.py
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from apps.api_server.middleware import (
    RequestTracingMiddleware,
    RateLimitMiddleware,
    ErrorHandlerMiddleware,
)
from apps.api_server.routes import (
    chat,
    health,
    sessions,
    tools,
    admin,
)

# Global app instance
_app: FastAPI | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await startup(app)
    yield
    # Shutdown
    await shutdown(app)


async def startup(app: FastAPI) -> None:
    """Initialize application resources."""
    from apps.api_server.dependencies import init_dependencies

    print("Starting Aegis Agent Platform API...")

    # Initialize dependencies
    await init_dependencies()

    # Store in app state
    app.state.initialized = True

    print("API startup complete.")


async def shutdown(app: FastAPI) -> None:
    """Cleanup application resources."""
    from apps.api_server.dependencies import cleanup_dependencies

    print("Shutting down Aegis Agent Platform API...")

    await cleanup_dependencies()

    print("API shutdown complete.")


def create_app(
    title: str = "Aegis Agent Platform",
    version: str = "1.0.0",
    debug: bool = False,
    enable_docs: bool = True,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        title: Application title
        version: Application version
        debug: Debug mode
        enable_docs: Enable OpenAPI docs
        cors_origins: Allowed CORS origins

    Returns:
        Configured FastAPI app
    """
    global _app

    app = FastAPI(
        title=title,
        version=version,
        debug=debug,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        openapi_url="/openapi.json" if enable_docs else None,
        lifespan=lifespan,
    )

    # Configure CORS
    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(RequestTracingMiddleware)

    # Rate limiting (optional)
    if os.getenv("ENABLE_RATE_LIMIT", "false").lower() == "true":
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
        )

    # Register routes
    app.include_router(health.router, tags=["health"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
    app.include_router(tools.router, prefix="/api/v1", tags=["tools"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])

    # Mount static files for web UI
    static_path = Path(__file__).parent.parent / "web-ui" / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

        # Serve web UI at /ui
        @app.get("/ui", include_in_schema=False)
        @app.get("/ui/{path:path}", include_in_schema=False)
        async def serve_ui(path: str = ""):
            """Serve Web UI."""
            index_path = static_path / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            return {"error": "Web UI not found"}

        # Serve test page at /test
        @app.get("/test", include_in_schema=False)
        async def serve_test():
            """Serve test page."""
            test_path = static_path / "test.html"
            if test_path.exists():
                return FileResponse(test_path)
            return {"error": "Test page not found"}

        # Serve simple page at /simple
        @app.get("/simple", include_in_schema=False)
        async def serve_simple():
            """Serve simple UI page."""
            simple_path = static_path / "simple.html"
            if simple_path.exists():
                return FileResponse(simple_path)
            return {"error": "Simple page not found"}

    _app = app
    return app


def get_app() -> FastAPI:
    """Get or create the application instance."""
    global _app
    if _app is None:
        _app = create_app()
    return _app
