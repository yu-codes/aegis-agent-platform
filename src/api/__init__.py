"""
Interface & Serving Layer

FastAPI-based API endpoints, streaming, and rate limiting.
"""

from src.api.app import create_app
from src.api.dependencies import get_agent_runtime, get_session_manager
from src.api.middleware import AuthMiddleware, RateLimitMiddleware, TracingMiddleware
from src.api.routes import admin, chat, sessions, tools
from src.api.streaming import EventStream, StreamingResponse

__all__ = [
    "AuthMiddleware",
    "EventStream",
    # Middleware
    "RateLimitMiddleware",
    # Streaming
    "StreamingResponse",
    "TracingMiddleware",
    "admin",
    # Routes
    "chat",
    # App
    "create_app",
    # Dependencies
    "get_agent_runtime",
    "get_session_manager",
    "sessions",
    "tools",
]
