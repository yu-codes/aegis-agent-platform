"""
Interface & Serving Layer

FastAPI-based API endpoints, streaming, and rate limiting.
"""

from src.api.app import create_app
from src.api.routes import chat, sessions, tools, admin
from src.api.middleware import RateLimitMiddleware, AuthMiddleware, TracingMiddleware
from src.api.streaming import StreamingResponse, EventStream
from src.api.dependencies import get_agent_runtime, get_session_manager

__all__ = [
    # App
    "create_app",
    # Routes
    "chat",
    "sessions",
    "tools",
    "admin",
    # Middleware
    "RateLimitMiddleware",
    "AuthMiddleware",
    "TracingMiddleware",
    # Streaming
    "StreamingResponse",
    "EventStream",
    # Dependencies
    "get_agent_runtime",
    "get_session_manager",
]
