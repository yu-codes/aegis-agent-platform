"""
API Server Application

FastAPI-based API server.

Components:
- app: Application factory
- routes: API routes
- middleware: Custom middleware
- dependencies: Dependency injection
"""

from apps.api_server.app import create_app, get_app
from apps.api_server.dependencies import (
    get_agent_orchestrator,
    get_session_manager,
    get_tool_registry,
    get_rag_retriever,
    get_metrics,
    get_tracer,
)

__all__ = [
    "create_app",
    "get_app",
    "get_agent_orchestrator",
    "get_session_manager",
    "get_tool_registry",
    "get_rag_retriever",
    "get_metrics",
    "get_tracer",
]
