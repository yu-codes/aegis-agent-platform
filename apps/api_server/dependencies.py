"""
Dependency Injection

FastAPI dependencies for the API.

Based on: src/api/dependencies.py
"""

import os
from functools import lru_cache
from typing import Any

# Global instances
_orchestrator = None
_session_manager = None
_tool_registry = None
_rag_retriever = None
_metrics = None
_tracer = None
_audit_log = None


async def init_dependencies() -> None:
    """Initialize all dependencies."""
    global _orchestrator, _session_manager, _tool_registry
    global _rag_retriever, _metrics, _tracer, _audit_log

    # Initialize observability first
    from services.observability import Tracer, MetricsCollector, AuditLog

    _tracer = Tracer(service_name="aegis-api")
    _metrics = MetricsCollector(namespace="aegis")
    _audit_log = AuditLog(enabled=True)

    # Initialize tool registry
    from services.tools import ToolRegistry

    _tool_registry = ToolRegistry()

    # Register built-in tools
    await _register_builtin_tools()

    # Initialize session manager
    from services.memory import SessionManager

    _session_manager = SessionManager(
        max_turns=int(os.getenv("MAX_CONVERSATION_TURNS", "50")),
        max_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "8000")),
    )

    # Initialize RAG retriever
    from services.rag import Retriever

    _rag_retriever = Retriever()  # Uses in-memory fallback

    # Initialize LLM adapter
    from services.reasoning.model_adapters.stub_adapter import StubAdapter

    llm_adapter = StubAdapter()

    # Initialize orchestrator
    from services.agent_core import AgentOrchestrator

    _orchestrator = AgentOrchestrator(
        llm=llm_adapter,
        memory=_session_manager,
        retriever=_rag_retriever,
    )


async def _register_builtin_tools() -> None:
    """Register built-in tools."""
    from services.tools.builtins import (
        WebSearchTool,
        WebFetchTool,
        CodeExecutorTool,
        CodeAnalyzerTool,
        FileReaderTool,
        FileWriterTool,
        CalculatorTool,
    )

    tools = [
        WebSearchTool(),
        WebFetchTool(),
        CodeAnalyzerTool(),
        CalculatorTool(),
    ]

    # Only register file tools if explicitly enabled
    if os.getenv("ENABLE_FILE_TOOLS", "false").lower() == "true":
        tools.extend([FileReaderTool(), FileWriterTool()])

    # Only register code executor if explicitly enabled
    if os.getenv("ENABLE_CODE_EXECUTOR", "false").lower() == "true":
        tools.append(CodeExecutorTool())

    for tool in tools:
        _tool_registry.register_instance(tool)


async def cleanup_dependencies() -> None:
    """Cleanup all dependencies."""
    global _orchestrator, _session_manager, _tool_registry
    global _rag_retriever, _metrics, _tracer, _audit_log

    # Cleanup
    _orchestrator = None
    _session_manager = None
    _tool_registry = None
    _rag_retriever = None
    _metrics = None
    _tracer = None
    _audit_log = None


def get_agent_orchestrator():
    """Get agent orchestrator dependency."""
    if _orchestrator is None:
        raise RuntimeError("Dependencies not initialized")
    return _orchestrator


def get_session_manager():
    """Get session manager dependency."""
    if _session_manager is None:
        raise RuntimeError("Dependencies not initialized")
    return _session_manager


def get_tool_registry():
    """Get tool registry dependency."""
    if _tool_registry is None:
        raise RuntimeError("Dependencies not initialized")
    return _tool_registry


def get_rag_retriever():
    """Get RAG retriever dependency."""
    if _rag_retriever is None:
        raise RuntimeError("Dependencies not initialized")
    return _rag_retriever


def get_metrics():
    """Get metrics collector dependency."""
    if _metrics is None:
        raise RuntimeError("Dependencies not initialized")
    return _metrics


def get_tracer():
    """Get tracer dependency."""
    if _tracer is None:
        raise RuntimeError("Dependencies not initialized")
    return _tracer


def get_audit_log():
    """Get audit log dependency."""
    if _audit_log is None:
        raise RuntimeError("Dependencies not initialized")
    return _audit_log


@lru_cache()
def get_settings():
    """Get application settings."""
    return {
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "max_concurrent_sessions": int(os.getenv("MAX_CONCURRENT_SESSIONS", "100")),
        "default_model": os.getenv("DEFAULT_MODEL", "gpt-4o"),
    }
