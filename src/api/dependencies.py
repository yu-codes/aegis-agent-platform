"""
FastAPI Dependencies

Dependency injection for API routes.

All cross-cutting concerns are provided through these dependencies,
ensuring components are properly wired without direct coupling.
"""

from typing import Any
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request

from src.domains import DomainAwareRuntime, DomainRegistry
from src.memory import Session, SessionManager
from src.runtime import AgentRuntime
from src.tools import ToolRegistry


async def get_components(request: Request) -> dict[str, Any]:
    """Get application components from state."""
    return getattr(request.app.state, "components", {})


async def get_session_manager(
    components: dict[str, Any] = Depends(get_components),
) -> SessionManager:
    """Get session manager."""
    if "session_manager" not in components:
        raise HTTPException(status_code=503, detail="Session manager not available")
    value = components["session_manager"]
    assert isinstance(value, SessionManager)
    return value


async def get_tool_registry(
    components: dict[str, Any] = Depends(get_components),
) -> ToolRegistry:
    """Get tool registry."""
    if "tool_registry" not in components:
        raise HTTPException(status_code=503, detail="Tool registry not available")
    value = components["tool_registry"]
    assert isinstance(value, ToolRegistry)
    return value


async def get_domain_registry(
    components: dict[str, Any] = Depends(get_components),
) -> DomainRegistry:
    """Get domain registry."""
    if "domain_registry" not in components:
        raise HTTPException(status_code=503, detail="Domain registry not available")
    value = components["domain_registry"]
    assert isinstance(value, DomainRegistry)
    return value


async def get_agent_runtime(
    components: dict[str, Any] = Depends(get_components),
) -> AgentRuntime:
    """
    Get the AgentRuntime - the base orchestration point.

    For domain-aware execution, use get_domain_aware_runtime instead.
    """
    if "agent_runtime" not in components:
        raise HTTPException(status_code=503, detail="Agent runtime not available")
    value = components["agent_runtime"]
    assert isinstance(value, AgentRuntime)
    return value


async def get_domain_aware_runtime(
    components: dict[str, Any] = Depends(get_components),
) -> DomainAwareRuntime:
    """
    Get the DomainAwareRuntime - the domain-aware orchestration point.

    This is the recommended runtime for production use. It resolves
    the appropriate domain profile before execution and configures
    all components accordingly.

    Domain selection:
    - Explicit: API parameter specifies domain
    - Inferred: Lightweight classification of input
    - Context: From session/user metadata
    - Fallback: Safe default domain
    """
    if "domain_aware_runtime" not in components:
        raise HTTPException(status_code=503, detail="Domain-aware runtime not available")
    value = components["domain_aware_runtime"]
    assert isinstance(value, DomainAwareRuntime)
    return value


async def get_session(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager),
) -> Session:
    """Get a session by ID."""
    session = await session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


async def get_current_user(
    request: Request,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None),
) -> dict[str, Any] | None:
    """
    Get the current user from request.

    Returns None if not authenticated.
    """
    # Check if already validated by middleware
    if hasattr(request.state, "user"):
        user = request.state.user
        if isinstance(user, dict):
            return user
        return None

    # Return None for unauthenticated requests
    return None


async def require_user(
    user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, Any]:
    """Require authenticated user."""
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )
    return user


async def get_trace_context(request: Request) -> dict[str, str | None]:
    """Get tracing context from request."""
    return {
        "trace_id": getattr(request.state, "trace_id", None),
        "request_id": getattr(request.state, "request_id", None),
    }
