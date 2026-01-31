"""
FastAPI Dependencies

Dependency injection for API routes.

All cross-cutting concerns are provided through these dependencies,
ensuring components are properly wired without direct coupling.
"""

from typing import Any
from uuid import UUID

from fastapi import Depends, HTTPException, Request, Header

from src.memory import SessionManager, Session
from src.tools import ToolRegistry
from src.runtime import AgentRuntime


async def get_components(request: Request) -> dict[str, Any]:
    """Get application components from state."""
    return getattr(request.app.state, "components", {})


async def get_session_manager(
    components: dict[str, Any] = Depends(get_components),
) -> SessionManager:
    """Get session manager."""
    if "session_manager" not in components:
        raise HTTPException(status_code=503, detail="Session manager not available")
    return components["session_manager"]


async def get_tool_registry(
    components: dict[str, Any] = Depends(get_components),
) -> ToolRegistry:
    """Get tool registry."""
    if "tool_registry" not in components:
        raise HTTPException(status_code=503, detail="Tool registry not available")
    return components["tool_registry"]


async def get_agent_runtime(
    components: dict[str, Any] = Depends(get_components),
) -> AgentRuntime:
    """
    Get the AgentRuntime - the single orchestration point for agent execution.
    
    This is the canonical way to get an agent instance. All agent execution
    must go through the runtime.
    """
    if "agent_runtime" not in components:
        raise HTTPException(status_code=503, detail="Agent runtime not available")
    return components["agent_runtime"]


async def get_session(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager),
) -> Session:
    """Get a session by ID."""
    session = await session_manager.get(session_id)
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
        return request.state.user
    
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


async def get_trace_context(request: Request) -> dict[str, str]:
    """Get tracing context from request."""
    return {
        "trace_id": getattr(request.state, "trace_id", None),
        "request_id": getattr(request.state, "request_id", None),
    }
    pass
