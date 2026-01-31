"""
Session Management Routes
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_session_manager, get_current_user
from src.memory import SessionManager

router = APIRouter()


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    
    metadata: dict[str, Any] | None = None
    system_prompt: str | None = None


class SessionResponse(BaseModel):
    """Session information."""
    
    id: UUID
    user_id: str | None = None
    created_at: str
    message_count: int
    metadata: dict[str, Any] = {}


class SessionListResponse(BaseModel):
    """List of sessions."""
    
    sessions: list[SessionResponse]
    total: int


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest | None = None,
    session_manager: SessionManager = Depends(get_session_manager),
    user: dict | None = Depends(get_current_user),
):
    """Create a new chat session."""
    session = await session_manager.create_session(
        user_id=user.get("id") if user else None,
        metadata=request.metadata if request else None,
    )
    
    return SessionResponse(
        id=session.id,
        user_id=session.user_id,
        created_at=session.created_at.isoformat(),
        message_count=len(session.messages),
        metadata=session.metadata,
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 20,
    offset: int = 0,
    session_manager: SessionManager = Depends(get_session_manager),
    user: dict | None = Depends(get_current_user),
):
    """List user's sessions."""
    user_id = user.get("id") if user else None
    
    # Get session IDs (offset is handled manually since backend doesn't support it)
    session_ids = await session_manager.list_sessions(
        user_id=user_id,
        limit=limit + offset,  # Get enough to handle offset
    )
    
    # Apply offset manually
    session_ids = session_ids[offset:offset + limit]
    
    # Fetch full session objects
    sessions = []
    for sid in session_ids:
        session = await session_manager.get_session(sid)
        if session:
            sessions.append(session)
    
    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=s.id,
                user_id=s.user_id,
                created_at=s.created_at.isoformat(),
                message_count=len(s.messages),
                metadata=s.metadata,
            )
            for s in sessions
        ],
        total=len(sessions),
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager),
    user: dict | None = Depends(get_current_user),
):
    """Get session details."""
    session = await session_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check ownership
    if user and session.user_id and session.user_id != user.get("id"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return SessionResponse(
        id=session.id,
        user_id=session.user_id,
        created_at=session.created_at.isoformat(),
        message_count=len(session.messages),
        metadata=session.metadata,
    )


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: UUID,
    limit: int = 50,
    session_manager: SessionManager = Depends(get_session_manager),
    user: dict | None = Depends(get_current_user),
):
    """Get messages from a session."""
    session = await session_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session.messages[-limit:]
    
    return {
        "messages": [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            for m in messages
        ],
        "total": len(session.messages),
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager),
    user: dict | None = Depends(get_current_user),
):
    """Delete a session."""
    session = await session_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check ownership
    if user and session.user_id and session.user_id != user.get("id"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    await session_manager.delete_session(session_id)
    
    return {"status": "deleted", "session_id": str(session_id)}
