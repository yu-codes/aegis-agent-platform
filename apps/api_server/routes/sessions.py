"""
Session Routes

Session management endpoints.

Based on: src/api/routes/sessions.py
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()


class SessionInfo(BaseModel):
    """Session information."""

    session_id: str
    created_at: datetime
    message_count: int
    last_activity: datetime | None


class SessionListResponse(BaseModel):
    """Session list response."""

    sessions: list[SessionInfo]
    total: int


class MessageHistory(BaseModel):
    """Conversation message."""

    role: str
    content: str
    timestamp: datetime


class SessionHistoryResponse(BaseModel):
    """Session history response."""

    session_id: str
    messages: list[MessageHistory]
    metadata: dict


@router.post("/sessions", response_model=SessionInfo)
async def create_session(domain: str | None = None):
    """
    Create a new conversation session.

    Args:
        domain: Optional domain context for the session
    """
    from apps.api_server.dependencies import get_session_manager

    memory = get_session_manager()
    session_id = await memory.create_session(metadata={"domain": domain})

    return SessionInfo(
        session_id=session_id,
        created_at=datetime.utcnow(),
        message_count=0,
        last_activity=None,
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    List active sessions.

    Args:
        limit: Maximum number of sessions to return
        offset: Pagination offset
    """
    from apps.api_server.dependencies import get_session_manager

    memory = get_session_manager()
    sessions = await memory.list_sessions(limit=limit, offset=offset)

    session_infos = []
    for session in sessions:
        session_infos.append(
            SessionInfo(
                session_id=session["session_id"],
                created_at=session.get("created_at", datetime.utcnow()),
                message_count=session.get("message_count", 0),
                last_activity=session.get("last_activity"),
            )
        )

    return SessionListResponse(
        sessions=session_infos,
        total=len(session_infos),
    )


@router.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
async def get_session(session_id: str):
    """
    Get session history.

    Args:
        session_id: Session ID
    """
    from apps.api_server.dependencies import get_session_manager

    memory = get_session_manager()

    try:
        history = await memory.get_history(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = [
        MessageHistory(
            role=turn.role,
            content=turn.content,
            timestamp=turn.timestamp,
        )
        for turn in history
    ]

    return SessionHistoryResponse(
        session_id=session_id,
        messages=messages,
        metadata=await memory.get_metadata(session_id) or {},
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session.

    Args:
        session_id: Session ID
    """
    from apps.api_server.dependencies import get_session_manager

    memory = get_session_manager()

    try:
        await memory.delete_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "deleted", "session_id": session_id}


@router.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """
    Clear session history but keep session.

    Args:
        session_id: Session ID
    """
    from apps.api_server.dependencies import get_session_manager

    memory = get_session_manager()

    try:
        await memory.clear_history(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "cleared", "session_id": session_id}


@router.post("/sessions/{session_id}/summarize")
async def summarize_session(session_id: str):
    """
    Summarize session history.

    Creates a summary and optionally compresses old messages.

    Args:
        session_id: Session ID
    """
    from apps.api_server.dependencies import get_session_manager
    from services.memory import Summarizer

    memory = get_session_manager()

    try:
        history = await memory.get_history(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if not history:
        return {"status": "empty", "summary": ""}

    # Create summarizer and generate summary
    summarizer = Summarizer()
    conversation_text = "\n".join(f"{turn.role}: {turn.content}" for turn in history)

    summary = await summarizer.summarize(conversation_text)

    return {
        "status": "summarized",
        "session_id": session_id,
        "summary": summary,
        "original_turns": len(history),
    }
