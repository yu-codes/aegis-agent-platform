"""
Chat Routes

Chat/conversation endpoints.

Based on: src/api/routes/chat.py
"""

import asyncio
import json
from typing import AsyncGenerator, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str = Field(description="Message role: user, assistant, system")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(description="User message")
    session_id: str | None = Field(
        default=None, description="Session ID for conversation continuity"
    )
    stream: bool = Field(default=False, description="Enable streaming response")
    model: str | None = Field(default=None, description="Override model selection")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Response temperature")
    max_tokens: int = Field(default=4096, ge=1, le=32768, description="Max response tokens")
    tools_enabled: bool = Field(default=True, description="Enable tool usage")
    domain: str | None = Field(default=None, description="Domain context")


class ChatResponse(BaseModel):
    """Chat response model."""

    message: str = Field(description="Assistant response")
    session_id: str = Field(description="Session ID")
    tools_used: list[str] = Field(default_factory=list, description="Tools used")
    metadata: dict = Field(default_factory=dict, description="Response metadata")


class FeedbackRequest(BaseModel):
    """Feedback request model."""

    session_id: str = Field(description="Session ID")
    message_index: int = Field(description="Message index in conversation")
    rating: int = Field(ge=1, le=5, description="Rating from 1 to 5")
    feedback_text: str | None = Field(default=None, description="Optional feedback text")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
):
    """
    Send a chat message and receive a response.

    Supports both synchronous and streaming responses.
    """
    from apps.api_server.dependencies import (
        get_agent_orchestrator,
        get_audit_log,
    )
    from uuid import uuid4

    orchestrator = get_agent_orchestrator()
    audit_log = get_audit_log()

    # Get or create session
    session_id = request.session_id or str(uuid4())

    # Get user ID from request state
    user_id = getattr(http_request.state, "user_id", None)

    if request.stream:
        return StreamingResponse(
            _stream_response(
                orchestrator=orchestrator,
                session_id=session_id,
                message=request.message,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools_enabled=request.tools_enabled,
                domain=request.domain,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response
    try:
        # Create execution context
        from services.agent_core.orchestrator import ExecutionContext
        from uuid import UUID

        context = ExecutionContext(
            session_id=UUID(session_id) if len(session_id) == 36 else uuid4(),
            user_id=user_id,
        )

        # Run agent
        result = await orchestrator.run(
            query=request.message,
            context=context,
            system_prompt=None,
        )

        # Audit log
        await audit_log.log(
            event_type="agent.run",
            action="chat",
            session_id=session_id,
            details={
                "input": request.message[:100],
                "success": result.success,
            },
        )

        return ChatResponse(
            message=result.response or "No response generated",
            session_id=session_id,
            tools_used=result.tools_used,
            metadata={
                "model": result.metadata.get("model", "stub"),
                "tokens_used": result.tokens_used,
                "duration_ms": int(result.duration * 1000) if result.duration else 0,
            },
        )

    except Exception as e:
        await audit_log.log(
            event_type="agent.error",
            action="chat",
            session_id=session_id,
            details={
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(
    orchestrator,
    session_id: str,
    message: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    tools_enabled: bool,
    domain: str | None,
) -> AsyncGenerator[str, None]:
    """Generate streaming response."""
    try:
        full_response = ""
        tools_used = []

        # Simplified streaming - just yield stub response for now
        stub_response = f"This is a streaming response for: {message}"
        for i in range(0, len(stub_response), 10):
            chunk = stub_response[i : i + 10]
            full_response += chunk
            yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
            await asyncio.sleep(0.05)  # Simulate streaming delay

        # Send done event
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'tools_used': tools_used})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@router.post("/chat/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a chat message."""
    from apps.api_server.dependencies import get_audit_log

    audit_log = get_audit_log()

    await audit_log.log(
        event_type="feedback.submitted",
        action="submit_feedback",
        session_id=request.session_id,
        details={
            "message_index": request.message_index,
            "rating": request.rating,
            "feedback_text": request.feedback_text,
        },
    )

    return {"status": "received"}
