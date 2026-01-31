"""
Chat API Routes

Main chat endpoint with streaming support.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import (
    get_session_manager,
    get_tool_registry,
    get_current_user,
    get_trace_context,
)
from src.api.streaming import StreamingResponse, stream_sse
from src.memory import SessionManager
from src.tools import ToolRegistry

router = APIRouter()


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    
    message: str = Field(..., min_length=1, max_length=100000)
    session_id: UUID | None = None
    
    # Options
    stream: bool = True
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    
    # Tool control
    tools: list[str] | None = None  # Specific tools to enable
    disable_tools: bool = False


class ChatResponse(BaseModel):
    """Response body for non-streaming chat."""
    
    message: str
    session_id: UUID
    
    # Metadata
    model: str | None = None
    usage: dict[str, int] | None = None
    tool_calls: list[dict[str, Any]] | None = None


@router.post("/chat")
async def chat(
    request: ChatRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    tool_registry: ToolRegistry = Depends(get_tool_registry),
    user: dict | None = Depends(get_current_user),
    trace_ctx: dict = Depends(get_trace_context),
):
    """
    Chat with the agent.
    
    Supports both streaming (SSE) and non-streaming responses.
    """
    from src.core.types import Message, ExecutionContext
    
    # Get or create session
    if request.session_id:
        session = await session_manager.get(request.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = await session_manager.create(
            user_id=user.get("id") if user else None,
        )
    
    # Build execution context
    context = ExecutionContext(
        session_id=session.id,
        user_id=user.get("id") if user else None,
        allowed_tools=set(request.tools) if request.tools else None,
        trace_id=trace_ctx.get("trace_id"),
    )
    
    # Add message to session
    user_message = Message(role="user", content=request.message)
    session.add_message(user_message)
    
    if request.stream:
        # Return streaming response
        return StreamingResponse(
            _stream_response(session, context, request, tool_registry),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming response
        response = await _generate_response(session, context, request, tool_registry)
        return ChatResponse(
            message=response["content"],
            session_id=session.id,
            model=response.get("model"),
            usage=response.get("usage"),
            tool_calls=response.get("tool_calls"),
        )


async def _stream_response(
    session,
    context,
    request: ChatRequest,
    tool_registry: ToolRegistry,
):
    """Generate streaming response."""
    # Placeholder implementation
    # In a full implementation, this would:
    # 1. Call the reasoning strategy with streaming
    # 2. Yield token events as they arrive
    # 3. Yield tool call events
    # 4. Yield the final response
    
    import json
    
    yield f"event: start\ndata: {json.dumps({'session_id': str(session.id)})}\n\n"
    
    # Simulate streaming
    response_text = f"This is a placeholder response to: {request.message}"
    
    for word in response_text.split():
        yield f"event: token\ndata: {json.dumps({'content': word + ' '})}\n\n"
    
    yield f"event: done\ndata: {json.dumps({'content': response_text})}\n\n"


async def _generate_response(
    session,
    context,
    request: ChatRequest,
    tool_registry: ToolRegistry,
) -> dict[str, Any]:
    """Generate non-streaming response."""
    # Placeholder implementation
    return {
        "content": f"This is a placeholder response to: {request.message}",
        "model": request.model or "gpt-4",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@router.post("/chat/{session_id}/regenerate")
async def regenerate_response(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager),
    user: dict | None = Depends(get_current_user),
):
    """
    Regenerate the last response.
    """
    session = await session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Remove last assistant message
    if session.messages and session.messages[-1].role == "assistant":
        session.messages.pop()
    
    # Placeholder - would regenerate response
    return {"status": "regenerated", "session_id": str(session_id)}
