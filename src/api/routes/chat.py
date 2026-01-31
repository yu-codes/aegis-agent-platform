"""
Chat API Routes

Main chat endpoint with streaming support.
Uses DomainAwareRuntime for domain-specific agent behavior.
"""

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import (
    get_current_user,
    get_domain_aware_runtime,
    get_session_manager,
    get_trace_context,
)
from src.api.streaming import StreamingResponse
from starlette.responses import StreamingResponse as StarletteStreamingResponse
from src.core.types import Message
from src.domains import DomainAwareRuntime
from src.memory import SessionManager
from src.runtime import AgentEventType

router = APIRouter()


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(..., min_length=1, max_length=100000)
    session_id: UUID | None = None

    # Domain configuration
    domain: str | None = Field(
        default=None,
        description="Explicit domain to use (e.g., 'technical_support', 'financial_analysis')",
    )

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

    # Domain info
    domain: str | None = None
    domain_resolution_method: str | None = None

    # Metadata
    model: str | None = None
    usage: dict[str, int] | None = None
    tool_calls: list[dict[str, Any]] | None = None

    # Debug info (optional)
    execution_id: str | None = None
    trace_id: str | None = None


@router.post("/chat")
async def chat(
    request: ChatRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    runtime: DomainAwareRuntime = Depends(get_domain_aware_runtime),
    user: dict[str, Any] | None = Depends(get_current_user),
    trace_ctx: dict[str, str | None] = Depends(get_trace_context),
) -> StarletteStreamingResponse | ChatResponse:
    """
    Chat with the agent.

    Supports both streaming (SSE) and non-streaming responses.
    All execution flows through DomainAwareRuntime for domain-specific behavior.

    Domain selection:
    - Explicit: Pass `domain` parameter
    - Inferred: Auto-detected from message content
    - Fallback: Uses default general domain
    """
    from src.core.types import ExecutionContext, Message, MessageRole

    # Get or create session
    if request.session_id:
        session = await session_manager.get_session(request.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = await session_manager.create_session(
            user_id=user.get("id") if user else None,
        )

    # Build execution context
    context = ExecutionContext(
        session_id=session.id,
        user_id=user.get("id") if user else None,
        allowed_tools=set(request.tools) if request.tools else None,
        trace_id=trace_ctx.get("trace_id"),
        enable_streaming=request.stream,
    )

    # Get conversation history from session
    history = list(session.messages) if session.messages else None

    if request.stream:
        # Return streaming response via DomainAwareRuntime
        return StreamingResponse(
            _stream_response(
                runtime, request.message, context, history, session, session_manager, request.domain
            ),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming response via DomainAwareRuntime
        result = await runtime.run(
            message=request.message,
            context=context,
            history=history,
            domain=request.domain,
        )

        # Persist messages to session
        session.add_message(Message(role=MessageRole.USER, content=request.message))
        session.add_message(Message(role=MessageRole.ASSISTANT, content=result.content))
        await session_manager.update_session(session)

        # Get domain resolution info
        domain_info = await runtime.resolve_domain(
            explicit_domain=request.domain,
            content=request.message,
        )

        return ChatResponse(
            message=result.content,
            session_id=session.id,
            domain=domain_info.profile.name,
            domain_resolution_method=domain_info.method.value,
            model=result.model,
            usage=(
                {
                    "total_tokens": result.total_tokens,
                    "tool_calls": result.tool_calls_count,
                }
                if result.total_tokens
                else None
            ),
            tool_calls=(
                [{"name": tr.name, "result": tr.result} for tr in result.tool_results]
                if result.tool_results
                else None
            ),
            execution_id=str(result.execution_id),
            trace_id=result.trace_id,
        )


async def _stream_response(
    runtime: DomainAwareRuntime,
    message: str,
    context: Any,
    history: list[Message] | None,
    session: Any,
    session_manager: SessionManager,
    domain: str | None = None,
) -> AsyncIterator[str]:
    """Generate streaming response via DomainAwareRuntime."""
    import json

    from src.core.types import Message, MessageRole
    from src.domains import DomainEventType

    yield f"event: start\ndata: {json.dumps({'session_id': str(session.id)})}\n\n"

    final_content = ""

    async for event in runtime.run_stream(message, context, history, domain=domain):
        # Handle domain resolution event
        if event.data.get("event_type") == DomainEventType.DOMAIN_RESOLVED:
            yield f"event: domain\ndata: {json.dumps(event.data)}\n\n"

        elif event.type == AgentEventType.CONTENT_CHUNK:
            chunk = event.data.get("content", "")
            final_content += chunk
            yield f"event: token\ndata: {json.dumps({'content': chunk})}\n\n"

        elif event.type == AgentEventType.TOOL_CALL_STARTED:
            yield f"event: tool_start\ndata: {json.dumps(event.data)}\n\n"

        elif event.type == AgentEventType.TOOL_CALL_COMPLETED:
            yield f"event: tool_end\ndata: {json.dumps(event.data)}\n\n"

        elif event.type == AgentEventType.RUN_COMPLETED:
            final_content = event.data.get("content", final_content)

            # Persist messages
            session.add_message(Message(role=MessageRole.USER, content=message))
            session.add_message(Message(role=MessageRole.ASSISTANT, content=final_content))
            await session_manager.update_session(session)

            yield f"event: done\ndata: {json.dumps({'content': final_content, **event.data})}\n\n"

        elif event.type == AgentEventType.RUN_FAILED:
            yield f"event: error\ndata: {json.dumps(event.data)}\n\n"

        elif event.type == AgentEventType.OUTPUT_BLOCKED:
            yield f"event: blocked\ndata: {json.dumps(event.data)}\n\n"


@router.post("/chat/{session_id}/regenerate")
async def regenerate_response(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager),
    user: dict[str, Any] | None = Depends(get_current_user),
) -> dict[str, str]:
    """
    Regenerate the last response.
    """
    session = await session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Remove last assistant message
    if session.messages and session.messages[-1].role == "assistant":
        session.messages.pop()

    # Placeholder - would regenerate response
    return {"status": "regenerated", "session_id": str(session_id)}
