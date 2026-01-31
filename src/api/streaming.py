"""
Streaming Response Utilities

Server-Sent Events and streaming support.
"""

import asyncio
import json
from typing import Any, AsyncIterator

from starlette.responses import StreamingResponse as StarletteStreamingResponse


class EventStream:
    """
    Server-Sent Events (SSE) stream.
    
    Formats events according to SSE spec.
    """
    
    def __init__(self):
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    
    async def send(
        self,
        data: Any,
        event: str | None = None,
        id: str | None = None,
        retry: int | None = None,
    ) -> None:
        """Send an event to the stream."""
        await self._queue.put({
            "data": data,
            "event": event,
            "id": id,
            "retry": retry,
        })
    
    async def close(self) -> None:
        """Close the stream."""
        await self._queue.put(None)
    
    async def __aiter__(self) -> AsyncIterator[str]:
        """Iterate over formatted SSE events."""
        while True:
            event = await self._queue.get()
            
            if event is None:
                break
            
            yield self._format_event(event)
    
    def _format_event(self, event: dict[str, Any]) -> str:
        """Format event as SSE."""
        lines = []
        
        if event.get("event"):
            lines.append(f"event: {event['event']}")
        
        if event.get("id"):
            lines.append(f"id: {event['id']}")
        
        if event.get("retry"):
            lines.append(f"retry: {event['retry']}")
        
        # Format data
        data = event["data"]
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
        
        for line in str(data).split("\n"):
            lines.append(f"data: {line}")
        
        lines.append("")  # Empty line to end event
        
        return "\n".join(lines) + "\n"


def StreamingResponse(
    content: AsyncIterator[str],
    media_type: str = "text/event-stream",
    **kwargs: Any,
) -> StarletteStreamingResponse:
    """
    Create a streaming response for SSE.
    """
    return StarletteStreamingResponse(
        content,
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
        **kwargs,
    )


async def stream_json_lines(
    generator: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[str]:
    """
    Stream JSON objects as newline-delimited JSON.
    
    Useful for streaming structured responses.
    """
    async for item in generator:
        yield json.dumps(item) + "\n"


async def stream_sse(
    generator: AsyncIterator[dict[str, Any]],
    event_type: str = "message",
) -> AsyncIterator[str]:
    """
    Stream events as SSE format.
    """
    event_id = 0
    
    async for item in generator:
        event_id += 1
        
        lines = [
            f"id: {event_id}",
            f"event: {event_type}",
            f"data: {json.dumps(item)}",
            "",
        ]
        
        yield "\n".join(lines) + "\n"
    
    # Send done event
    yield "event: done\ndata: {}\n\n"
