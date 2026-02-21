"""
Tools Routes

Tool management endpoints.

Based on: src/api/routes/tools.py
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


class ToolInfo(BaseModel):
    """Tool information."""

    name: str
    description: str
    parameters: dict
    permission_level: str


class ToolListResponse(BaseModel):
    """Tool list response."""

    tools: list[ToolInfo]
    total: int


class ToolCallRequest(BaseModel):
    """Tool call request."""

    tool_name: str = Field(description="Name of the tool to call")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")
    session_id: str | None = Field(default=None, description="Session context")


class ToolCallResponse(BaseModel):
    """Tool call response."""

    tool_name: str
    result: Any
    success: bool
    error: str | None = None
    execution_time_ms: float


@router.get("/tools", response_model=ToolListResponse)
async def list_tools():
    """
    List available tools.

    Returns all registered tools and their schemas.
    """
    from apps.api_server.dependencies import get_tool_registry

    registry = get_tool_registry()
    tools = registry.list_tools()

    tool_infos = []
    for tool in tools:
        # Convert parameters to dict format
        params_dict = {}
        for param in tool.parameters:
            params_dict[param.name] = {
                "type": param.type,
                "description": param.description,
                "required": param.required,
            }

        tool_infos.append(
            ToolInfo(
                name=tool.name,
                description=tool.description,
                parameters=params_dict,
                permission_level=(
                    tool.permission_level.value
                    if hasattr(tool.permission_level, "value")
                    else tool.permission_level
                ),
            )
        )

    return ToolListResponse(tools=tool_infos, total=len(tool_infos))


@router.get("/tools/{tool_name}")
async def get_tool(tool_name: str):
    """
    Get tool details.

    Args:
        tool_name: Name of the tool
    """
    from apps.api_server.dependencies import get_tool_registry

    registry = get_tool_registry()
    tool = registry.get(tool_name)

    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "schema_openai": registry.get_openai_schema(tool_name),
        "schema_anthropic": registry.get_anthropic_schema(tool_name),
    }


@router.post("/tools/call", response_model=ToolCallResponse)
async def call_tool(
    request: ToolCallRequest,
    http_request: Request,
):
    """
    Call a tool directly.

    Executes a tool with provided arguments.
    """
    import time
    from apps.api_server.dependencies import get_tool_registry, get_audit_log
    from services.tools import ToolExecutor, ToolValidator

    registry = get_tool_registry()
    audit_log = get_audit_log()

    tool = registry.get(request.tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{request.tool_name}' not found")

    # Validate arguments
    validator = ToolValidator()
    validation_result = validator.validate_input(
        request.tool_name, request.arguments, tool.parameters
    )

    if not validation_result.valid:
        raise HTTPException(status_code=400, detail=validation_result.error)

    # Execute tool
    executor = ToolExecutor(registry=registry)

    start_time = time.time()

    try:
        result = await executor.execute(
            tool_name=request.tool_name,
            arguments=request.arguments,
        )

        execution_time = (time.time() - start_time) * 1000

        # Audit log
        user_id = getattr(http_request.state, "user_id", None)
        await audit_log.log_tool_call(
            tool_name=request.tool_name,
            user_id=user_id,
            session_id=request.session_id,
            arguments=request.arguments,
            result=result.result,
            success=result.success,
        )

        return ToolCallResponse(
            tool_name=request.tool_name,
            result=result.result,
            success=result.success,
            error=result.error,
            execution_time_ms=execution_time,
        )

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000

        return ToolCallResponse(
            tool_name=request.tool_name,
            result=None,
            success=False,
            error=str(e),
            execution_time_ms=execution_time,
        )


@router.post("/tools/register")
async def register_tool(
    name: str,
    description: str,
    parameters: dict,
    endpoint: str,
):
    """
    Register a new tool dynamically.

    Note: This creates a tool that calls an external endpoint.
    """
    from apps.api_server.dependencies import get_tool_registry

    registry = get_tool_registry()

    # Create HTTP tool wrapper
    async def http_tool(**kwargs):
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=kwargs, timeout=30)
            return response.json()

    registry.register(
        name=name,
        description=description,
        parameters=parameters,
        func=http_tool,
    )

    return {"status": "registered", "name": name}
