"""
Tool Management Routes
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_tool_registry, get_current_user
from src.tools import ToolRegistry, ToolCategory

router = APIRouter()


class ToolInfo(BaseModel):
    """Tool information."""
    
    name: str
    description: str
    category: str
    version: str
    parameters: dict[str, Any]
    is_dangerous: bool = False


class ToolListResponse(BaseModel):
    """List of tools."""
    
    tools: list[ToolInfo]
    total: int


@router.get("/tools", response_model=ToolListResponse)
async def list_tools(
    category: str | None = None,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
    user: dict | None = Depends(get_current_user),
):
    """List available tools."""
    # Get tools, optionally filtered by category
    if category:
        try:
            cat = ToolCategory(category)
            tools = tool_registry.list_tools(category=cat)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
    else:
        tools = tool_registry.get_all_definitions()
    
    return ToolListResponse(
        tools=[
            ToolInfo(
                name=t.name,
                description=t.description,
                category=t.category.value,
                version=t.version,
                parameters=t.parameters,
                is_dangerous=t.is_dangerous,
            )
            for t in tools
        ],
        total=len(tools),
    )


@router.get("/tools/{tool_name}", response_model=ToolInfo)
async def get_tool(
    tool_name: str,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
):
    """Get tool details."""
    tool = tool_registry.get(tool_name)
    
    if tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    return ToolInfo(
        name=tool.name,
        description=tool.description,
        category=tool.category.value,
        version=tool.version,
        parameters=tool.parameters,
        is_dangerous=tool.is_dangerous,
    )


class ExecuteToolRequest(BaseModel):
    """Request to execute a tool."""
    
    arguments: dict[str, Any] = {}


@router.post("/tools/{tool_name}/execute")
async def execute_tool(
    tool_name: str,
    request: ExecuteToolRequest,
    tool_registry: ToolRegistry = Depends(get_tool_registry),
    user: dict | None = Depends(get_current_user),
):
    """
    Execute a tool directly.
    
    Note: This is for testing/admin purposes.
    Normal tool execution happens through chat.
    """
    from src.core.types import ExecutionContext
    from src.tools.executor import ToolExecutor
    
    tool = tool_registry.get(tool_name)
    if tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    # Create context
    context = ExecutionContext(
        user_id=user.get("id") if user else None,
    )
    
    # Execute
    executor = ToolExecutor(tool_registry)
    result = await executor.execute(tool_name, request.arguments, context)
    
    if result.error:
        return {
            "success": False,
            "error": result.error,
            "duration_ms": result.duration_ms,
        }
    
    return {
        "success": True,
        "result": result.result,
        "duration_ms": result.duration_ms,
    }


@router.get("/tools/categories")
async def list_categories():
    """List available tool categories."""
    return {
        "categories": [cat.value for cat in ToolCategory],
    }
