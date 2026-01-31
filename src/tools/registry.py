"""
Tool Registry

Schema-driven registration and discovery of tools.
Tools are first-class citizens with full metadata.

Design decisions:
- Decorator-based registration for convenience
- JSON Schema for argument validation
- Versioning support
- Category/tag organization
"""

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, get_type_hints


class ToolCategory(str, Enum):
    """Categories for organizing tools."""

    SEARCH = "search"
    COMPUTE = "compute"
    DATA = "data"
    COMMUNICATION = "communication"
    FILE = "file"
    WEB = "web"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class ToolDefinition:
    """
    Complete definition of a tool.

    Contains all metadata needed for:
    - LLM to understand and use the tool
    - Executor to validate and run it
    - UI to display it
    """

    name: str
    description: str
    function: Callable[..., Awaitable[Any]]

    # Schema
    parameters: dict[str, Any]  # JSON Schema for arguments
    return_type: str = "string"

    # Metadata
    version: str = "1.0.0"
    category: ToolCategory = ToolCategory.CUSTOM
    tags: list[str] = field(default_factory=list)

    # Execution hints
    is_async: bool = True
    timeout_seconds: float = 30.0
    requires_confirmation: bool = False

    # Safety
    is_dangerous: bool = False  # Requires elevated permissions
    allowed_scopes: list[str] = field(default_factory=lambda: ["*"])

    # Documentation
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


def _extract_schema_from_function(func: Callable) -> dict[str, Any]:
    """
    Extract JSON Schema from function signature.

    Uses type hints and docstrings to build the schema.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "context"):
            continue

        param_type = hints.get(param_name, str)
        param_schema = _type_to_schema(param_type)

        # Add description from docstring if available
        properties[param_name] = param_schema

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _type_to_schema(python_type: type) -> dict[str, Any]:
    """Convert Python type to JSON Schema."""
    import typing

    # Handle Optional
    origin = getattr(python_type, "__origin__", None)

    if origin is typing.Union:
        args = python_type.__args__
        if type(None) in args:
            # Optional type
            non_none = next(a for a in args if a is not type(None))
            schema = _type_to_schema(non_none)
            return schema

    if origin is list:
        item_type = python_type.__args__[0] if python_type.__args__ else str
        return {"type": "array", "items": _type_to_schema(item_type)}

    if origin is dict:
        return {"type": "object"}

    # Basic types
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    return type_map.get(python_type, {"type": "string"})


def tool(
    name: str | None = None,
    description: str | None = None,
    category: ToolCategory = ToolCategory.CUSTOM,
    tags: list[str] | None = None,
    requires_confirmation: bool = False,
    is_dangerous: bool = False,
    timeout: float = 30.0,
    version: str = "1.0.0",
) -> Callable:
    """
    Decorator to register a function as a tool.

    Usage:
        @tool(name="search", description="Search the web")
        async def search(query: str) -> str:
            ...
    """

    def decorator(func: Callable) -> Callable:
        # Extract metadata
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Execute {tool_name}"

        # Build schema from signature
        schema = _extract_schema_from_function(func)

        # Create definition
        definition = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            function=func,
            parameters=schema,
            category=category,
            tags=tags or [],
            requires_confirmation=requires_confirmation,
            is_dangerous=is_dangerous,
            timeout_seconds=timeout,
            version=version,
            is_async=inspect.iscoroutinefunction(func),
        )

        # Attach definition to function
        func._tool_definition = definition

        @wraps(func)
        async def wrapper(*args, **kwargs):
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        wrapper._tool_definition = definition
        return wrapper

    return decorator


class ToolRegistry:
    """
    Central registry for all tools.

    Provides:
    - Tool registration and discovery
    - Schema retrieval for LLM
    - Category-based filtering
    - Version management
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._categories: dict[ToolCategory, set[str]] = {cat: set() for cat in ToolCategory}

    def register(self, definition: ToolDefinition) -> None:
        """Register a tool definition."""
        self._tools[definition.name] = definition
        self._categories[definition.category].add(definition.name)

    def register_function(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> ToolDefinition:
        """
        Register a function as a tool.

        Alternative to using the @tool decorator.
        """
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Execute {tool_name}"
        schema = _extract_schema_from_function(func)

        definition = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            function=func,
            parameters=schema,
            is_async=inspect.iscoroutinefunction(func),
            **kwargs,
        )

        self.register(definition)
        return definition

    def register_decorated(self, func: Callable) -> None:
        """Register a function that was decorated with @tool."""
        if hasattr(func, "_tool_definition"):
            self.register(func._tool_definition)
        else:
            raise ValueError(f"Function {func.__name__} is not decorated with @tool")

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(
        self,
        category: ToolCategory | None = None,
        tags: list[str] | None = None,
    ) -> list[ToolDefinition]:
        """List tools, optionally filtered."""
        tools = list(self._tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        if tags:
            tag_set = set(tags)
            tools = [t for t in tools if tag_set & set(t.tags)]

        return tools

    def get_all_definitions(self) -> list[ToolDefinition]:
        """Get all registered tool definitions."""
        return list(self._tools.values())

    def get_schemas_for_llm(
        self,
        format: str = "openai",
        tool_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get tool schemas in LLM-specific format.

        Args:
            format: "openai" or "anthropic"
            tool_names: Optional list of tools to include
        """
        tools = self._tools.values()

        if tool_names:
            tools = [t for t in tools if t.name in tool_names]

        if format == "openai":
            return [t.to_openai_format() for t in tools]
        elif format == "anthropic":
            return [t.to_anthropic_format() for t in tools]
        else:
            raise ValueError(f"Unknown format: {format}")

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            tool = self._tools.pop(name)
            self._categories[tool.category].discard(name)
            return True
        return False


# Default global registry
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
