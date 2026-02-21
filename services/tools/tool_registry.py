"""
Tool Registry

Tool registration and discovery.

Based on: src/tools/registry.py
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from functools import wraps
import inspect
import json


@dataclass
class ToolParameter:
    """A tool parameter definition."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    """A tool definition."""

    name: str
    description: str = ""
    parameters: list[ToolParameter] = field(default_factory=list)

    # Handler
    handler: Callable | None = None
    is_async: bool = False

    # Permissions
    requires_permission: bool = False
    permission_level: str = "user"  # user, admin, system

    # Rate limiting
    rate_limit: int | None = None  # calls per minute

    # Metadata
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    version: str = "1.0.0"

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function schema."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict:
        """Convert to Anthropic tool schema."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """
    Tool registration and discovery.

    Manages available tools and their definitions.
    """

    _instance: "ToolRegistry | None" = None

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._categories: dict[str, list[str]] = {}

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        parameters: list[ToolParameter] | None = None,
        category: str = "general",
        permission_level: str = "user",
        rate_limit: int | None = None,
        tags: list[str] | None = None,
    ) -> ToolDefinition:
        """
        Register a tool.

        Args:
            name: Tool name
            handler: Handler function
            description: Tool description
            parameters: Parameter definitions
            category: Tool category
            permission_level: Required permission level
            rate_limit: Calls per minute limit
            tags: Tool tags

        Returns:
            Tool definition
        """
        # Auto-extract parameters if not provided
        if parameters is None:
            parameters = self._extract_parameters(handler)

        # Auto-extract description from docstring
        if not description and handler.__doc__:
            description = handler.__doc__.strip().split("\n")[0]

        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            is_async=inspect.iscoroutinefunction(handler),
            requires_permission=permission_level != "user",
            permission_level=permission_level,
            rate_limit=rate_limit,
            category=category,
            tags=tags or [],
        )

        self._tools[name] = tool_def

        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

        return tool_def

    def register_instance(
        self,
        tool_instance: object,
        name: str | None = None,
        category: str = "general",
        permission_level: str = "user",
    ) -> ToolDefinition | None:
        """
        Register a tool from a class instance.

        Looks for callable methods (excluding dunder methods) and registers them.

        Args:
            tool_instance: Tool class instance
            name: Optional name override (defaults to class name)
            category: Tool category
            permission_level: Required permission level

        Returns:
            Tool definition for the primary method, or None
        """
        cls_name = tool_instance.__class__.__name__
        base_name = name or cls_name.lower().replace("tool", "")

        # Get docstring from class
        description = tool_instance.__class__.__doc__ or ""
        description = description.strip().split("\n")[0] if description else ""

        # Look for primary method (usually the first public async or sync method)
        primary_method = None
        for attr_name in dir(tool_instance):
            if attr_name.startswith("_"):
                continue
            attr = getattr(tool_instance, attr_name)
            if callable(attr) and not attr_name.startswith("_"):
                # Prefer common names like search, execute, run, call, fetch, read, write
                if attr_name in (
                    "search",
                    "execute",
                    "run",
                    "call",
                    "fetch",
                    "read",
                    "write",
                    "analyze",
                    "calculate",
                ):
                    primary_method = attr
                    break
                elif primary_method is None:
                    primary_method = attr

        if primary_method is None:
            return None

        return self.register(
            name=base_name,
            handler=primary_method,
            description=description,
            category=category,
            permission_level=permission_level,
        )

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            tool = self._tools[name]
            if tool.category in self._categories:
                self._categories[tool.category] = [
                    t for t in self._categories[tool.category] if t != name
                ]
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(
        self,
        category: str | None = None,
        tags: list[str] | None = None,
        permission_level: str | None = None,
    ) -> list[ToolDefinition]:
        """
        List tools with optional filters.

        Args:
            category: Filter by category
            tags: Filter by tags (any match)
            permission_level: Filter by max permission level

        Returns:
            List of matching tools
        """
        tools = list(self._tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]

        if permission_level:
            level_order = ["user", "admin", "system"]
            max_idx = level_order.index(permission_level) if permission_level in level_order else 0
            tools = [t for t in tools if level_order.index(t.permission_level) <= max_idx]

        return tools

    def get_categories(self) -> list[str]:
        """Get all tool categories."""
        return list(self._categories.keys())

    def get_schemas(
        self,
        format: str = "openai",
        tools: list[str] | None = None,
    ) -> list[dict]:
        """
        Get tool schemas.

        Args:
            format: Schema format (openai, anthropic)
            tools: Specific tools to include

        Returns:
            List of tool schemas
        """
        tool_list = (
            [self._tools[t] for t in tools if t in self._tools]
            if tools
            else list(self._tools.values())
        )

        if format == "anthropic":
            return [t.to_anthropic_schema() for t in tool_list]
        return [t.to_openai_schema() for t in tool_list]

    def _extract_parameters(self, handler: Callable) -> list[ToolParameter]:
        """Extract parameters from function signature."""
        sig = inspect.signature(handler)
        params = []

        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue

            # Determine type
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object",
                }
                param_type = type_map.get(param.annotation, "string")

            # Determine if required
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default

            params.append(
                ToolParameter(
                    name=name,
                    type=param_type,
                    required=required,
                    default=default,
                )
            )

        return params

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[ToolDefinition]:
        return iter(self._tools.values())


# Global registry instance
_registry = ToolRegistry.get_instance()


def tool(
    name: str | None = None,
    description: str = "",
    category: str = "general",
    permission_level: str = "user",
    rate_limit: int | None = None,
    tags: list[str] | None = None,
) -> Callable:
    """
    Decorator to register a function as a tool.

    Usage:
        @tool(name="search", category="retrieval")
        def search_documents(query: str) -> list[dict]:
            '''Search for documents.'''
            ...
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        _registry.register(
            name=tool_name,
            handler=func,
            description=description,
            category=category,
            permission_level=permission_level,
            rate_limit=rate_limit,
            tags=tags,
        )
        return func

    return decorator
