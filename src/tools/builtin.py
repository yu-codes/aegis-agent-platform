"""
Built-in Tools

Common tools that ship with Aegis.
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Any

from src.tools.registry import tool, ToolCategory


@tool(
    name="get_current_time",
    description="Get the current date and time in the specified timezone",
    category=ToolCategory.SYSTEM,
    tags=["time", "utility"],
)
async def get_current_time(timezone: str = "UTC") -> str:
    """Get current time in the specified timezone."""
    from zoneinfo import ZoneInfo
    
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"Error getting time: {str(e)}"


@tool(
    name="calculate",
    description="Evaluate a mathematical expression. Supports basic arithmetic, powers, and common functions.",
    category=ToolCategory.COMPUTE,
    tags=["math", "calculate"],
)
async def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, abs, round
    """
    import math
    
    # Safe subset of functions
    safe_functions = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }
    
    try:
        # Sanitize expression
        for char in expression:
            if char not in "0123456789+-*/().**, abcdefghijklmnopqrstuvwxyz":
                return f"Invalid character in expression: {char}"
        
        result = eval(expression, {"__builtins__": {}}, safe_functions)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool(
    name="web_search",
    description="Search the web for information. Returns a summary of search results.",
    category=ToolCategory.WEB,
    tags=["search", "web"],
    timeout=30.0,
)
async def web_search(query: str, num_results: int = 5) -> str:
    """
    Perform a web search.
    
    Note: This is a placeholder. In production, integrate with
    a search API (Google, Bing, Serper, etc.)
    """
    # Placeholder implementation
    return f"[Web search for '{query}' would return {num_results} results. Integrate with a search API for real results.]"


@tool(
    name="http_get",
    description="Fetch content from a URL using HTTP GET request",
    category=ToolCategory.WEB,
    tags=["http", "web", "fetch"],
    timeout=30.0,
)
async def http_get(url: str, headers: dict[str, str] | None = None) -> str:
    """Fetch content from a URL."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=25) as response:
                if response.status != 200:
                    return f"HTTP Error: {response.status}"
                
                content = await response.text()
                
                # Truncate if too long
                if len(content) > 10000:
                    content = content[:10000] + "\n[Content truncated...]"
                
                return content
    except asyncio.TimeoutError:
        return "Request timed out"
    except Exception as e:
        return f"Error fetching URL: {str(e)}"


@tool(
    name="json_parse",
    description="Parse a JSON string and extract data using a JSONPath-like query",
    category=ToolCategory.DATA,
    tags=["json", "parse", "data"],
)
async def json_parse(json_string: str, path: str | None = None) -> str:
    """
    Parse JSON and optionally extract a value.
    
    Path uses dot notation: "data.users.0.name"
    """
    import json
    
    try:
        data = json.loads(json_string)
        
        if path:
            for key in path.split("."):
                if isinstance(data, list):
                    data = data[int(key)]
                elif isinstance(data, dict):
                    data = data[key]
                else:
                    return f"Cannot traverse path at: {key}"
        
        return json.dumps(data, indent=2)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Path not found: {str(e)}"
    except Exception as e:
        return f"Error parsing JSON: {str(e)}"


@tool(
    name="text_summary",
    description="Create a summary of the given text",
    category=ToolCategory.DATA,
    tags=["text", "summary", "nlp"],
)
async def text_summary(text: str, max_sentences: int = 3) -> str:
    """
    Create a simple extractive summary.
    
    Note: This is a basic implementation. For better summaries,
    integrate with an LLM or specialized summarization model.
    """
    import re
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= max_sentences:
        return text
    
    # Simple heuristic: take first sentence and most "important" ones
    # (those with more words)
    scored = [(s, len(s.split())) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Include first sentence and top scored ones
    summary_sentences = [sentences[0]]
    for sentence, _ in scored[:max_sentences - 1]:
        if sentence not in summary_sentences:
            summary_sentences.append(sentence)
    
    # Sort by original order
    ordered = [s for s in sentences if s in summary_sentences][:max_sentences]
    
    return " ".join(ordered)


@tool(
    name="string_transform",
    description="Transform a string using various operations",
    category=ToolCategory.DATA,
    tags=["string", "transform", "text"],
)
async def string_transform(
    text: str,
    operation: str,  # uppercase, lowercase, title, reverse, strip
) -> str:
    """Apply a transformation to text."""
    operations = {
        "uppercase": str.upper,
        "lowercase": str.lower,
        "title": str.title,
        "reverse": lambda s: s[::-1],
        "strip": str.strip,
        "capitalize": str.capitalize,
    }
    
    if operation not in operations:
        return f"Unknown operation: {operation}. Available: {list(operations.keys())}"
    
    return operations[operation](text)


def register_builtin_tools(registry) -> None:
    """Register all built-in tools with a registry."""
    registry.register_decorated(get_current_time)
    registry.register_decorated(calculate)
    registry.register_decorated(web_search)
    registry.register_decorated(http_get)
    registry.register_decorated(json_parse)
    registry.register_decorated(text_summary)
    registry.register_decorated(string_transform)
