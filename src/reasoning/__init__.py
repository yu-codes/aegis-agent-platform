"""
Reasoning Core Module

Contains LLM adapters, prompt management, and reasoning strategies.
This is the "brain" of the agent system.
"""

from src.reasoning.llm.base import BaseLLMAdapter
from src.reasoning.llm.openai_adapter import OpenAIAdapter
from src.reasoning.llm.anthropic_adapter import AnthropicAdapter
from src.reasoning.prompts.template import PromptTemplate, PromptRegistry
from src.reasoning.strategies.base import ReasoningStrategy
from src.reasoning.strategies.react import ReActStrategy
from src.reasoning.strategies.tool_calling import ToolCallingStrategy

__all__ = [
    # LLM Adapters
    "BaseLLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    # Prompts
    "PromptTemplate",
    "PromptRegistry",
    # Strategies
    "ReasoningStrategy",
    "ReActStrategy",
    "ToolCallingStrategy",
]
