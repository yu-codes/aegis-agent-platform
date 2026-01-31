"""
Prompt Module

Templates and versioning for prompts.
"""

from src.reasoning.prompts.template import (
    PromptTemplate,
    PromptRegistry,
    get_prompt_registry,
)

__all__ = [
    "PromptTemplate",
    "PromptRegistry",
    "get_prompt_registry",
]
