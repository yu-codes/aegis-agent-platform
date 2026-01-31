"""
Prompt Module

Templates and versioning for prompts.
"""

from src.reasoning.prompts.template import (
    PromptRegistry,
    PromptTemplate,
    get_prompt_registry,
)

__all__ = [
    "PromptRegistry",
    "PromptTemplate",
    "get_prompt_registry",
]
