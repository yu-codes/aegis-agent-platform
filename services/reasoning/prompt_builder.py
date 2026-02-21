"""
Prompt Builder

Template-based prompt construction.

Design decisions:
- Jinja2-style templates
- Variable injection
- Prompt composition
- Version management
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4
import re


@dataclass
class PromptVariable:
    """A variable in a prompt template."""

    name: str
    description: str = ""
    required: bool = True
    default: Any = None
    type: str = "string"


@dataclass
class PromptTemplate:
    """A reusable prompt template."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    template: str = ""
    variables: list[PromptVariable] = field(default_factory=list)

    # Metadata
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_variable_names(self) -> list[str]:
        """Extract variable names from template."""
        # Match {{variable}} or {variable}
        pattern = r"\{\{?\s*(\w+)\s*\}?\}"
        return list(set(re.findall(pattern, self.template)))


class PromptBuilder:
    """
    Build prompts from templates and variables.

    Supports variable injection and composition.
    """

    def __init__(self):
        self._templates: dict[str, PromptTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default templates."""
        self.register(
            PromptTemplate(
                name="system_default",
                description="Default system prompt",
                template="""You are a helpful AI assistant.

{{#if persona}}
Persona: {{persona}}
{{/if}}

{{#if guidelines}}
Guidelines:
{{guidelines}}
{{/if}}

Current date: {{current_date}}""",
                variables=[
                    PromptVariable(name="persona", required=False),
                    PromptVariable(name="guidelines", required=False),
                    PromptVariable(name="current_date", default="unknown"),
                ],
            )
        )

        self.register(
            PromptTemplate(
                name="rag_context",
                description="RAG context injection",
                template="""Use the following context to answer the user's question.
If the context doesn't contain relevant information, say so.

Context:
{{context}}

---
User Question: {{query}}""",
                variables=[
                    PromptVariable(name="context", required=True),
                    PromptVariable(name="query", required=True),
                ],
            )
        )

        self.register(
            PromptTemplate(
                name="tool_result",
                description="Tool result formatting",
                template="""Tool: {{tool_name}}
Result: {{result}}

{{#if error}}
Error: {{error}}
{{/if}}""",
                variables=[
                    PromptVariable(name="tool_name", required=True),
                    PromptVariable(name="result", required=True),
                    PromptVariable(name="error", required=False),
                ],
            )
        )

        self.register(
            PromptTemplate(
                name="chain_of_thought",
                description="Chain of thought reasoning",
                template="""Let's think through this step by step:

Question: {{question}}

{{#if context}}
Available context:
{{context}}
{{/if}}

Please reason through this carefully before providing your answer.""",
                variables=[
                    PromptVariable(name="question", required=True),
                    PromptVariable(name="context", required=False),
                ],
            )
        )

    def register(self, template: PromptTemplate) -> None:
        """Register a template."""
        self._templates[template.name] = template

    def get_template(self, name: str) -> PromptTemplate | None:
        """Get template by name."""
        return self._templates.get(name)

    def list_templates(self) -> list[PromptTemplate]:
        """List all templates."""
        return list(self._templates.values())

    def build(
        self,
        template_name: str,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """
        Build prompt from template.

        Args:
            template_name: Name of template to use
            variables: Variables to inject

        Returns:
            Rendered prompt string
        """
        template = self._templates.get(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        return self.render(template.template, variables or {})

    def render(
        self,
        template_str: str,
        variables: dict[str, Any],
    ) -> str:
        """
        Render a template string with variables.

        Supports:
        - {{variable}} - simple substitution
        - {{#if var}}...{{/if}} - conditional blocks
        - {{#each items}}...{{/each}} - iteration
        """
        result = template_str

        # Process conditionals first
        result = self._process_conditionals(result, variables)

        # Process iterations
        result = self._process_each(result, variables)

        # Simple variable substitution
        result = self._substitute_variables(result, variables)

        return result.strip()

    def _substitute_variables(
        self,
        template: str,
        variables: dict[str, Any],
    ) -> str:
        """Substitute {{variable}} patterns."""

        def replace(match):
            var_name = match.group(1).strip()
            value = variables.get(var_name, "")
            return str(value) if value else ""

        pattern = r"\{\{\s*(\w+)\s*\}\}"
        return re.sub(pattern, replace, template)

    def _process_conditionals(
        self,
        template: str,
        variables: dict[str, Any],
    ) -> str:
        """Process {{#if var}}...{{/if}} blocks."""
        pattern = r"\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}"

        def replace(match):
            var_name = match.group(1).strip()
            content = match.group(2)

            value = variables.get(var_name)
            if value:
                return content
            return ""

        return re.sub(pattern, replace, template, flags=re.DOTALL)

    def _process_each(
        self,
        template: str,
        variables: dict[str, Any],
    ) -> str:
        """Process {{#each items}}...{{/each}} blocks."""
        pattern = r"\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}"

        def replace(match):
            var_name = match.group(1).strip()
            item_template = match.group(2)

            items = variables.get(var_name, [])
            if not items:
                return ""

            results = []
            for i, item in enumerate(items):
                item_vars = {"item": item, "index": i}
                if isinstance(item, dict):
                    item_vars.update(item)
                rendered = self._substitute_variables(item_template, item_vars)
                results.append(rendered)

            return "\n".join(results)

        return re.sub(pattern, replace, template, flags=re.DOTALL)

    def compose(
        self,
        parts: list[tuple[str, dict[str, Any]]],
        separator: str = "\n\n",
    ) -> str:
        """
        Compose multiple template parts.

        Args:
            parts: List of (template_name, variables) tuples
            separator: Separator between parts

        Returns:
            Combined prompt string
        """
        rendered_parts = []

        for template_name, variables in parts:
            rendered = self.build(template_name, variables)
            if rendered:
                rendered_parts.append(rendered)

        return separator.join(rendered_parts)
