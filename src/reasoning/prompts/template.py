"""
Prompt Template System

Provides templating and versioning for prompts.
Enables A/B testing, tracking, and safe prompt updates.

Design decisions:
- Templates use Jinja2 for flexibility
- Immutable templates (create new versions, don't modify)
- Registry pattern for centralized access
- Metadata supports evaluation and tracking
"""

import hashlib
from datetime import datetime
from typing import Any

from jinja2 import Environment, BaseLoader, TemplateSyntaxError, UndefinedError
from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """
    A versioned prompt template.
    
    Templates are immutable after creation. To update a prompt,
    create a new version. This enables:
    - Rollback to previous versions
    - A/B testing between versions
    - Tracking which version produced which outputs
    """
    
    name: str
    template: str
    version: str
    description: str = ""
    
    # Metadata
    author: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: list[str] = Field(default_factory=list)
    
    # Expected variables (for validation)
    required_variables: set[str] = Field(default_factory=set)
    optional_variables: set[str] = Field(default_factory=set)
    
    # Configuration
    max_tokens: int | None = None  # Recommended max tokens for response
    temperature: float | None = None  # Recommended temperature
    
    class Config:
        frozen = True
    
    @property
    def content_hash(self) -> str:
        """
        Generate hash of template content.
        
        Used for detecting changes and deduplication.
        """
        content = f"{self.name}:{self.version}:{self.template}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def render(self, **variables: Any) -> str:
        """
        Render the template with provided variables.
        
        Args:
            **variables: Template variables
            
        Returns:
            Rendered prompt string
            
        Raises:
            ValueError: If required variables are missing
            TemplateSyntaxError: If template is invalid
        """
        # Check required variables
        missing = self.required_variables - set(variables.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Render with Jinja2
        env = Environment(loader=BaseLoader())
        try:
            jinja_template = env.from_string(self.template)
            return jinja_template.render(**variables)
        except UndefinedError as e:
            raise ValueError(f"Undefined variable in template: {e}")
    
    def validate_template(self) -> list[str]:
        """
        Validate template syntax.
        
        Returns list of errors (empty if valid).
        """
        errors = []
        env = Environment(loader=BaseLoader())
        
        try:
            env.parse(self.template)
        except TemplateSyntaxError as e:
            errors.append(f"Syntax error: {e}")
        
        return errors


class PromptRegistry:
    """
    Central registry for prompt templates.
    
    Provides:
    - Template storage and retrieval
    - Version management
    - Default prompts for common use cases
    """
    
    def __init__(self):
        self._templates: dict[str, dict[str, PromptTemplate]] = {}
        self._default_versions: dict[str, str] = {}
        
        # Register built-in templates
        self._register_defaults()
    
    def _register_defaults(self) -> None:
        """Register default system prompts."""
        
        # System prompt for general agent
        self.register(PromptTemplate(
            name="agent_system",
            version="1.0.0",
            template="""You are Aegis, an intelligent AI assistant with access to tools.

Your capabilities:
- Answer questions accurately and helpfully
- Use tools when they would help accomplish the user's goal
- Break down complex tasks into steps
- Admit when you don't know something

Guidelines:
- Be concise but thorough
- Verify information before presenting it as fact
- Use tools proactively when they would be helpful
- If a tool fails, explain what happened and try alternatives

{% if context %}
Additional context:
{{ context }}
{% endif %}

{% if tools %}
Available tools: {{ tools | join(', ') }}
{% endif %}""",
            description="Default system prompt for the Aegis agent",
            required_variables=set(),
            optional_variables={"context", "tools"},
            tags=["system", "default"],
        ))
        self.set_default_version("agent_system", "1.0.0")
        
        # ReAct reasoning prompt
        self.register(PromptTemplate(
            name="react_reasoning",
            version="1.0.0",
            template="""You are solving a task step by step using the ReAct framework.

Task: {{ task }}

{% if previous_steps %}
Previous steps:
{% for step in previous_steps %}
Thought {{ loop.index }}: {{ step.thought }}
Action {{ loop.index }}: {{ step.action }}
Observation {{ loop.index }}: {{ step.observation }}
{% endfor %}
{% endif %}

Now continue with the next step. Use this format:
Thought: [Your reasoning about what to do next]
Action: [The action to take - either use a tool or provide final answer]
Action Input: [Input for the action if using a tool]

If you have enough information to answer, use:
Thought: [Your final reasoning]
Action: Final Answer
Action Input: [Your complete answer to the task]""",
            description="ReAct framework prompt for step-by-step reasoning",
            required_variables={"task"},
            optional_variables={"previous_steps"},
            tags=["reasoning", "react"],
        ))
        self.set_default_version("react_reasoning", "1.0.0")
        
        # Tool calling prompt
        self.register(PromptTemplate(
            name="tool_calling",
            version="1.0.0",
            template="""Based on the conversation, determine if you need to use any tools.

Available tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
  Parameters: {{ tool.parameters | tojson }}
{% endfor %}

User request: {{ user_message }}

If you need to use a tool, respond with the tool call.
If you can answer directly without tools, provide your response.
If you need multiple tools, you can call them in sequence.""",
            description="Prompt for tool selection and calling",
            required_variables={"tools", "user_message"},
            tags=["tools"],
        ))
        self.set_default_version("tool_calling", "1.0.0")
        
        # RAG context prompt
        self.register(PromptTemplate(
            name="rag_context",
            version="1.0.0",
            template="""Use the following retrieved documents to help answer the question.
If the documents don't contain relevant information, say so and answer based on your knowledge.

Retrieved Documents:
{% for doc in documents %}
[Document {{ loop.index }}] (Relevance: {{ "%.2f"|format(doc.score) }})
Source: {{ doc.source or 'Unknown' }}
{{ doc.content }}

{% endfor %}
Question: {{ question }}

Instructions:
- Cite document numbers when using information from them
- If documents conflict, note the discrepancy
- Don't make up information not in the documents or your training""",
            description="Prompt for incorporating RAG context",
            required_variables={"documents", "question"},
            tags=["rag", "context"],
        ))
        self.set_default_version("rag_context", "1.0.0")
        
        # Summary prompt for memory
        self.register(PromptTemplate(
            name="conversation_summary",
            version="1.0.0",
            template="""Summarize the following conversation, preserving key information.

Conversation:
{% for msg in messages %}
{{ msg.role.value }}: {{ msg.content }}
{% endfor %}

Create a concise summary that captures:
- The main topics discussed
- Key decisions or conclusions reached
- Important facts or context mentioned
- Any pending tasks or questions

Summary:""",
            description="Prompt for summarizing conversation history",
            required_variables={"messages"},
            tags=["memory", "summary"],
        ))
        self.set_default_version("conversation_summary", "1.0.0")
    
    def register(self, template: PromptTemplate) -> None:
        """
        Register a new template version.
        
        Args:
            template: The template to register
        """
        # Validate before registering
        errors = template.validate_template()
        if errors:
            raise ValueError(f"Invalid template: {errors}")
        
        if template.name not in self._templates:
            self._templates[template.name] = {}
        
        self._templates[template.name][template.version] = template
    
    def get(
        self,
        name: str,
        version: str | None = None,
    ) -> PromptTemplate | None:
        """
        Get a template by name and optional version.
        
        Args:
            name: Template name
            version: Specific version (uses default if not specified)
            
        Returns:
            Template or None if not found
        """
        if name not in self._templates:
            return None
        
        if version is None:
            version = self._default_versions.get(name)
            if version is None:
                # Return latest version
                versions = sorted(self._templates[name].keys())
                version = versions[-1] if versions else None
        
        if version is None:
            return None
        
        return self._templates[name].get(version)
    
    def set_default_version(self, name: str, version: str) -> None:
        """Set the default version for a template."""
        if name not in self._templates:
            raise ValueError(f"Template not found: {name}")
        if version not in self._templates[name]:
            raise ValueError(f"Version not found: {name}:{version}")
        
        self._default_versions[name] = version
    
    def list_templates(self) -> list[str]:
        """List all registered template names."""
        return list(self._templates.keys())
    
    def list_versions(self, name: str) -> list[str]:
        """List all versions of a template."""
        if name not in self._templates:
            return []
        return sorted(self._templates[name].keys())


# Global registry instance
_registry: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    """Get or create the global prompt registry."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
