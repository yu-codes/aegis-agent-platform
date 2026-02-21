"""
Domain Registry

Domain-specific RAG configurations.

Based on: src/domains/registry.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
import yaml


@dataclass
class DomainConfig:
    """Configuration for a domain."""

    id: str = ""
    name: str = ""
    description: str = ""
    version: str = "1.0.0"

    # System prompt
    system_prompt: str = ""

    # Model settings
    model: str = "claude-3-5-sonnet"
    max_tokens: int = 4096
    temperature: float = 0.7

    # RAG settings
    rag_enabled: bool = True
    rag_namespace: str = "default"
    rag_top_k: int = 5
    rag_min_score: float = 0.5

    # Tool settings
    enabled_tools: list[str] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)

    # Safety settings
    content_filter_level: str = "moderate"  # strict, moderate, low
    pii_handling: str = "warn"  # block, warn, allow
    injection_guard: bool = True

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "DomainConfig":
        """Create from dictionary."""
        model_settings = data.get("model", {})
        rag_settings = data.get("rag", {})
        tools_settings = data.get("tools", {})
        safety_settings = data.get("safety", {})

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            system_prompt=data.get("system_prompt", ""),
            model=model_settings.get("default", "claude-3-5-sonnet"),
            max_tokens=model_settings.get("max_tokens", 4096),
            temperature=model_settings.get("temperature", 0.7),
            rag_enabled=rag_settings.get("enabled", True),
            rag_namespace=rag_settings.get("namespace", "default"),
            rag_top_k=rag_settings.get("top_k", 5),
            rag_min_score=rag_settings.get("min_similarity", 0.5),
            enabled_tools=tools_settings.get("enabled", []),
            disabled_tools=tools_settings.get("disabled", []),
            content_filter_level=safety_settings.get("content_filter", "moderate"),
            pii_handling=safety_settings.get("pii_handling", "warn"),
            injection_guard=safety_settings.get("injection_guard", True),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "system_prompt": self.system_prompt,
            "model": {
                "default": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            "rag": {
                "enabled": self.rag_enabled,
                "namespace": self.rag_namespace,
                "top_k": self.rag_top_k,
                "min_similarity": self.rag_min_score,
            },
            "tools": {
                "enabled": self.enabled_tools,
                "disabled": self.disabled_tools,
            },
            "safety": {
                "content_filter": self.content_filter_level,
                "pii_handling": self.pii_handling,
                "injection_guard": self.injection_guard,
            },
            "metadata": self.metadata,
        }


class DomainRegistry:
    """
    Registry for domain configurations.

    Manages domain-specific settings for RAG, tools, and safety.
    """

    def __init__(self):
        self._domains: dict[str, DomainConfig] = {}
        self._default_domain: str | None = None

    def register(self, config: DomainConfig) -> None:
        """Register a domain configuration."""
        self._domains[config.id] = config

        if self._default_domain is None:
            self._default_domain = config.id

    def unregister(self, domain_id: str) -> bool:
        """Unregister a domain."""
        if domain_id in self._domains:
            del self._domains[domain_id]
            if self._default_domain == domain_id:
                self._default_domain = next(iter(self._domains), None)
            return True
        return False

    def get(self, domain_id: str) -> DomainConfig | None:
        """Get domain by ID."""
        return self._domains.get(domain_id)

    def get_default(self) -> DomainConfig | None:
        """Get default domain."""
        if self._default_domain:
            return self._domains.get(self._default_domain)
        return None

    def set_default(self, domain_id: str) -> bool:
        """Set default domain."""
        if domain_id in self._domains:
            self._default_domain = domain_id
            return True
        return False

    def list_domains(self) -> list[DomainConfig]:
        """List all registered domains."""
        return list(self._domains.values())

    def __len__(self) -> int:
        return len(self._domains)

    def __iter__(self) -> Iterator[DomainConfig]:
        return iter(self._domains.values())

    def __contains__(self, domain_id: str) -> bool:
        return domain_id in self._domains

    @classmethod
    def from_directory(cls, path: Path | str) -> "DomainRegistry":
        """Load domains from a directory of YAML files."""
        registry = cls()
        path = Path(path)

        if not path.exists():
            return registry

        for yaml_file in path.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                    if data:
                        # Handle multiple domains in one file
                        if isinstance(data, list):
                            for item in data:
                                config = DomainConfig.from_dict(item)
                                if config.id:
                                    registry.register(config)
                        else:
                            config = DomainConfig.from_dict(data)
                            if not config.id:
                                config.id = yaml_file.stem
                            registry.register(config)
            except Exception as e:
                print(f"Failed to load domain from {yaml_file}: {e}")

        return registry

    def save_to_directory(self, path: Path | str) -> None:
        """Save domains to a directory of YAML files."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for domain in self._domains.values():
            file_path = path / f"{domain.id}.yaml"
            with open(file_path, "w") as f:
                yaml.dump(domain.to_dict(), f, default_flow_style=False)
