"""
Domain Registry

Loads, validates, and manages domain profiles.

Design decisions:
- Profiles loaded from YAML files for easy editing
- Supports directory scanning and single-file loading
- Validates profiles at load time
- Thread-safe registry access
- Inheritance resolution (extends)
- Hot-reload support (for development)
"""

from pathlib import Path
from threading import RLock
from typing import Iterator
import logging

import yaml
from pydantic import ValidationError

from src.domains.profile import DomainProfile


logger = logging.getLogger(__name__)


class DomainRegistryError(Exception):
    """Base error for domain registry operations."""
    pass


class DomainNotFoundError(DomainRegistryError):
    """Raised when a domain is not found."""
    
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Domain not found: {name}")


class DomainValidationError(DomainRegistryError):
    """Raised when a domain profile fails validation."""
    
    def __init__(self, name: str, errors: list[str]):
        self.name = name
        self.errors = errors
        super().__init__(f"Domain validation failed for '{name}': {errors}")


class DomainCircularInheritanceError(DomainRegistryError):
    """Raised when circular inheritance is detected."""
    
    def __init__(self, chain: list[str]):
        self.chain = chain
        super().__init__(f"Circular inheritance detected: {' -> '.join(chain)}")


# Default fallback domain used when no domain matches
DEFAULT_DOMAIN = DomainProfile(
    name="general",
    version="1.0.0",
    display_name="General Purpose",
    description="Safe fallback domain for general conversations",
    prompt=DomainProfile.model_fields["prompt"].default,
    rag=DomainProfile.model_fields["rag"].default,
    memory=DomainProfile.model_fields["memory"].default,
    tools=DomainProfile.model_fields["tools"].default,
    reasoning=DomainProfile.model_fields["reasoning"].default,
    safety=DomainProfile.model_fields["safety"].default,
)


class DomainRegistry:
    """
    Registry for domain profiles.
    
    Manages loading, validation, inheritance resolution,
    and retrieval of domain profiles.
    
    Thread-safe: All operations use internal locking.
    
    Usage:
        registry = DomainRegistry.from_directory("config/domains")
        profile = registry.get("technical_support")
        
        # Or programmatically:
        registry = DomainRegistry()
        registry.register(my_profile)
    """
    
    def __init__(self, fallback: DomainProfile | None = None):
        """
        Initialize an empty registry.
        
        Args:
            fallback: Default domain when none matches (defaults to general)
        """
        self._domains: dict[str, DomainProfile] = {}
        self._lock = RLock()
        self._fallback = fallback or DEFAULT_DOMAIN
        
        # Always register the fallback
        self._domains[self._fallback.name] = self._fallback
    
    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        fallback: DomainProfile | None = None,
        recursive: bool = True,
    ) -> "DomainRegistry":
        """
        Create registry by scanning a directory for YAML profiles.
        
        Args:
            directory: Path to scan for *.yaml and *.yml files
            fallback: Default domain profile
            recursive: Scan subdirectories
            
        Returns:
            DomainRegistry with all valid profiles loaded
            
        Raises:
            DomainRegistryError: If directory doesn't exist
        """
        path = Path(directory)
        if not path.exists():
            raise DomainRegistryError(f"Directory not found: {directory}")
        if not path.is_dir():
            raise DomainRegistryError(f"Not a directory: {directory}")
        
        registry = cls(fallback=fallback)
        
        # Find all YAML files
        pattern = "**/*.y*ml" if recursive else "*.y*ml"
        for yaml_file in path.glob(pattern):
            if yaml_file.is_file():
                try:
                    registry.load_file(yaml_file)
                except DomainValidationError as e:
                    logger.warning(f"Skipping invalid profile {yaml_file}: {e}")
                except Exception as e:
                    logger.error(f"Error loading {yaml_file}: {e}")
        
        # Resolve inheritance after all profiles loaded
        registry._resolve_all_inheritance()
        
        return registry
    
    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        fallback: DomainProfile | None = None,
    ) -> "DomainRegistry":
        """Create registry from a single YAML file."""
        registry = cls(fallback=fallback)
        registry.load_file(filepath)
        return registry
    
    def load_file(self, filepath: str | Path) -> DomainProfile:
        """
        Load a domain profile from a YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            Loaded DomainProfile
            
        Raises:
            DomainValidationError: If profile is invalid
        """
        path = Path(filepath)
        if not path.exists():
            raise DomainRegistryError(f"File not found: {filepath}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if data is None:
            raise DomainValidationError(str(filepath), ["Empty file"])
        
        return self.register_from_dict(data, source=str(filepath))
    
    def register_from_dict(
        self,
        data: dict,
        source: str | None = None,
    ) -> DomainProfile:
        """
        Register a domain from a dictionary.
        
        Args:
            data: Profile data dictionary
            source: Source reference for error messages
            
        Returns:
            Created DomainProfile
        """
        try:
            profile = DomainProfile(**data)
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            name = data.get("name", source or "unknown")
            raise DomainValidationError(name, errors) from e
        
        return self.register(profile)
    
    def register(self, profile: DomainProfile) -> DomainProfile:
        """
        Register a domain profile.
        
        Args:
            profile: Profile to register
            
        Returns:
            The registered profile
            
        Note:
            If a profile with the same name exists, it's replaced.
        """
        with self._lock:
            existing = self._domains.get(profile.name)
            if existing:
                logger.info(
                    f"Replacing domain '{profile.name}' "
                    f"v{existing.version} -> v{profile.version}"
                )
            self._domains[profile.name] = profile
            logger.debug(f"Registered domain: {profile.name} v{profile.version}")
            return profile
    
    def unregister(self, name: str) -> bool:
        """
        Remove a domain from the registry.
        
        Args:
            name: Domain name to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name == self._fallback.name:
                raise DomainRegistryError("Cannot unregister fallback domain")
            if name in self._domains:
                del self._domains[name]
                return True
            return False
    
    def get(self, name: str) -> DomainProfile:
        """
        Get a domain profile by name.
        
        Args:
            name: Domain name
            
        Returns:
            DomainProfile
            
        Raises:
            DomainNotFoundError: If domain not found
        """
        with self._lock:
            if name not in self._domains:
                raise DomainNotFoundError(name)
            return self._domains[name]
    
    def get_or_fallback(self, name: str | None) -> DomainProfile:
        """
        Get a domain profile, falling back to default if not found.
        
        Args:
            name: Domain name (None returns fallback)
            
        Returns:
            DomainProfile (never raises for not found)
        """
        if name is None:
            return self._fallback
        
        with self._lock:
            return self._domains.get(name, self._fallback)
    
    def exists(self, name: str) -> bool:
        """Check if a domain exists."""
        with self._lock:
            return name in self._domains
    
    def list_domains(self) -> list[str]:
        """Get list of all registered domain names."""
        with self._lock:
            return list(self._domains.keys())
    
    def all_profiles(self) -> list[DomainProfile]:
        """Get all registered profiles."""
        with self._lock:
            return list(self._domains.values())
    
    def __iter__(self) -> Iterator[DomainProfile]:
        """Iterate over all profiles."""
        with self._lock:
            return iter(list(self._domains.values()))
    
    def __len__(self) -> int:
        """Number of registered domains."""
        with self._lock:
            return len(self._domains)
    
    def __contains__(self, name: str) -> bool:
        """Check if domain exists."""
        return self.exists(name)
    
    @property
    def fallback(self) -> DomainProfile:
        """Get the fallback domain."""
        return self._fallback
    
    def _resolve_all_inheritance(self) -> None:
        """
        Resolve inheritance for all profiles.
        
        Called after loading all profiles to resolve 'extends' references.
        """
        with self._lock:
            resolved: set[str] = set()
            
            for name in list(self._domains.keys()):
                if name not in resolved:
                    self._resolve_inheritance(name, resolved, [])
    
    def _resolve_inheritance(
        self,
        name: str,
        resolved: set[str],
        chain: list[str],
    ) -> DomainProfile:
        """
        Resolve inheritance for a single profile.
        
        Args:
            name: Profile name to resolve
            resolved: Set of already-resolved profiles
            chain: Current inheritance chain (for cycle detection)
            
        Returns:
            Resolved profile
        """
        if name in chain:
            raise DomainCircularInheritanceError(chain + [name])
        
        profile = self._domains.get(name)
        if profile is None:
            raise DomainNotFoundError(name)
        
        if name in resolved:
            return profile
        
        if profile.extends is None:
            resolved.add(name)
            return profile
        
        # Resolve parent first
        parent = self._resolve_inheritance(
            profile.extends,
            resolved,
            chain + [name],
        )
        
        # Merge parent with child (child overrides parent)
        merged = self._merge_profiles(parent, profile)
        self._domains[name] = merged
        resolved.add(name)
        
        return merged
    
    def _merge_profiles(
        self,
        parent: DomainProfile,
        child: DomainProfile,
    ) -> DomainProfile:
        """
        Merge parent profile into child.
        
        Child values override parent values.
        For nested configs, merging is shallow (child section replaces parent).
        """
        parent_dict = parent.model_dump()
        child_dict = child.model_dump()
        
        # Start with parent
        merged = parent_dict.copy()
        
        # Override with non-default child values
        for key, value in child_dict.items():
            if key in ("name", "version", "extends", "created_at", "updated_at"):
                # Always use child's identity/metadata
                merged[key] = value
            elif value is not None:
                # Child overrides parent
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    # Shallow merge for config sections
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
        
        # Clear extends since we've resolved it
        merged["extends"] = None
        
        return DomainProfile(**merged)
    
    def reload(self, directory: str | Path) -> int:
        """
        Reload all profiles from directory.
        
        Returns:
            Number of profiles loaded
        """
        path = Path(directory)
        count = 0
        
        for yaml_file in path.glob("**/*.y*ml"):
            try:
                self.load_file(yaml_file)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to reload {yaml_file}: {e}")
        
        self._resolve_all_inheritance()
        return count
    
    def export_all(self, directory: str | Path) -> None:
        """Export all profiles to YAML files."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            for profile in self._domains.values():
                filepath = path / f"{profile.name}.yaml"
                with open(filepath, "w", encoding="utf-8") as f:
                    yaml.dump(
                        profile.model_dump(exclude_none=True),
                        f,
                        default_flow_style=False,
                        sort_keys=False,
                    )
