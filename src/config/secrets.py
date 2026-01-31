"""
Secret Management

Provides secure access to sensitive configuration values.
Abstracts secret storage backend (env vars, vault, cloud KMS).

Design decisions:
- Secrets are never logged or serialized to plain text
- Support for multiple secret backends (extensible)
- Lazy loading to avoid exposing secrets at import time
"""

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import SecretStr


class SecretBackend(str, Enum):
    """Supported secret storage backends."""

    ENVIRONMENT = "environment"
    # Future: VAULT = "vault"
    # Future: AWS_SECRETS = "aws_secrets"
    # Future: GCP_SECRET_MANAGER = "gcp_secret_manager"


class SecretProvider(ABC):
    """Abstract base for secret providers."""

    @abstractmethod
    async def get_secret(self, key: str) -> SecretStr | None:
        """Retrieve a secret by key."""
        pass

    @abstractmethod
    async def set_secret(self, key: str, value: str) -> None:
        """Store a secret (if supported by backend)."""
        pass

    @abstractmethod
    async def delete_secret(self, key: str) -> None:
        """Delete a secret (if supported by backend)."""
        pass


class EnvironmentSecretProvider(SecretProvider):
    """
    Environment variable based secret provider.

    Simple and works everywhere. For production, consider
    using a proper secret management system.
    """

    def __init__(self, prefix: str = "AEGIS_SECRET_"):
        self._prefix = prefix

    async def get_secret(self, key: str) -> SecretStr | None:
        """Get secret from environment variable."""
        env_key = f"{self._prefix}{key.upper()}"
        value = os.environ.get(env_key)
        return SecretStr(value) if value else None

    async def set_secret(self, key: str, value: str) -> None:
        """
        Set environment variable.

        Note: This only affects the current process.
        Not recommended for production use.
        """
        env_key = f"{self._prefix}{key.upper()}"
        os.environ[env_key] = value

    async def delete_secret(self, key: str) -> None:
        """Remove environment variable."""
        env_key = f"{self._prefix}{key.upper()}"
        os.environ.pop(env_key, None)


class SecretManager:
    """
    Unified secret management interface.

    Provides a consistent API regardless of the underlying
    secret storage backend. Supports caching to reduce
    backend calls.

    Usage:
        manager = SecretManager()
        api_key = await manager.get("openai_api_key")
    """

    def __init__(
        self,
        backend: SecretBackend = SecretBackend.ENVIRONMENT,
        cache_secrets: bool = True,
        **backend_kwargs: Any,
    ):
        self._cache_enabled = cache_secrets
        self._cache: dict[str, SecretStr] = {}
        self._provider = self._create_provider(backend, **backend_kwargs)

    def _create_provider(
        self,
        backend: SecretBackend,
        **kwargs: Any,
    ) -> SecretProvider:
        """Factory method for secret providers."""
        if backend == SecretBackend.ENVIRONMENT:
            return EnvironmentSecretProvider(**kwargs)
        # Future backends would be instantiated here
        raise ValueError(f"Unsupported secret backend: {backend}")

    async def get(self, key: str, required: bool = False) -> SecretStr | None:
        """
        Retrieve a secret.

        Args:
            key: Secret identifier
            required: If True, raises error when secret not found

        Returns:
            Secret value wrapped in SecretStr, or None

        Raises:
            ValueError: If required=True and secret not found
        """
        # Check cache first
        if self._cache_enabled and key in self._cache:
            return self._cache[key]

        secret = await self._provider.get_secret(key)

        if secret is None and required:
            raise ValueError(f"Required secret not found: {key}")

        # Cache the result
        if self._cache_enabled and secret is not None:
            self._cache[key] = secret

        return secret

    async def set(self, key: str, value: str) -> None:
        """Store a secret."""
        await self._provider.set_secret(key, value)

        # Update cache
        if self._cache_enabled:
            self._cache[key] = SecretStr(value)

    async def delete(self, key: str) -> None:
        """Delete a secret."""
        await self._provider.delete_secret(key)

        # Remove from cache
        self._cache.pop(key, None)

    def clear_cache(self) -> None:
        """Clear the secret cache."""
        self._cache.clear()


# Module-level singleton for convenience
_secret_manager: SecretManager | None = None


def get_secret_manager() -> SecretManager:
    """Get or create the global secret manager instance."""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager
