"""
Configuration Module

Centralized configuration management for the Aegis platform.
Handles settings, secrets, and environment-specific configurations.
"""

from src.config.settings import Settings, get_settings
from src.config.secrets import SecretManager
from src.config.model_routing import ModelRouter, ModelConfig

__all__ = [
    "Settings",
    "get_settings",
    "SecretManager",
    "ModelRouter",
    "ModelConfig",
]
