"""
Advanced Capabilities Module

Multi-agent orchestration, critic agent, and plugin system.
"""

from src.advanced.critic import CriticAgent, CriticStrategy, Critique
from src.advanced.multi_agent import AgentDefinition, AgentOrchestrator, AgentPool
from src.advanced.plugins import Plugin, PluginContext, PluginManager

__all__ = [
    "AgentDefinition",
    "AgentOrchestrator",
    # Multi-agent
    "AgentPool",
    # Critic
    "CriticAgent",
    "CriticStrategy",
    "Critique",
    "Plugin",
    "PluginContext",
    # Plugins
    "PluginManager",
]
