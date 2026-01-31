"""
Advanced Capabilities Module

Multi-agent orchestration, critic agent, and plugin system.
"""

from src.advanced.multi_agent import AgentPool, AgentOrchestrator, AgentDefinition
from src.advanced.critic import CriticAgent, Critique, CriticStrategy
from src.advanced.plugins import PluginManager, Plugin, PluginContext

__all__ = [
    # Multi-agent
    "AgentPool",
    "AgentOrchestrator",
    "AgentDefinition",
    # Critic
    "CriticAgent",
    "Critique",
    "CriticStrategy",
    # Plugins
    "PluginManager",
    "Plugin",
    "PluginContext",
]
