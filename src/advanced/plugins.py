"""
Plugin System

Extensible plugin architecture.

Design decisions:
- Simple plugin interface
- Hook-based extension
- Lifecycle management
- Sandboxed execution
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import UUID, uuid4


class PluginState(str, Enum):
    """Plugin lifecycle states."""
    
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


class HookType(str, Enum):
    """Types of hooks that plugins can register."""
    
    # Request lifecycle
    PRE_REQUEST = "pre_request"
    POST_REQUEST = "post_request"
    
    # Message processing
    PRE_PROCESS = "pre_process"
    POST_PROCESS = "post_process"
    
    # Tool execution
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    
    # Response generation
    PRE_RESPONSE = "pre_response"
    POST_RESPONSE = "post_response"
    
    # Session lifecycle
    SESSION_CREATE = "session_create"
    SESSION_END = "session_end"


@dataclass
class PluginContext:
    """
    Context passed to plugin hooks.
    
    Contains data relevant to the current operation.
    """
    
    # Request info
    request_id: str | None = None
    session_id: UUID | None = None
    user_id: str | None = None
    
    # Current data (varies by hook type)
    data: dict[str, Any] = field(default_factory=dict)
    
    # Plugin can set this to modify the flow
    modified_data: dict[str, Any] | None = None
    should_continue: bool = True
    error: str | None = None


@dataclass
class PluginMetadata:
    """Plugin metadata."""
    
    name: str
    version: str
    description: str = ""
    author: str = ""
    
    # Dependencies
    requires: list[str] = field(default_factory=list)  # Other plugin names
    
    # Configuration schema
    config_schema: dict[str, Any] | None = None


class Plugin(ABC):
    """
    Base class for plugins.
    
    Plugins extend platform functionality through hooks.
    """
    
    def __init__(self):
        self.id = uuid4()
        self.state = PluginState.REGISTERED
        self._config: dict[str, Any] = {}
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    async def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the plugin.
        
        Called when plugin is activated.
        """
        self._config = config
        self.state = PluginState.ACTIVE
    
    async def shutdown(self) -> None:
        """
        Shutdown the plugin.
        
        Called when plugin is deactivated.
        """
        self.state = PluginState.DISABLED
    
    def get_hooks(self) -> dict[HookType, Callable]:
        """
        Return hooks this plugin provides.
        
        Override to register hooks.
        """
        return {}


class LoggingPlugin(Plugin):
    """Example plugin that logs all requests."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="logging",
            version="1.0.0",
            description="Logs all requests and responses",
            author="Aegis Team",
        )
    
    def get_hooks(self) -> dict[HookType, Callable]:
        return {
            HookType.PRE_REQUEST: self.on_pre_request,
            HookType.POST_RESPONSE: self.on_post_response,
        }
    
    async def on_pre_request(self, context: PluginContext) -> None:
        print(f"[LOG] Request: {context.request_id}")
    
    async def on_post_response(self, context: PluginContext) -> None:
        print(f"[LOG] Response for: {context.request_id}")


class PluginManager:
    """
    Manages plugin lifecycle and hook execution.
    """
    
    def __init__(self):
        self._plugins: dict[str, Plugin] = {}
        self._hooks: dict[HookType, list[tuple[str, Callable]]] = {
            hook: [] for hook in HookType
        }
    
    async def register(self, plugin: Plugin, config: dict[str, Any] | None = None) -> bool:
        """
        Register and initialize a plugin.
        """
        name = plugin.metadata.name
        
        if name in self._plugins:
            return False
        
        # Check dependencies
        for dep in plugin.metadata.requires:
            if dep not in self._plugins:
                raise ValueError(f"Missing dependency: {dep}")
        
        try:
            plugin.state = PluginState.INITIALIZING
            await plugin.initialize(config or {})
            
            self._plugins[name] = plugin
            
            # Register hooks
            for hook_type, handler in plugin.get_hooks().items():
                self._hooks[hook_type].append((name, handler))
            
            return True
            
        except Exception as e:
            plugin.state = PluginState.ERROR
            raise
    
    async def unregister(self, plugin_name: str) -> bool:
        """
        Unregister and shutdown a plugin.
        """
        if plugin_name not in self._plugins:
            return False
        
        plugin = self._plugins[plugin_name]
        
        # Remove hooks
        for hook_type in HookType:
            self._hooks[hook_type] = [
                (name, handler) for name, handler in self._hooks[hook_type]
                if name != plugin_name
            ]
        
        await plugin.shutdown()
        del self._plugins[plugin_name]
        
        return True
    
    async def execute_hook(
        self,
        hook_type: HookType,
        context: PluginContext,
    ) -> PluginContext:
        """
        Execute all handlers for a hook type.
        
        Returns the (possibly modified) context.
        """
        for plugin_name, handler in self._hooks[hook_type]:
            if not context.should_continue:
                break
            
            try:
                await handler(context)
            except Exception as e:
                # Log error but continue
                print(f"Plugin {plugin_name} error in {hook_type}: {e}")
                context.error = str(e)
        
        return context
    
    def get_plugin(self, name: str) -> Plugin | None:
        """Get a plugin by name."""
        return self._plugins.get(name)
    
    def list_plugins(self) -> list[PluginMetadata]:
        """List all registered plugins."""
        return [p.metadata for p in self._plugins.values()]
    
    def is_active(self, name: str) -> bool:
        """Check if a plugin is active."""
        plugin = self._plugins.get(name)
        return plugin is not None and plugin.state == PluginState.ACTIVE


# Pre-built plugins


class RateLimitPlugin(Plugin):
    """Plugin that rate limits requests per user."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="rate_limit",
            version="1.0.0",
            description="Rate limits requests per user",
            config_schema={
                "requests_per_minute": {"type": "integer", "default": 30},
            },
        )
    
    def __init__(self):
        super().__init__()
        self._requests: dict[str, list[float]] = {}
    
    def get_hooks(self) -> dict[HookType, Callable]:
        return {
            HookType.PRE_REQUEST: self.check_rate_limit,
        }
    
    async def check_rate_limit(self, context: PluginContext) -> None:
        import time
        
        user_id = context.user_id or "anonymous"
        now = time.time()
        rpm = self._config.get("requests_per_minute", 30)
        
        if user_id not in self._requests:
            self._requests[user_id] = []
        
        # Clean old entries
        self._requests[user_id] = [
            t for t in self._requests[user_id]
            if now - t < 60
        ]
        
        if len(self._requests[user_id]) >= rpm:
            context.should_continue = False
            context.error = "Rate limit exceeded"
            return
        
        self._requests[user_id].append(now)


class MetricsPlugin(Plugin):
    """Plugin that collects metrics."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="metrics",
            version="1.0.0",
            description="Collects platform metrics",
        )
    
    def __init__(self):
        super().__init__()
        self._request_count = 0
        self._error_count = 0
    
    def get_hooks(self) -> dict[HookType, Callable]:
        return {
            HookType.PRE_REQUEST: self.on_request,
            HookType.POST_RESPONSE: self.on_response,
        }
    
    async def on_request(self, context: PluginContext) -> None:
        self._request_count += 1
    
    async def on_response(self, context: PluginContext) -> None:
        if context.error:
            self._error_count += 1
    
    def get_metrics(self) -> dict[str, Any]:
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
        }
