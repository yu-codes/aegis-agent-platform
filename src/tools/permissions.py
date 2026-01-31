"""
Tool Permission Management

RBAC and scope-based access control for tools.

Design decisions:
- Role-based permission model
- Scope-based restrictions
- Rate limiting per user/tool
- Audit trail for access decisions
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.core.types import ExecutionContext


class ToolScope(str, Enum):
    """Scopes that restrict tool access."""

    READ = "read"  # Read-only operations
    WRITE = "write"  # Write/modify operations
    EXECUTE = "execute"  # Execute commands/code
    NETWORK = "network"  # Network access
    FILE = "file"  # File system access
    ADMIN = "admin"  # Administrative operations


@dataclass
class Permission:
    """A permission grant for a user/role."""

    # Who has the permission
    role: str  # Role name (e.g., "user", "admin", "developer")

    # What they can access
    tool_pattern: str  # Tool name or pattern (e.g., "search", "file_*", "*")
    scopes: set[ToolScope] = field(default_factory=lambda: {ToolScope.READ})

    # Restrictions
    rate_limit_per_minute: int | None = None
    rate_limit_per_hour: int | None = None

    # Validity
    expires_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def matches_tool(self, tool_name: str) -> bool:
        """Check if this permission applies to a tool."""
        import fnmatch

        return fnmatch.fnmatch(tool_name, self.tool_pattern)


@dataclass
class RateLimitState:
    """Tracks rate limit state for a user/tool combination."""

    minute_count: int = 0
    minute_reset: datetime = field(default_factory=datetime.utcnow)
    hour_count: int = 0
    hour_reset: datetime = field(default_factory=datetime.utcnow)

    def check_and_increment(
        self,
        minute_limit: int | None,
        hour_limit: int | None,
    ) -> tuple[bool, float | None]:
        """
        Check rate limit and increment if allowed.

        Returns (is_allowed, retry_after_seconds).
        """
        now = datetime.utcnow()

        # Reset counters if windows expired
        if now - self.minute_reset > timedelta(minutes=1):
            self.minute_count = 0
            self.minute_reset = now

        if now - self.hour_reset > timedelta(hours=1):
            self.hour_count = 0
            self.hour_reset = now

        # Check limits
        if minute_limit and self.minute_count >= minute_limit:
            retry_after = (self.minute_reset + timedelta(minutes=1) - now).total_seconds()
            return False, max(0, retry_after)

        if hour_limit and self.hour_count >= hour_limit:
            retry_after = (self.hour_reset + timedelta(hours=1) - now).total_seconds()
            return False, max(0, retry_after)

        # Increment counters
        self.minute_count += 1
        self.hour_count += 1

        return True, None


class PermissionManager:
    """
    Manages tool permissions with RBAC.

    Features:
    - Role-based access control
    - Pattern-based tool matching
    - Rate limiting
    - Permission caching
    """

    def __init__(self):
        # Role -> Permissions mapping
        self._role_permissions: dict[str, list[Permission]] = {}

        # User -> Roles mapping
        self._user_roles: dict[str, set[str]] = {}

        # Rate limit state: (user_id, tool_name) -> state
        self._rate_limits: dict[tuple[str, str], RateLimitState] = {}

        # Default permissions
        self._setup_defaults()

    def _setup_defaults(self) -> None:
        """Setup default role permissions."""
        # Default user role
        self.add_role_permission(
            "user",
            Permission(
                role="user",
                tool_pattern="*",
                scopes={ToolScope.READ, ToolScope.EXECUTE},
                rate_limit_per_minute=30,
                rate_limit_per_hour=500,
            ),
        )

        # Admin role
        self.add_role_permission(
            "admin",
            Permission(
                role="admin",
                tool_pattern="*",
                scopes=set(ToolScope),
            ),
        )

        # Read-only role
        self.add_role_permission(
            "readonly",
            Permission(
                role="readonly",
                tool_pattern="*",
                scopes={ToolScope.READ},
                rate_limit_per_minute=60,
            ),
        )

    def add_role_permission(self, role: str, permission: Permission) -> None:
        """Add a permission to a role."""
        if role not in self._role_permissions:
            self._role_permissions[role] = []
        self._role_permissions[role].append(permission)

    def assign_role(self, user_id: str, role: str) -> None:
        """Assign a role to a user."""
        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()
        self._user_roles[user_id].add(role)

    def revoke_role(self, user_id: str, role: str) -> None:
        """Revoke a role from a user."""
        if user_id in self._user_roles:
            self._user_roles[user_id].discard(role)

    def get_user_roles(self, user_id: str) -> set[str]:
        """Get all roles for a user."""
        return self._user_roles.get(user_id, {"user"})  # Default to "user"

    def get_user_permissions(self, user_id: str) -> list[Permission]:
        """Get all permissions for a user."""
        roles = self.get_user_roles(user_id)
        permissions = []

        for role in roles:
            role_perms = self._role_permissions.get(role, [])
            permissions.extend(p for p in role_perms if not p.is_expired)

        return permissions

    async def check_permission(
        self,
        user_id: str | None,
        tool_name: str,
        context: ExecutionContext,
        required_scopes: set[ToolScope] | None = None,
    ) -> bool:
        """
        Check if a user has permission to use a tool.

        Returns True if permission is granted.
        """
        # Anonymous users get minimal access
        if user_id is None:
            return self._check_anonymous_access(tool_name)

        # Get user permissions
        permissions = self.get_user_permissions(user_id)

        # Find matching permissions
        for perm in permissions:
            if perm.matches_tool(tool_name):
                # Check scopes if required
                if required_scopes:
                    if required_scopes.issubset(perm.scopes):
                        return True
                else:
                    return True

        return False

    def _check_anonymous_access(self, tool_name: str) -> bool:
        """Check if anonymous access is allowed for a tool."""
        # By default, anonymous users can only use safe tools
        safe_tools = {"search", "calculate", "get_weather", "get_time"}
        return tool_name in safe_tools

    async def check_rate_limit(
        self,
        user_id: str | None,
        tool_name: str,
    ) -> tuple[bool, float | None]:
        """
        Check if user is within rate limits.

        Returns (is_allowed, retry_after_seconds).
        """
        if user_id is None:
            # Anonymous rate limit
            return True, None  # Or implement stricter limits

        # Get applicable rate limits from permissions
        permissions = self.get_user_permissions(user_id)
        minute_limit = None
        hour_limit = None

        for perm in permissions:
            if perm.matches_tool(tool_name):
                if perm.rate_limit_per_minute:
                    if minute_limit is None:
                        minute_limit = perm.rate_limit_per_minute
                    else:
                        minute_limit = max(minute_limit, perm.rate_limit_per_minute)

                if perm.rate_limit_per_hour:
                    if hour_limit is None:
                        hour_limit = perm.rate_limit_per_hour
                    else:
                        hour_limit = max(hour_limit, perm.rate_limit_per_hour)

        if minute_limit is None and hour_limit is None:
            return True, None

        # Get or create rate limit state
        key = (user_id, tool_name)
        if key not in self._rate_limits:
            self._rate_limits[key] = RateLimitState()

        state = self._rate_limits[key]
        return state.check_and_increment(minute_limit, hour_limit)

    async def record_usage(
        self,
        user_id: str | None,
        tool_name: str,
    ) -> None:
        """
        Record tool usage.

        Note: This is called after successful execution.
        Rate limit check already incremented, so this is for
        additional tracking/logging.
        """
        # Could be extended for analytics, billing, etc.
        pass

    def clear_rate_limits(self, user_id: str | None = None) -> None:
        """Clear rate limit state."""
        if user_id:
            # Clear for specific user
            keys_to_remove = [k for k in self._rate_limits if k[0] == user_id]
            for key in keys_to_remove:
                del self._rate_limits[key]
        else:
            # Clear all
            self._rate_limits.clear()
