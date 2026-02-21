"""
Permission Checker

Permission and access control.

Based on: src/safety/rbac.py, src/tools/permissions.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PermissionLevel(str, Enum):
    """Permission levels."""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class Permission:
    """A permission definition."""

    resource: str = ""
    action: str = ""
    level: PermissionLevel = PermissionLevel.READ

    # Constraints
    conditions: dict[str, Any] = field(default_factory=dict)
    expires_at: datetime | None = None

    def to_string(self) -> str:
        """Convert to string representation."""
        return f"{self.resource}:{self.action}:{self.level.value}"

    @classmethod
    def from_string(cls, s: str) -> "Permission":
        """Parse from string."""
        parts = s.split(":")
        return cls(
            resource=parts[0] if len(parts) > 0 else "",
            action=parts[1] if len(parts) > 1 else "*",
            level=PermissionLevel(parts[2]) if len(parts) > 2 else PermissionLevel.READ,
        )


@dataclass
class Role:
    """A role with permissions."""

    id: str = ""
    name: str = ""
    description: str = ""
    permissions: list[Permission] = field(default_factory=list)
    inherit_from: list[str] = field(default_factory=list)


@dataclass
class CheckResult:
    """Result of permission check."""

    allowed: bool = False
    reason: str = ""
    matching_permission: Permission | None = None


class PermissionChecker:
    """
    Permission checking service.

    Checks access permissions for resources and actions.
    """

    # Built-in roles
    BUILTIN_ROLES = {
        "guest": Role(
            id="guest",
            name="Guest",
            permissions=[
                Permission("chat", "read", PermissionLevel.READ),
                Permission("tools", "list", PermissionLevel.READ),
            ],
        ),
        "user": Role(
            id="user",
            name="User",
            permissions=[
                Permission("chat", "*", PermissionLevel.EXECUTE),
                Permission("tools", "execute", PermissionLevel.EXECUTE),
                Permission("memory", "read", PermissionLevel.READ),
                Permission("memory", "write", PermissionLevel.WRITE),
            ],
            inherit_from=["guest"],
        ),
        "admin": Role(
            id="admin",
            name="Administrator",
            permissions=[
                Permission("*", "*", PermissionLevel.ADMIN),
            ],
            inherit_from=["user"],
        ),
    }

    def __init__(self):
        self._roles: dict[str, Role] = dict(self.BUILTIN_ROLES)
        self._user_roles: dict[str, list[str]] = {}  # user_id -> role_ids

    def register_role(self, role: Role) -> None:
        """Register a custom role."""
        self._roles[role.id] = role

    def assign_role(self, user_id: str, role_id: str) -> bool:
        """Assign a role to a user."""
        if role_id not in self._roles:
            return False

        if user_id not in self._user_roles:
            self._user_roles[user_id] = []

        if role_id not in self._user_roles[user_id]:
            self._user_roles[user_id].append(role_id)

        return True

    def revoke_role(self, user_id: str, role_id: str) -> bool:
        """Revoke a role from a user."""
        if user_id not in self._user_roles:
            return False

        if role_id in self._user_roles[user_id]:
            self._user_roles[user_id].remove(role_id)
            return True

        return False

    def check(
        self,
        user_id: str,
        resource: str,
        action: str,
        required_level: PermissionLevel = PermissionLevel.EXECUTE,
    ) -> CheckResult:
        """
        Check if user has permission.

        Args:
            user_id: User ID
            resource: Resource to access
            action: Action to perform
            required_level: Minimum required level

        Returns:
            Check result
        """
        result = CheckResult()

        # Get all permissions for user
        permissions = self._get_user_permissions(user_id)

        if not permissions:
            result.reason = "No permissions found for user"
            return result

        # Level hierarchy
        level_hierarchy = {
            PermissionLevel.NONE: 0,
            PermissionLevel.READ: 1,
            PermissionLevel.WRITE: 2,
            PermissionLevel.EXECUTE: 3,
            PermissionLevel.ADMIN: 4,
        }

        required_value = level_hierarchy[required_level]

        for perm in permissions:
            # Check resource match
            if perm.resource != "*" and perm.resource != resource:
                continue

            # Check action match
            if perm.action != "*" and perm.action != action:
                continue

            # Check level
            perm_value = level_hierarchy.get(perm.level, 0)
            if perm_value >= required_value:
                # Check expiration
                if perm.expires_at and perm.expires_at < datetime.utcnow():
                    continue

                result.allowed = True
                result.matching_permission = perm
                result.reason = f"Allowed by permission: {perm.to_string()}"
                return result

        result.reason = (
            f"No matching permission for {resource}:{action} at level {required_level.value}"
        )
        return result

    def _get_user_permissions(self, user_id: str) -> list[Permission]:
        """Get all permissions for a user."""
        role_ids = self._user_roles.get(user_id, ["guest"])
        permissions = []
        visited_roles = set()

        def add_role_permissions(role_id: str):
            if role_id in visited_roles:
                return
            visited_roles.add(role_id)

            role = self._roles.get(role_id)
            if not role:
                return

            permissions.extend(role.permissions)

            # Add inherited permissions
            for parent_id in role.inherit_from:
                add_role_permissions(parent_id)

        for role_id in role_ids:
            add_role_permissions(role_id)

        return permissions

    def get_user_roles(self, user_id: str) -> list[Role]:
        """Get roles for a user."""
        role_ids = self._user_roles.get(user_id, ["guest"])
        return [self._roles[rid] for rid in role_ids if rid in self._roles]

    def list_roles(self) -> list[Role]:
        """List all roles."""
        return list(self._roles.values())

    def create_temporary_permission(
        self,
        resource: str,
        action: str,
        level: PermissionLevel,
        expires_in_seconds: int,
    ) -> Permission:
        """Create a temporary permission."""
        from datetime import timedelta

        return Permission(
            resource=resource,
            action=action,
            level=level,
            expires_at=datetime.utcnow() + timedelta(seconds=expires_in_seconds),
        )
