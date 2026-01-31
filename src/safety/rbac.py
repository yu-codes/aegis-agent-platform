"""
Role-Based Access Control (RBAC)

Fine-grained access control for the agent platform.

Design decisions:
- Role hierarchy
- Resource-based permissions
- Policy-based evaluation
- Tenant isolation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Permission(str, Enum):
    """System permissions."""
    
    # Session permissions
    SESSION_CREATE = "session:create"
    SESSION_READ = "session:read"
    SESSION_DELETE = "session:delete"
    
    # Tool permissions
    TOOL_EXECUTE = "tool:execute"
    TOOL_REGISTER = "tool:register"
    TOOL_ADMIN = "tool:admin"
    
    # Knowledge permissions
    KNOWLEDGE_READ = "knowledge:read"
    KNOWLEDGE_WRITE = "knowledge:write"
    KNOWLEDGE_DELETE = "knowledge:delete"
    
    # Admin permissions
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    USER_MANAGE = "user:manage"
    
    # Model permissions
    MODEL_USE = "model:use"
    MODEL_CONFIGURE = "model:configure"


@dataclass
class Role:
    """
    A role with associated permissions.
    """
    
    name: str
    description: str = ""
    
    # Permissions
    permissions: set[Permission] = field(default_factory=set)
    
    # Resource restrictions
    allowed_models: set[str] | None = None  # None = all
    allowed_tools: set[str] | None = None
    max_tokens_per_request: int | None = None
    
    # Role hierarchy
    inherits_from: list[str] = field(default_factory=list)
    
    # Metadata
    is_system: bool = False  # Cannot be modified
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class User:
    """A user in the system."""
    
    id: str
    email: str | None = None
    name: str | None = None
    
    # Roles
    roles: set[str] = field(default_factory=lambda: {"user"})
    
    # Tenant
    tenant_id: str | None = None
    
    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Custom attributes for policies
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourcePolicy:
    """
    A policy for resource access.
    
    Policies allow fine-grained control beyond role permissions.
    """
    
    name: str
    description: str = ""
    
    # Target
    resource_type: str = "*"  # e.g., "tool", "model", "session"
    resource_pattern: str = "*"  # e.g., "dangerous_*", specific name
    
    # Conditions
    conditions: dict[str, Any] = field(default_factory=dict)
    # Example conditions:
    # {"user.attributes.department": "engineering"}
    # {"time.hour": {"gte": 9, "lte": 17}}
    
    # Effect
    effect: str = "allow"  # "allow" or "deny"
    
    # Priority (higher = evaluated first)
    priority: int = 0


class RBACManager:
    """
    Manages roles, users, and permissions.
    """
    
    def __init__(self):
        self._roles: dict[str, Role] = {}
        self._users: dict[str, User] = {}
        
        self._setup_default_roles()
    
    def _setup_default_roles(self) -> None:
        """Create default system roles."""
        # Anonymous role
        self.add_role(Role(
            name="anonymous",
            description="Unauthenticated users",
            permissions={Permission.SESSION_CREATE},
            is_system=True,
        ))
        
        # Basic user role
        self.add_role(Role(
            name="user",
            description="Authenticated user",
            permissions={
                Permission.SESSION_CREATE,
                Permission.SESSION_READ,
                Permission.SESSION_DELETE,
                Permission.TOOL_EXECUTE,
                Permission.KNOWLEDGE_READ,
                Permission.MODEL_USE,
            },
            max_tokens_per_request=100000,
            is_system=True,
        ))
        
        # Developer role
        self.add_role(Role(
            name="developer",
            description="Developer with extended permissions",
            permissions={
                Permission.TOOL_REGISTER,
                Permission.KNOWLEDGE_WRITE,
                Permission.MODEL_CONFIGURE,
            },
            inherits_from=["user"],
            is_system=True,
        ))
        
        # Admin role
        self.add_role(Role(
            name="admin",
            description="Administrator",
            permissions=set(Permission),  # All permissions
            is_system=True,
        ))
    
    def add_role(self, role: Role) -> None:
        """Add a role."""
        self._roles[role.name] = role
    
    def get_role(self, name: str) -> Role | None:
        """Get a role by name."""
        return self._roles.get(name)
    
    def add_user(self, user: User) -> None:
        """Add a user."""
        self._users[user.id] = user
    
    def get_user(self, user_id: str) -> User | None:
        """Get a user by ID."""
        return self._users.get(user_id)
    
    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user."""
        user = self._users.get(user_id)
        if not user:
            return False
        
        if role_name not in self._roles:
            return False
        
        user.roles.add(role_name)
        return True
    
    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Revoke a role from a user."""
        user = self._users.get(user_id)
        if not user:
            return False
        
        user.roles.discard(role_name)
        return True
    
    def get_effective_permissions(self, user_id: str) -> set[Permission]:
        """
        Get all permissions for a user, including inherited.
        """
        user = self._users.get(user_id)
        if not user:
            # Anonymous permissions
            anon_role = self._roles.get("anonymous")
            return anon_role.permissions if anon_role else set()
        
        permissions: set[Permission] = set()
        processed_roles: set[str] = set()
        
        def collect_permissions(role_name: str):
            if role_name in processed_roles:
                return
            processed_roles.add(role_name)
            
            role = self._roles.get(role_name)
            if not role:
                return
            
            permissions.update(role.permissions)
            
            # Process inherited roles
            for parent in role.inherits_from:
                collect_permissions(parent)
        
        for role_name in user.roles:
            collect_permissions(role_name)
        
        return permissions
    
    def has_permission(
        self,
        user_id: str | None,
        permission: Permission,
    ) -> bool:
        """Check if a user has a specific permission."""
        if user_id is None:
            # Check anonymous permissions
            anon_role = self._roles.get("anonymous")
            if anon_role:
                return permission in anon_role.permissions
            return False
        
        permissions = self.get_effective_permissions(user_id)
        return permission in permissions
    
    def get_allowed_models(self, user_id: str) -> set[str] | None:
        """Get models allowed for a user."""
        user = self._users.get(user_id)
        if not user:
            return set()
        
        # Collect from all roles
        models: set[str] = set()
        has_restriction = False
        
        for role_name in user.roles:
            role = self._roles.get(role_name)
            if role:
                if role.allowed_models is None:
                    # No restriction = all models
                    return None
                has_restriction = True
                models.update(role.allowed_models)
        
        return models if has_restriction else None
    
    def get_allowed_tools(self, user_id: str) -> set[str] | None:
        """Get tools allowed for a user."""
        user = self._users.get(user_id)
        if not user:
            return set()
        
        tools: set[str] = set()
        has_restriction = False
        
        for role_name in user.roles:
            role = self._roles.get(role_name)
            if role:
                if role.allowed_tools is None:
                    return None
                has_restriction = True
                tools.update(role.allowed_tools)
        
        return tools if has_restriction else None


class PolicyEvaluator:
    """
    Evaluates resource policies.
    
    Policies provide fine-grained control that goes
    beyond simple role-based permissions.
    """
    
    def __init__(self, rbac: RBACManager):
        self._rbac = rbac
        self._policies: list[ResourcePolicy] = []
    
    def add_policy(self, policy: ResourcePolicy) -> None:
        """Add a policy."""
        self._policies.append(policy)
        # Keep sorted by priority (descending)
        self._policies.sort(key=lambda p: -p.priority)
    
    def evaluate(
        self,
        user_id: str | None,
        resource_type: str,
        resource_name: str,
        action: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Evaluate if an action is allowed.
        
        Returns True if allowed, False otherwise.
        """
        context = context or {}
        
        # Build evaluation context
        user = self._rbac.get_user(user_id) if user_id else None
        
        eval_context = {
            "user": {
                "id": user_id,
                "roles": list(user.roles) if user else ["anonymous"],
                "attributes": user.attributes if user else {},
            },
            "resource": {
                "type": resource_type,
                "name": resource_name,
            },
            "action": action,
            **context,
        }
        
        # Check policies in priority order
        for policy in self._policies:
            if not self._matches_resource(policy, resource_type, resource_name):
                continue
            
            if not self._matches_conditions(policy.conditions, eval_context):
                continue
            
            # Policy matches
            return policy.effect == "allow"
        
        # No matching policy - default to RBAC permissions
        return self._check_rbac_permission(user_id, resource_type, action)
    
    def _matches_resource(
        self,
        policy: ResourcePolicy,
        resource_type: str,
        resource_name: str,
    ) -> bool:
        """Check if policy matches the resource."""
        import fnmatch
        
        if policy.resource_type != "*" and policy.resource_type != resource_type:
            return False
        
        if policy.resource_pattern != "*":
            if not fnmatch.fnmatch(resource_name, policy.resource_pattern):
                return False
        
        return True
    
    def _matches_conditions(
        self,
        conditions: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        """Check if all conditions are met."""
        for key, expected in conditions.items():
            actual = self._get_nested(context, key)
            
            if isinstance(expected, dict):
                # Complex condition
                if not self._evaluate_complex_condition(actual, expected):
                    return False
            else:
                # Simple equality
                if actual != expected:
                    return False
        
        return True
    
    def _get_nested(self, obj: dict, path: str) -> Any:
        """Get nested value from dict using dot notation."""
        parts = path.split(".")
        current = obj
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def _evaluate_complex_condition(
        self,
        actual: Any,
        condition: dict[str, Any],
    ) -> bool:
        """Evaluate complex condition operators."""
        for op, value in condition.items():
            if op == "eq" and actual != value:
                return False
            if op == "ne" and actual == value:
                return False
            if op == "gt" and not (actual is not None and actual > value):
                return False
            if op == "gte" and not (actual is not None and actual >= value):
                return False
            if op == "lt" and not (actual is not None and actual < value):
                return False
            if op == "lte" and not (actual is not None and actual <= value):
                return False
            if op == "in" and actual not in value:
                return False
            if op == "nin" and actual in value:
                return False
            if op == "contains" and value not in (actual or []):
                return False
        
        return True
    
    def _check_rbac_permission(
        self,
        user_id: str | None,
        resource_type: str,
        action: str,
    ) -> bool:
        """Fall back to RBAC permission check."""
        # Map resource+action to permission
        permission_map = {
            ("tool", "execute"): Permission.TOOL_EXECUTE,
            ("tool", "register"): Permission.TOOL_REGISTER,
            ("session", "create"): Permission.SESSION_CREATE,
            ("session", "read"): Permission.SESSION_READ,
            ("session", "delete"): Permission.SESSION_DELETE,
            ("knowledge", "read"): Permission.KNOWLEDGE_READ,
            ("knowledge", "write"): Permission.KNOWLEDGE_WRITE,
            ("model", "use"): Permission.MODEL_USE,
        }
        
        permission = permission_map.get((resource_type, action))
        if permission:
            return self._rbac.has_permission(user_id, permission)
        
        return False
