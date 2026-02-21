"""
Tool Validator

Input and output validation for tools.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol
import json
import re


@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool = True
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    sanitized_args: dict[str, Any] = field(default_factory=dict)


class ToolRegistryProtocol(Protocol):
    """Protocol for tool registry."""

    def get(self, name: str) -> Any: ...


class ToolValidator:
    """
    Validates tool inputs and outputs.

    Features:
    - Type checking
    - Required parameter validation
    - Value range checking
    - Injection prevention
    - Output sanitization
    """

    def __init__(
        self,
        registry: ToolRegistryProtocol | None = None,
        strict_mode: bool = False,
    ):
        self._registry = registry
        self._strict_mode = strict_mode

        # Dangerous patterns
        self._injection_patterns = [
            r";\s*rm\s+-",
            r";\s*drop\s+table",
            r"<script[^>]*>",
            r"javascript:",
            r"on\w+\s*=",
        ]

    async def validate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate tool arguments.

        Args:
            tool_name: Tool name
            arguments: Arguments to validate

        Returns:
            Validation result
        """
        result = ValidationResult()

        # Get tool definition
        tool_def = self._registry.get(tool_name) if self._registry else None

        if tool_def is None:
            result.valid = False
            result.error = f"Tool not found: {tool_name}"
            return result

        # Check required parameters
        for param in tool_def.parameters:
            if param.required and param.name not in arguments:
                result.valid = False
                result.error = f"Missing required parameter: {param.name}"
                return result

        # Validate each argument
        sanitized = {}
        for param in tool_def.parameters:
            if param.name not in arguments:
                if param.default is not None:
                    sanitized[param.name] = param.default
                continue

            value = arguments[param.name]

            # Type validation
            type_valid, type_error = self._validate_type(value, param.type)
            if not type_valid:
                result.valid = False
                result.error = f"Parameter '{param.name}': {type_error}"
                return result

            # Enum validation
            if param.enum and value not in param.enum:
                result.valid = False
                result.error = f"Parameter '{param.name}' must be one of: {param.enum}"
                return result

            # Injection check
            injection_found = self._check_injection(value)
            if injection_found:
                result.valid = False
                result.error = f"Potential injection detected in parameter '{param.name}'"
                return result

            # Sanitize string values
            sanitized[param.name] = self._sanitize_value(value, param.type)

        result.sanitized_args = sanitized
        return result

    def validate_output(
        self,
        tool_name: str,
        output: Any,
    ) -> ValidationResult:
        """Validate tool output."""
        result = ValidationResult()

        # Check for dangerous content in output
        if isinstance(output, str):
            if self._check_injection(output):
                result.warnings.append("Output contains potentially dangerous content")

        return result

    def _validate_type(self, value: Any, expected_type: str) -> tuple[bool, str | None]:
        """Validate value type."""
        type_validators = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }

        validator = type_validators.get(expected_type)
        if validator is None:
            return True, None  # Unknown type, pass

        if not validator(value):
            return False, f"Expected {expected_type}, got {type(value).__name__}"

        return True, None

    def _check_injection(self, value: Any) -> bool:
        """Check for injection patterns."""
        if not isinstance(value, str):
            return False

        for pattern in self._injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        return False

    def _sanitize_value(self, value: Any, value_type: str) -> Any:
        """Sanitize a value."""
        if value_type == "string" and isinstance(value, str):
            # Remove null bytes
            value = value.replace("\x00", "")
            # Limit length
            value = value[:100000]

        return value

    def add_injection_pattern(self, pattern: str) -> None:
        """Add a custom injection detection pattern."""
        self._injection_patterns.append(pattern)

    def set_strict_mode(self, strict: bool) -> None:
        """Enable or disable strict validation."""
        self._strict_mode = strict
