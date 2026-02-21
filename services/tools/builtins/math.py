"""
Math Tools

Mathematical computation tools.
"""

from dataclasses import dataclass
from typing import Any
import math
import operator


@dataclass
class CalculationResult:
    """Result of a calculation."""

    success: bool = False
    result: Any = None
    error: str | None = None
    expression: str = ""


class CalculatorTool:
    """
    Safe calculator tool.

    Evaluates mathematical expressions safely.
    """

    # Safe math functions
    SAFE_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "ceil": math.ceil,
        "floor": math.floor,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "radians": math.radians,
        "degrees": math.degrees,
    }

    # Safe constants
    SAFE_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
    }

    # Safe operators
    SAFE_OPERATORS = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "//": operator.floordiv,
        "%": operator.mod,
        "**": operator.pow,
    }

    def calculate(self, expression: str) -> CalculationResult:
        """
        Calculate a mathematical expression.

        Args:
            expression: Math expression (e.g., "2 + 2", "sqrt(16)")

        Returns:
            Calculation result
        """
        result = CalculationResult(expression=expression)

        try:
            # Create safe namespace
            namespace = {
                **self.SAFE_FUNCTIONS,
                **self.SAFE_CONSTANTS,
            }

            # Sanitize expression
            safe_expr = self._sanitize_expression(expression)

            # Evaluate
            value = eval(safe_expr, {"__builtins__": {}}, namespace)

            result.success = True
            result.result = value

        except ZeroDivisionError:
            result.error = "Division by zero"
        except ValueError as e:
            result.error = f"Math error: {str(e)}"
        except SyntaxError:
            result.error = "Invalid expression syntax"
        except Exception as e:
            result.error = f"Calculation error: {str(e)}"

        return result

    def _sanitize_expression(self, expression: str) -> str:
        """Sanitize expression for safe evaluation."""
        # Remove dangerous patterns
        dangerous = ["import", "exec", "eval", "__", "open", "file", "input", "print"]

        expr_lower = expression.lower()
        for pattern in dangerous:
            if pattern in expr_lower:
                raise ValueError(f"Dangerous pattern detected: {pattern}")

        return expression

    def convert_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> CalculationResult:
        """
        Convert between units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Conversion result
        """
        result = CalculationResult(expression=f"{value} {from_unit} to {to_unit}")

        # Unit conversion factors (to base unit)
        length_to_meters = {
            "m": 1.0,
            "km": 1000.0,
            "cm": 0.01,
            "mm": 0.001,
            "mi": 1609.344,
            "ft": 0.3048,
            "in": 0.0254,
            "yd": 0.9144,
        }

        weight_to_kg = {
            "kg": 1.0,
            "g": 0.001,
            "mg": 0.000001,
            "lb": 0.453592,
            "oz": 0.0283495,
        }

        temp_converters = {
            ("c", "f"): lambda x: x * 9 / 5 + 32,
            ("f", "c"): lambda x: (x - 32) * 5 / 9,
            ("c", "k"): lambda x: x + 273.15,
            ("k", "c"): lambda x: x - 273.15,
            ("f", "k"): lambda x: (x - 32) * 5 / 9 + 273.15,
            ("k", "f"): lambda x: (x - 273.15) * 9 / 5 + 32,
        }

        from_lower = from_unit.lower()
        to_lower = to_unit.lower()

        try:
            # Temperature
            if (from_lower, to_lower) in temp_converters:
                converted = temp_converters[(from_lower, to_lower)](value)
                result.success = True
                result.result = converted
                return result

            # Length
            if from_lower in length_to_meters and to_lower in length_to_meters:
                base = value * length_to_meters[from_lower]
                converted = base / length_to_meters[to_lower]
                result.success = True
                result.result = converted
                return result

            # Weight
            if from_lower in weight_to_kg and to_lower in weight_to_kg:
                base = value * weight_to_kg[from_lower]
                converted = base / weight_to_kg[to_lower]
                result.success = True
                result.result = converted
                return result

            result.error = f"Unknown unit conversion: {from_unit} to {to_unit}"

        except Exception as e:
            result.error = f"Conversion error: {str(e)}"

        return result

    def statistics(self, values: list[float]) -> dict[str, float]:
        """
        Calculate statistics for a list of values.

        Args:
            values: List of numbers

        Returns:
            Dict with statistics
        """
        if not values:
            return {}

        n = len(values)
        mean = sum(values) / n

        # Variance and std dev
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = variance**0.5

        # Median
        sorted_values = sorted(values)
        mid = n // 2
        if n % 2 == 0:
            median = (sorted_values[mid - 1] + sorted_values[mid]) / 2
        else:
            median = sorted_values[mid]

        return {
            "count": n,
            "sum": sum(values),
            "mean": mean,
            "median": median,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "variance": variance,
            "std_dev": std_dev,
        }
