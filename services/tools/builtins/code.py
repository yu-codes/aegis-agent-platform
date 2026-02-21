"""
Code Tools

Code execution and analysis tools.
"""

from dataclasses import dataclass, field
from typing import Any
import io
import sys
import traceback


@dataclass
class CodeExecutionResult:
    """Result of code execution."""

    success: bool = False
    output: str = ""
    error: str | None = None
    return_value: Any = None
    execution_time_ms: float = 0.0


class CodeExecutorTool:
    """
    Safe code execution tool.

    Executes Python code in a sandboxed environment.
    """

    def __init__(
        self,
        allowed_modules: list[str] | None = None,
        max_execution_time: float = 5.0,
        max_output_length: int = 10000,
    ):
        self._allowed_modules = allowed_modules or [
            "math",
            "json",
            "re",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "operator",
        ]
        self._max_time = max_execution_time
        self._max_output = max_output_length

    def execute(self, code: str) -> CodeExecutionResult:
        """
        Execute Python code.

        Args:
            code: Python code to execute

        Returns:
            Execution result
        """
        import time

        result = CodeExecutionResult()
        start_time = time.time()

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Create restricted globals
        restricted_globals = self._create_restricted_globals()

        try:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Execute code
            exec(code, restricted_globals)

            result.success = True
            result.output = stdout_capture.getvalue()[: self._max_output]

            # Check for return value
            if "_result" in restricted_globals:
                result.return_value = restricted_globals["_result"]

        except Exception as e:
            result.error = f"{type(e).__name__}: {str(e)}"
            result.output = stderr_capture.getvalue()[: self._max_output]

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    async def execute_async(self, code: str) -> CodeExecutionResult:
        """Async wrapper for execution."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(None, self.execute, code)

    def _create_restricted_globals(self) -> dict:
        """Create restricted execution environment."""
        allowed_builtins = {
            "print",
            "len",
            "range",
            "enumerate",
            "zip",
            "map",
            "filter",
            "sorted",
            "reversed",
            "list",
            "dict",
            "set",
            "tuple",
            "str",
            "int",
            "float",
            "bool",
            "sum",
            "min",
            "max",
            "abs",
            "round",
            "isinstance",
            "type",
            "hasattr",
            "getattr",
            "any",
            "all",
        }

        restricted_builtins = {
            name: getattr(
                __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__, name
            )
            for name in allowed_builtins
            if hasattr(
                __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__, name
            )
        }

        # Add safe __import__
        def safe_import(name, *args):
            if name in self._allowed_modules:
                return __import__(name, *args)
            raise ImportError(f"Import of '{name}' is not allowed")

        restricted_builtins["__import__"] = safe_import

        return {"__builtins__": restricted_builtins}


class CodeAnalyzerTool:
    """
    Code analysis tool.

    Analyzes code for issues and patterns.
    """

    def analyze(
        self,
        code: str,
        language: str = "python",
    ) -> dict[str, Any]:
        """
        Analyze code.

        Args:
            code: Code to analyze
            language: Programming language

        Returns:
            Analysis results
        """
        if language != "python":
            return {"error": f"Unsupported language: {language}"}

        results = {
            "syntax_valid": True,
            "syntax_error": None,
            "warnings": [],
            "metrics": {},
            "imports": [],
            "functions": [],
            "classes": [],
        }

        # Check syntax
        try:
            import ast

            tree = ast.parse(code)
        except SyntaxError as e:
            results["syntax_valid"] = False
            results["syntax_error"] = f"Line {e.lineno}: {e.msg}"
            return results

        # Extract information
        results["imports"] = self._extract_imports(tree)
        results["functions"] = self._extract_functions(tree)
        results["classes"] = self._extract_classes(tree)
        results["metrics"] = self._calculate_metrics(code, tree)
        results["warnings"] = self._check_warnings(tree)

        return results

    def _extract_imports(self, tree) -> list[str]:
        """Extract import statements."""
        import ast

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append(module)
        return imports

    def _extract_functions(self, tree) -> list[dict]:
        """Extract function definitions."""
        import ast

        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(
                    {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "line": node.lineno,
                        "is_async": False,
                    }
                )
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.append(
                    {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "line": node.lineno,
                        "is_async": True,
                    }
                )
        return functions

    def _extract_classes(self, tree) -> list[dict]:
        """Extract class definitions."""
        import ast

        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(
                    {
                        "name": node.name,
                        "bases": [
                            base.id if isinstance(base, ast.Name) else str(base)
                            for base in node.bases
                        ],
                        "line": node.lineno,
                        "methods": [
                            n.name
                            for n in node.body
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        ],
                    }
                )
        return classes

    def _calculate_metrics(self, code: str, tree) -> dict:
        """Calculate code metrics."""
        import ast

        lines = code.split("\n")

        return {
            "total_lines": len(lines),
            "code_lines": len([l for l in lines if l.strip() and not l.strip().startswith("#")]),
            "comment_lines": len([l for l in lines if l.strip().startswith("#")]),
            "blank_lines": len([l for l in lines if not l.strip()]),
            "function_count": len(
                [
                    n
                    for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
            ),
            "class_count": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
        }

    def _check_warnings(self, tree) -> list[str]:
        """Check for common issues."""
        import ast

        warnings = []

        for node in ast.walk(tree):
            # Check for bare except
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                warnings.append(f"Bare except at line {node.lineno}")

            # Check for mutable default arguments
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        warnings.append(
                            f"Mutable default argument in {node.name} at line {node.lineno}"
                        )

        return warnings
