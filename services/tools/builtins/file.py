"""
File Tools

File reading and writing tools.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FileOperationResult:
    """Result of a file operation."""

    success: bool = False
    content: str | None = None
    error: str | None = None
    path: str = ""
    size: int = 0


class FileReaderTool:
    """
    File reading tool.

    Reads files with safety restrictions.
    """

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        max_file_size: int = 1024 * 1024,  # 1MB
        allowed_extensions: list[str] | None = None,
    ):
        self._allowed_paths = [Path(p).resolve() for p in (allowed_paths or ["."])]
        self._max_size = max_file_size
        self._allowed_extensions = allowed_extensions or [
            ".txt",
            ".md",
            ".py",
            ".js",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".xml",
            ".html",
            ".css",
            ".toml",
            ".ini",
        ]

    def read(self, path: str) -> FileOperationResult:
        """
        Read a file.

        Args:
            path: File path to read

        Returns:
            File operation result
        """
        result = FileOperationResult(path=path)

        try:
            file_path = Path(path).resolve()

            # Security checks
            if not self._is_path_allowed(file_path):
                result.error = "Access denied: path not in allowed directories"
                return result

            if not self._is_extension_allowed(file_path):
                result.error = f"Access denied: extension not allowed"
                return result

            if not file_path.exists():
                result.error = "File not found"
                return result

            if not file_path.is_file():
                result.error = "Not a file"
                return result

            # Check file size
            size = file_path.stat().st_size
            if size > self._max_size:
                result.error = f"File too large: {size} bytes (max {self._max_size})"
                return result

            # Read file
            content = file_path.read_text(encoding="utf-8", errors="replace")

            result.success = True
            result.content = content
            result.size = len(content)

        except PermissionError:
            result.error = "Permission denied"
        except Exception as e:
            result.error = f"Error reading file: {str(e)}"

        return result

    async def read_async(self, path: str) -> FileOperationResult:
        """Async wrapper for read."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(None, self.read, path)

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is in allowed directories."""
        for allowed in self._allowed_paths:
            try:
                path.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False

    def _is_extension_allowed(self, path: Path) -> bool:
        """Check if file extension is allowed."""
        return path.suffix.lower() in self._allowed_extensions


class FileWriterTool:
    """
    File writing tool.

    Writes files with safety restrictions.
    """

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        max_file_size: int = 1024 * 1024,  # 1MB
        allowed_extensions: list[str] | None = None,
        create_directories: bool = True,
    ):
        self._allowed_paths = [Path(p).resolve() for p in (allowed_paths or ["."])]
        self._max_size = max_file_size
        self._allowed_extensions = allowed_extensions or [
            ".txt",
            ".md",
            ".py",
            ".js",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".xml",
            ".html",
            ".css",
            ".toml",
            ".ini",
        ]
        self._create_dirs = create_directories

    def write(
        self,
        path: str,
        content: str,
        append: bool = False,
    ) -> FileOperationResult:
        """
        Write to a file.

        Args:
            path: File path to write
            content: Content to write
            append: Append to file instead of overwrite

        Returns:
            File operation result
        """
        result = FileOperationResult(path=path)

        try:
            file_path = Path(path).resolve()

            # Security checks
            if not self._is_path_allowed(file_path):
                result.error = "Access denied: path not in allowed directories"
                return result

            if not self._is_extension_allowed(file_path):
                result.error = f"Access denied: extension not allowed"
                return result

            # Check content size
            if len(content) > self._max_size:
                result.error = f"Content too large: {len(content)} bytes (max {self._max_size})"
                return result

            # Create directory if needed
            if self._create_dirs and not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            mode = "a" if append else "w"
            file_path.write_text(content, encoding="utf-8")

            result.success = True
            result.size = len(content)

        except PermissionError:
            result.error = "Permission denied"
        except Exception as e:
            result.error = f"Error writing file: {str(e)}"

        return result

    async def write_async(
        self,
        path: str,
        content: str,
        append: bool = False,
    ) -> FileOperationResult:
        """Async wrapper for write."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, self.write, path, content, append
        )

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is in allowed directories."""
        for allowed in self._allowed_paths:
            try:
                path.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False

    def _is_extension_allowed(self, path: Path) -> bool:
        """Check if file extension is allowed."""
        return path.suffix.lower() in self._allowed_extensions
