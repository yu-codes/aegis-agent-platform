"""
Logger

Structured logging implementation.

Based on: src/observability/logging.py
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable, TextIO


class LogLevel(IntEnum):
    """Log levels."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogEntry:
    """A structured log entry."""

    level: LogLevel = LogLevel.INFO
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Context
    logger_name: str = ""
    trace_id: str | None = None
    span_id: str | None = None

    # Structured data
    data: dict[str, Any] = field(default_factory=dict)

    # Error info
    error: str | None = None
    stack_trace: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        entry = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.name,
            "message": self.message,
            "logger": self.logger_name,
        }

        if self.trace_id:
            entry["trace_id"] = self.trace_id
        if self.span_id:
            entry["span_id"] = self.span_id
        if self.data:
            entry["data"] = self.data
        if self.error:
            entry["error"] = self.error
        if self.stack_trace:
            entry["stack_trace"] = self.stack_trace

        return entry

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class LogHandler:
    """Base log handler."""

    def handle(self, entry: LogEntry) -> None:
        """Handle a log entry."""
        raise NotImplementedError


class ConsoleHandler(LogHandler):
    """Console log handler."""

    def __init__(
        self,
        stream: TextIO = sys.stdout,
        format: str = "text",  # text, json
        colors: bool = True,
    ):
        self._stream = stream
        self._format = format
        self._colors = colors and stream.isatty()

        self._colors_map = {
            LogLevel.DEBUG: "\033[36m",  # Cyan
            LogLevel.INFO: "\033[32m",  # Green
            LogLevel.WARNING: "\033[33m",  # Yellow
            LogLevel.ERROR: "\033[31m",  # Red
            LogLevel.CRITICAL: "\033[35m",  # Magenta
        }
        self._reset = "\033[0m"

    def handle(self, entry: LogEntry) -> None:
        """Write log entry to console."""
        if self._format == "json":
            line = entry.to_json()
        else:
            line = self._format_text(entry)

        self._stream.write(line + "\n")
        self._stream.flush()

    def _format_text(self, entry: LogEntry) -> str:
        """Format as human-readable text."""
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        level_name = entry.level.name.ljust(8)

        if self._colors:
            color = self._colors_map.get(entry.level, "")
            level_name = f"{color}{level_name}{self._reset}"

        line = f"{timestamp} {level_name} [{entry.logger_name}] {entry.message}"

        if entry.data:
            line += f" | {entry.data}"

        if entry.error:
            line += f" | ERROR: {entry.error}"

        return line


class FileHandler(LogHandler):
    """File log handler."""

    def __init__(
        self,
        filepath: str,
        format: str = "json",
        max_size_mb: int = 100,
    ):
        self._filepath = filepath
        self._format = format
        self._max_size = max_size_mb * 1024 * 1024
        self._file: TextIO | None = None

    def handle(self, entry: LogEntry) -> None:
        """Write log entry to file."""
        if self._file is None:
            self._file = open(self._filepath, "a")

        if self._format == "json":
            line = entry.to_json()
        else:
            line = f"{entry.timestamp.isoformat()} {entry.level.name} {entry.message}"

        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close file."""
        if self._file:
            self._file.close()
            self._file = None


class Logger:
    """
    Structured logger.

    Provides structured logging with context propagation.
    """

    _loggers: dict[str, "Logger"] = {}
    _root_level: LogLevel = LogLevel.INFO
    _handlers: list[LogHandler] = []

    def __init__(self, name: str):
        self._name = name
        self._level: LogLevel | None = None
        self._context: dict[str, Any] = {}

    @classmethod
    def get_logger(cls, name: str) -> "Logger":
        """Get or create a logger."""
        if name not in cls._loggers:
            cls._loggers[name] = cls(name)
        return cls._loggers[name]

    @classmethod
    def configure(
        cls,
        level: LogLevel = LogLevel.INFO,
        handlers: list[LogHandler] | None = None,
    ) -> None:
        """Configure root logger."""
        cls._root_level = level
        if handlers:
            cls._handlers = handlers
        elif not cls._handlers:
            cls._handlers = [ConsoleHandler()]

    @classmethod
    def add_handler(cls, handler: LogHandler) -> None:
        """Add a handler."""
        cls._handlers.append(handler)

    def set_level(self, level: LogLevel) -> None:
        """Set logger level."""
        self._level = level

    def with_context(self, **kwargs: Any) -> "Logger":
        """Create a child logger with additional context."""
        child = Logger(self._name)
        child._level = self._level
        child._context = {**self._context, **kwargs}
        return child

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, error: Exception | None = None, **kwargs: Any) -> None:
        """Log error message."""
        import traceback

        error_str = None
        stack_trace = None

        if error:
            error_str = f"{type(error).__name__}: {str(error)}"
            stack_trace = traceback.format_exc()

        self._log(LogLevel.ERROR, message, error=error_str, stack_trace=stack_trace, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def _log(
        self,
        level: LogLevel,
        message: str,
        error: str | None = None,
        stack_trace: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Internal log method."""
        effective_level = self._level or self._root_level
        if level < effective_level:
            return

        # Get trace context
        trace_id = None
        span_id = None

        try:
            from services.observability.tracing import _current_span

            span = _current_span.get()
            if span:
                trace_id = str(span.context.trace_id)
                span_id = str(span.context.span_id)
        except Exception:
            pass

        entry = LogEntry(
            level=level,
            message=message,
            logger_name=self._name,
            trace_id=trace_id,
            span_id=span_id,
            data={**self._context, **kwargs},
            error=error,
            stack_trace=stack_trace,
        )

        for handler in self._handlers:
            try:
                handler.handle(entry)
            except Exception:
                pass


# Module-level convenience functions
def get_logger(name: str) -> Logger:
    """Get a logger."""
    return Logger.get_logger(name)


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    format: str = "text",
    json_output: bool = False,
) -> None:
    """Configure logging."""
    fmt = "json" if json_output else format
    Logger.configure(
        level=level,
        handlers=[ConsoleHandler(format=fmt)],
    )
