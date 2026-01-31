"""
Structured Logging

JSON-structured logging with context propagation.

Design decisions:
- Structured JSON output
- Log level filtering
- Context enrichment
- Multiple handlers
"""

import contextvars
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, TextIO


class LogLevel(IntEnum):
    """Log levels matching Python's logging module."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogRecord:
    """A structured log record."""

    level: LogLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Context
    logger_name: str = "aegis"

    # Structured data
    data: dict[str, Any] = field(default_factory=dict)

    # Error info
    error: str | None = None
    error_type: str | None = None
    stack_trace: str | None = None

    # Tracing context
    trace_id: str | None = None
    span_id: str | None = None

    # Request context
    request_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "level": LogLevel(self.level).name,
            "logger": self.logger_name,
            "message": self.message,
        }

        if self.data:
            result["data"] = self.data

        if self.error:
            result["error"] = {
                "message": self.error,
                "type": self.error_type,
                "stack_trace": self.stack_trace,
            }

        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.request_id:
            result["request_id"] = self.request_id
        if self.user_id:
            result["user_id"] = self.user_id
        if self.session_id:
            result["session_id"] = self.session_id

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class LogHandler:
    """Base class for log handlers."""

    def __init__(self, level: LogLevel = LogLevel.DEBUG):
        self.level = level

    def should_handle(self, level: LogLevel) -> bool:
        """Check if this handler should process a log at this level."""
        return level >= self.level

    def handle(self, record: LogRecord) -> None:
        """Handle a log record."""
        pass


class ConsoleHandler(LogHandler):
    """Outputs logs to console."""

    def __init__(
        self,
        level: LogLevel = LogLevel.DEBUG,
        stream: TextIO | None = None,
        json_output: bool = True,
    ):
        super().__init__(level)
        self.stream = stream or sys.stderr
        self.json_output = json_output

    def handle(self, record: LogRecord) -> None:
        if not self.should_handle(record.level):
            return

        if self.json_output:
            output = record.to_json()
        else:
            # Human-readable format
            output = (
                f"[{record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{LogLevel(record.level).name:8s} {record.message}"
            )
            if record.data:
                output += f" | {record.data}"
            if record.error:
                output += f" | ERROR: {record.error}"

        print(output, file=self.stream)


class FileHandler(LogHandler):
    """Outputs logs to file."""

    def __init__(
        self,
        filename: str,
        level: LogLevel = LogLevel.DEBUG,
    ):
        super().__init__(level)
        self.filename = filename
        self._file = None

    def handle(self, record: LogRecord) -> None:
        if not self.should_handle(record.level):
            return

        if self._file is None:
            self._file = open(self.filename, "a")

        self._file.write(record.to_json() + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


class BufferHandler(LogHandler):
    """Buffers logs in memory for testing."""

    def __init__(self, level: LogLevel = LogLevel.DEBUG, max_records: int = 1000):
        super().__init__(level)
        self.records: list[LogRecord] = []
        self._max_records = max_records

    def handle(self, record: LogRecord) -> None:
        if not self.should_handle(record.level):
            return

        self.records.append(record)

        if len(self.records) > self._max_records:
            self.records = self.records[-self._max_records :]

    def clear(self) -> None:
        self.records.clear()


# Context variables for log enrichment
_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)


class StructuredLogger:
    """
    Main structured logging interface.

    Features:
    - JSON structured output
    - Context propagation
    - Multiple handlers
    - Level filtering
    """

    def __init__(
        self,
        name: str = "aegis",
        level: LogLevel = LogLevel.INFO,
        handlers: list[LogHandler] | None = None,
    ):
        self.name = name
        self.level = level
        self.handlers = handlers or [ConsoleHandler()]

    def _log(
        self,
        level: LogLevel,
        message: str,
        data: dict[str, Any] | None = None,
        error: Exception | None = None,
        **extra: Any,
    ) -> None:
        """Internal log method."""
        if level < self.level:
            return

        # Get context
        context = _log_context.get()

        # Build record
        record = LogRecord(
            level=level,
            message=message,
            logger_name=self.name,
            data={**(data or {}), **extra},
            trace_id=context.get("trace_id"),
            span_id=context.get("span_id"),
            request_id=context.get("request_id"),
            user_id=context.get("user_id"),
            session_id=context.get("session_id"),
        )

        # Add error info if present
        if error:
            import traceback

            record.error = str(error)
            record.error_type = type(error).__name__
            record.stack_trace = traceback.format_exc()

        # Send to handlers
        for handler in self.handlers:
            try:
                handler.handle(record)
            except Exception:
                pass  # Don't let logging errors affect main flow

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, error: Exception | None = None, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, message, error=error, **kwargs)

    def critical(self, message: str, error: Exception | None = None, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL, message, error=error, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log an exception at ERROR level."""
        import sys

        error = sys.exc_info()[1]
        self.error(message, error=error, **kwargs)

    @staticmethod
    @contextmanager
    def context(**kwargs: Any):
        """
        Context manager for adding context to logs.

        Usage:
            with logger.context(request_id="123", user_id="user1"):
                logger.info("Processing request")
        """
        current = _log_context.get()
        new_context = {**current, **kwargs}
        token = _log_context.set(new_context)

        try:
            yield
        finally:
            _log_context.reset(token)

    @staticmethod
    def set_context(**kwargs: Any) -> None:
        """Set context values that persist until changed."""
        current = _log_context.get()
        _log_context.set({**current, **kwargs})

    @staticmethod
    def clear_context() -> None:
        """Clear all context values."""
        _log_context.set({})


def get_logger(name: str = "aegis") -> StructuredLogger:
    """Get a logger instance."""
    return StructuredLogger(name=name)


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    json_output: bool = True,
    log_file: str | None = None,
) -> StructuredLogger:
    """Configure and return the default logger."""
    handlers: list[LogHandler] = [
        ConsoleHandler(level=level, json_output=json_output),
    ]

    if log_file:
        handlers.append(FileHandler(log_file, level=level))

    return StructuredLogger(level=level, handlers=handlers)
