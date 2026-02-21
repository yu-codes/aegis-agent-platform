"""
State Manager

Manages execution state and checkpoints.

Design decisions:
- Immutable state snapshots
- State persistence for recovery
- Event sourcing for audit trail
- Memory-efficient state diffing
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class StateStatus(str, Enum):
    """Overall state status."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    CHECKPOINTED = "checkpointed"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StateSnapshot:
    """Immutable snapshot of execution state."""

    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Execution info
    session_id: UUID | None = None
    request_id: str | None = None

    # State data
    status: StateStatus = StateStatus.INITIALIZED
    current_step: int = 0
    total_steps: int = 0

    # Context
    messages: list[dict] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)

    # Metrics
    tokens_used: int = 0
    tool_calls: int = 0
    iterations: int = 0

    # Error info
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "session_id": str(self.session_id) if self.session_id else None,
            "request_id": self.request_id,
            "status": self.status.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "messages": self.messages,
            "variables": self.variables,
            "tokens_used": self.tokens_used,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StateSnapshot":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else uuid4(),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else datetime.utcnow()
            ),
            session_id=UUID(data["session_id"]) if data.get("session_id") else None,
            request_id=data.get("request_id"),
            status=StateStatus(data.get("status", "initialized")),
            current_step=data.get("current_step", 0),
            total_steps=data.get("total_steps", 0),
            messages=data.get("messages", []),
            variables=data.get("variables", {}),
            tokens_used=data.get("tokens_used", 0),
            tool_calls=data.get("tool_calls", 0),
            iterations=data.get("iterations", 0),
            error=data.get("error"),
        )


@dataclass
class StateChange:
    """A change event for state."""

    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    change_type: str = ""
    old_value: Any = None
    new_value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StateManager:
    """
    Manages execution state.

    Provides state storage, snapshots, and recovery.
    """

    def __init__(self, max_history: int = 100):
        self._current_state: StateSnapshot | None = None
        self._history: list[StateSnapshot] = []
        self._changes: list[StateChange] = []
        self._max_history = max_history

    @property
    def current(self) -> StateSnapshot | None:
        return self._current_state

    @property
    def history(self) -> list[StateSnapshot]:
        return list(self._history)

    def initialize(
        self,
        session_id: UUID,
        request_id: str | None = None,
    ) -> StateSnapshot:
        """Initialize a new state."""
        self._current_state = StateSnapshot(
            session_id=session_id,
            request_id=request_id or str(uuid4()),
            status=StateStatus.INITIALIZED,
        )
        self._record_change("initialize", None, self._current_state)
        return self._current_state

    def update(
        self,
        status: StateStatus | None = None,
        current_step: int | None = None,
        total_steps: int | None = None,
        messages: list[dict] | None = None,
        variables: dict[str, Any] | None = None,
        tokens_used: int | None = None,
        tool_calls: int | None = None,
        iterations: int | None = None,
        error: str | None = None,
    ) -> StateSnapshot:
        """Update current state with new values."""
        if not self._current_state:
            raise RuntimeError("State not initialized")

        # Create new snapshot with updated values
        new_state = StateSnapshot(
            session_id=self._current_state.session_id,
            request_id=self._current_state.request_id,
            status=status if status is not None else self._current_state.status,
            current_step=(
                current_step if current_step is not None else self._current_state.current_step
            ),
            total_steps=total_steps if total_steps is not None else self._current_state.total_steps,
            messages=messages if messages is not None else self._current_state.messages,
            variables=variables if variables is not None else self._current_state.variables,
            tokens_used=tokens_used if tokens_used is not None else self._current_state.tokens_used,
            tool_calls=tool_calls if tool_calls is not None else self._current_state.tool_calls,
            iterations=iterations if iterations is not None else self._current_state.iterations,
            error=error if error is not None else self._current_state.error,
        )

        self._record_change("update", self._current_state, new_state)
        self._current_state = new_state

        return new_state

    def checkpoint(self) -> StateSnapshot:
        """Create a checkpoint of current state."""
        if not self._current_state:
            raise RuntimeError("State not initialized")

        # Save to history
        self._history.append(self._current_state)

        # Trim history if needed
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        # Update status
        return self.update(status=StateStatus.CHECKPOINTED)

    def restore(self, checkpoint_id: UUID) -> StateSnapshot | None:
        """Restore state from checkpoint."""
        for snapshot in reversed(self._history):
            if snapshot.id == checkpoint_id:
                self._current_state = StateSnapshot(
                    session_id=snapshot.session_id,
                    request_id=snapshot.request_id,
                    status=StateStatus.RUNNING,
                    current_step=snapshot.current_step,
                    total_steps=snapshot.total_steps,
                    messages=snapshot.messages.copy(),
                    variables=snapshot.variables.copy(),
                    tokens_used=snapshot.tokens_used,
                    tool_calls=snapshot.tool_calls,
                    iterations=snapshot.iterations,
                )
                self._record_change("restore", None, self._current_state)
                return self._current_state

        return None

    def get_checkpoint(self, index: int = -1) -> StateSnapshot | None:
        """Get checkpoint by index (default: latest)."""
        if not self._history:
            return None
        try:
            return self._history[index]
        except IndexError:
            return None

    def _record_change(
        self,
        change_type: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """Record a state change."""
        change = StateChange(
            change_type=change_type,
            old_value=(
                old_value.to_dict() if old_value and hasattr(old_value, "to_dict") else old_value
            ),
            new_value=(
                new_value.to_dict() if new_value and hasattr(new_value, "to_dict") else new_value
            ),
        )
        self._changes.append(change)

    def get_changes(self) -> list[StateChange]:
        """Get all recorded changes."""
        return list(self._changes)

    def export_state(self) -> dict:
        """Export full state for persistence."""
        return {
            "current": self._current_state.to_dict() if self._current_state else None,
            "history": [s.to_dict() for s in self._history],
        }

    def import_state(self, data: dict) -> None:
        """Import state from persisted data."""
        if data.get("current"):
            self._current_state = StateSnapshot.from_dict(data["current"])
        else:
            self._current_state = None

        self._history = [StateSnapshot.from_dict(s) for s in data.get("history", [])]
