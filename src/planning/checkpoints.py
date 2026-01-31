"""
Checkpoint Management

Saves and restores execution state for recovery.

Design decisions:
- Serializable plan state
- Multiple storage backends
- Incremental checkpointing
- Recovery from any checkpoint
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from src.planning.controller import ExecutionPlan


@dataclass
class Checkpoint:
    """
    A saved state of an execution plan.
    """

    id: UUID
    plan_id: UUID

    # State snapshot
    state: dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    level: int = 0
    completed_subtasks: list[str] = field(default_factory=list)

    # Context
    context_data: dict[str, Any] = field(default_factory=dict)


class CheckpointStorage(ABC):
    """Abstract storage backend for checkpoints."""

    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        pass

    @abstractmethod
    async def load(self, checkpoint_id: UUID) -> Checkpoint | None:
        """Load a checkpoint by ID."""
        pass

    @abstractmethod
    async def load_latest(self, plan_id: UUID) -> Checkpoint | None:
        """Load the latest checkpoint for a plan."""
        pass

    @abstractmethod
    async def list_checkpoints(self, plan_id: UUID) -> list[Checkpoint]:
        """List all checkpoints for a plan."""
        pass

    @abstractmethod
    async def delete(self, checkpoint_id: UUID) -> bool:
        """Delete a checkpoint."""
        pass


class InMemoryCheckpointStorage(CheckpointStorage):
    """In-memory checkpoint storage for testing."""

    def __init__(self):
        self._checkpoints: dict[UUID, Checkpoint] = {}
        self._by_plan: dict[UUID, list[UUID]] = {}

    async def save(self, checkpoint: Checkpoint) -> None:
        self._checkpoints[checkpoint.id] = checkpoint

        if checkpoint.plan_id not in self._by_plan:
            self._by_plan[checkpoint.plan_id] = []
        self._by_plan[checkpoint.plan_id].append(checkpoint.id)

    async def load(self, checkpoint_id: UUID) -> Checkpoint | None:
        return self._checkpoints.get(checkpoint_id)

    async def load_latest(self, plan_id: UUID) -> Checkpoint | None:
        checkpoint_ids = self._by_plan.get(plan_id, [])
        if not checkpoint_ids:
            return None

        # Get most recent
        checkpoints = [self._checkpoints[cid] for cid in checkpoint_ids]
        return max(checkpoints, key=lambda c: c.created_at)

    async def list_checkpoints(self, plan_id: UUID) -> list[Checkpoint]:
        checkpoint_ids = self._by_plan.get(plan_id, [])
        return [self._checkpoints[cid] for cid in checkpoint_ids]

    async def delete(self, checkpoint_id: UUID) -> bool:
        if checkpoint_id in self._checkpoints:
            cp = self._checkpoints.pop(checkpoint_id)
            if cp.plan_id in self._by_plan:
                self._by_plan[cp.plan_id] = [
                    cid for cid in self._by_plan[cp.plan_id] if cid != checkpoint_id
                ]
            return True
        return False


class RedisCheckpointStorage(CheckpointStorage):
    """Redis-based checkpoint storage."""

    def __init__(self, redis_client, prefix: str = "aegis:checkpoint:"):
        self._redis = redis_client
        self._prefix = prefix

    def _key(self, checkpoint_id: UUID) -> str:
        return f"{self._prefix}{checkpoint_id}"

    def _plan_key(self, plan_id: UUID) -> str:
        return f"{self._prefix}plan:{plan_id}"

    async def save(self, checkpoint: Checkpoint) -> None:
        # Serialize checkpoint
        data = {
            "id": str(checkpoint.id),
            "plan_id": str(checkpoint.plan_id),
            "state": checkpoint.state,
            "created_at": checkpoint.created_at.isoformat(),
            "level": checkpoint.level,
            "completed_subtasks": checkpoint.completed_subtasks,
            "context_data": checkpoint.context_data,
        }

        await self._redis.set(
            self._key(checkpoint.id),
            json.dumps(data),
        )

        # Add to plan index
        await self._redis.zadd(
            self._plan_key(checkpoint.plan_id),
            {str(checkpoint.id): checkpoint.created_at.timestamp()},
        )

    async def load(self, checkpoint_id: UUID) -> Checkpoint | None:
        data = await self._redis.get(self._key(checkpoint_id))
        if not data:
            return None

        return self._deserialize(json.loads(data))

    async def load_latest(self, plan_id: UUID) -> Checkpoint | None:
        # Get highest scored (most recent) checkpoint
        result = await self._redis.zrevrange(
            self._plan_key(plan_id),
            0,
            0,
        )

        if not result:
            return None

        return await self.load(UUID(result[0]))

    async def list_checkpoints(self, plan_id: UUID) -> list[Checkpoint]:
        checkpoint_ids = await self._redis.zrange(
            self._plan_key(plan_id),
            0,
            -1,
        )

        checkpoints = []
        for cid in checkpoint_ids:
            cp = await self.load(UUID(cid))
            if cp:
                checkpoints.append(cp)

        return checkpoints

    async def delete(self, checkpoint_id: UUID) -> bool:
        data = await self._redis.get(self._key(checkpoint_id))
        if not data:
            return False

        cp_data = json.loads(data)
        plan_id = cp_data["plan_id"]

        await self._redis.delete(self._key(checkpoint_id))
        await self._redis.zrem(self._plan_key(UUID(plan_id)), str(checkpoint_id))

        return True

    def _deserialize(self, data: dict) -> Checkpoint:
        return Checkpoint(
            id=UUID(data["id"]),
            plan_id=UUID(data["plan_id"]),
            state=data["state"],
            created_at=datetime.fromisoformat(data["created_at"]),
            level=data["level"],
            completed_subtasks=data["completed_subtasks"],
            context_data=data["context_data"],
        )


class CheckpointManager:
    """
    High-level checkpoint management.

    Provides:
    - Automatic checkpointing
    - Recovery from failures
    - Cleanup of old checkpoints
    """

    def __init__(
        self,
        storage: CheckpointStorage,
        max_checkpoints_per_plan: int = 10,
        auto_cleanup: bool = True,
    ):
        self._storage = storage
        self._max_checkpoints = max_checkpoints_per_plan
        self._auto_cleanup = auto_cleanup

    async def save(self, plan: "ExecutionPlan") -> Checkpoint:
        """Create a checkpoint from current plan state."""
        from uuid import uuid4

        # Serialize plan state
        state = {
            "status": plan.status.value,
            "current_level": plan.current_level,
            "subtasks": [
                {
                    "id": str(st.id),
                    "status": st.status.value,
                    "result": st.result,
                    "error": st.error,
                }
                for st in plan.subtasks
            ],
        }

        completed = [str(st.id) for st in plan.subtasks if st.status.value == "completed"]

        context_data = {}
        if plan.context:
            context_data = {
                "session_id": str(plan.context.session_id) if plan.context.session_id else None,
                "user_id": plan.context.user_id,
            }

        checkpoint = Checkpoint(
            id=uuid4(),
            plan_id=plan.id,
            state=state,
            level=plan.current_level,
            completed_subtasks=completed,
            context_data=context_data,
        )

        await self._storage.save(checkpoint)

        # Cleanup old checkpoints if needed
        if self._auto_cleanup:
            await self._cleanup_old_checkpoints(plan.id)

        return checkpoint

    async def restore(self, plan: "ExecutionPlan", checkpoint_id: UUID | None = None) -> bool:
        """
        Restore plan state from a checkpoint.

        If checkpoint_id is None, restores from latest.
        """
        if checkpoint_id:
            checkpoint = await self._storage.load(checkpoint_id)
        else:
            checkpoint = await self._storage.load_latest(plan.id)

        if not checkpoint:
            return False

        # Restore state
        state = checkpoint.state

        from src.planning.controller import PlanStatus
        from src.planning.decomposer import SubTaskStatus

        plan.status = PlanStatus(state["status"])
        plan.current_level = state["current_level"]

        # Restore subtask states
        subtask_states = {st["id"]: st for st in state.get("subtasks", [])}

        for subtask in plan.subtasks:
            if str(subtask.id) in subtask_states:
                st_state = subtask_states[str(subtask.id)]
                subtask.status = SubTaskStatus(st_state["status"])
                subtask.result = st_state.get("result")
                subtask.error = st_state.get("error")

        return True

    async def get_latest(self, plan_id: UUID) -> Checkpoint | None:
        """Get the latest checkpoint for a plan."""
        return await self._storage.load_latest(plan_id)

    async def list_checkpoints(self, plan_id: UUID) -> list[Checkpoint]:
        """List all checkpoints for a plan."""
        return await self._storage.list_checkpoints(plan_id)

    async def delete_checkpoint(self, checkpoint_id: UUID) -> bool:
        """Delete a specific checkpoint."""
        return await self._storage.delete(checkpoint_id)

    async def _cleanup_old_checkpoints(self, plan_id: UUID) -> None:
        """Remove old checkpoints exceeding the limit."""
        checkpoints = await self._storage.list_checkpoints(plan_id)

        if len(checkpoints) <= self._max_checkpoints:
            return

        # Sort by creation time, oldest first
        checkpoints.sort(key=lambda c: c.created_at)

        # Delete oldest checkpoints
        to_delete = len(checkpoints) - self._max_checkpoints
        for checkpoint in checkpoints[:to_delete]:
            await self._storage.delete(checkpoint.id)


# Import for type hints
