"""
Step Controller

Orchestrates execution of decomposed tasks.

Design decisions:
- Parallel execution where dependencies allow
- Progress tracking and reporting
- Error handling with retry/skip options
- Integration with checkpoints for recovery
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from src.core.types import ExecutionContext, Task, Message
from src.core.exceptions import PlanningError
from src.planning.decomposer import SubTask, SubTaskStatus


class PlanStatus(str, Enum):
    """Status of an execution plan."""
    
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StepResult:
    """Result of executing a single step."""
    
    subtask_id: UUID
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    messages: list[Message] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """
    An execution plan for a task.
    
    Contains all subtasks and tracks execution state.
    """
    
    id: UUID = field(default_factory=uuid4)
    
    # Original task
    task: Task = field(default_factory=lambda: Task())
    
    # Decomposed subtasks
    subtasks: list[SubTask] = field(default_factory=list)
    
    # Execution levels (parallel groups)
    levels: list[list[SubTask]] = field(default_factory=list)
    
    # State
    status: PlanStatus = PlanStatus.PENDING
    current_level: int = 0
    
    # Results
    results: dict[UUID, StepResult] = field(default_factory=dict)
    
    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    # Metadata
    context: ExecutionContext | None = None
    
    @property
    def completed_count(self) -> int:
        return sum(
            1 for st in self.subtasks
            if st.status == SubTaskStatus.COMPLETED
        )
    
    @property
    def total_count(self) -> int:
        return len(self.subtasks)
    
    @property
    def progress(self) -> float:
        if not self.subtasks:
            return 1.0
        return self.completed_count / self.total_count


@dataclass
class StepEvent:
    """Event emitted during step execution."""
    
    event_type: str  # "step_started", "step_completed", "step_failed", "plan_completed"
    plan_id: UUID
    subtask_id: UUID | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class StepExecutor:
    """
    Executes individual steps.
    
    Uses the agent's reasoning system to complete subtasks.
    """
    
    def __init__(self, reasoning_strategy):
        self._strategy = reasoning_strategy
    
    async def execute(
        self,
        subtask: SubTask,
        context: ExecutionContext,
        history: list[Message] | None = None,
    ) -> StepResult:
        """Execute a single subtask."""
        import time
        
        start_time = time.perf_counter()
        
        # Build prompt from subtask
        user_message = Message(
            role="user",
            content=f"Complete the following task:\n\n{subtask.description}\n\nObjective: {subtask.objective}",
        )
        
        messages = (history or []) + [user_message]
        
        try:
            # Execute using reasoning strategy
            response = await self._strategy.execute(messages, context)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return StepResult(
                subtask_id=subtask.id,
                success=True,
                result=response.content,
                duration_ms=duration_ms,
                messages=messages + [Message(role="assistant", content=response.content)],
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return StepResult(
                subtask_id=subtask.id,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
                messages=messages,
            )


class StepController:
    """
    Orchestrates plan execution.
    
    Features:
    - Level-by-level execution
    - Parallel execution within levels
    - Progress tracking
    - Error handling
    - Checkpoint integration
    """
    
    def __init__(
        self,
        step_executor: StepExecutor,
        checkpoint_manager: "CheckpointManager | None" = None,
        max_parallel: int = 3,
        retry_failed: bool = True,
        max_retries: int = 2,
    ):
        self._executor = step_executor
        self._checkpoints = checkpoint_manager
        self._max_parallel = max_parallel
        self._retry_failed = retry_failed
        self._max_retries = max_retries
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        context: ExecutionContext,
    ) -> AsyncIterator[StepEvent]:
        """
        Execute a plan, yielding events as steps complete.
        """
        plan.status = PlanStatus.RUNNING
        plan.started_at = datetime.utcnow()
        plan.context = context
        
        completed_ids: set[UUID] = set()
        retry_counts: dict[UUID, int] = {}
        
        # Process level by level
        for level_idx, level in enumerate(plan.levels):
            plan.current_level = level_idx
            
            # Execute subtasks in this level (with parallelism limit)
            tasks_to_run = [st for st in level if st.id not in completed_ids]
            
            while tasks_to_run:
                batch = tasks_to_run[:self._max_parallel]
                tasks_to_run = tasks_to_run[self._max_parallel:]
                
                # Run batch in parallel
                async for event in self._execute_batch(batch, plan, context):
                    yield event
                    
                    if event.event_type == "step_completed":
                        completed_ids.add(event.subtask_id)
                    
                    elif event.event_type == "step_failed":
                        # Handle failure
                        subtask_id = event.subtask_id
                        retry_count = retry_counts.get(subtask_id, 0)
                        
                        if self._retry_failed and retry_count < self._max_retries:
                            retry_counts[subtask_id] = retry_count + 1
                            # Re-add to queue
                            subtask = next(st for st in level if st.id == subtask_id)
                            tasks_to_run.append(subtask)
                        else:
                            # Mark as failed
                            plan.status = PlanStatus.FAILED
                            yield StepEvent(
                                event_type="plan_failed",
                                plan_id=plan.id,
                                subtask_id=subtask_id,
                                data={"reason": "max_retries_exceeded"},
                            )
                            return
            
            # Checkpoint after each level
            if self._checkpoints:
                await self._checkpoints.save(plan)
        
        # Plan completed
        plan.status = PlanStatus.COMPLETED
        plan.completed_at = datetime.utcnow()
        
        yield StepEvent(
            event_type="plan_completed",
            plan_id=plan.id,
            data={
                "total_steps": plan.total_count,
                "completed_steps": plan.completed_count,
            },
        )
    
    async def _execute_batch(
        self,
        batch: list[SubTask],
        plan: ExecutionPlan,
        context: ExecutionContext,
    ) -> AsyncIterator[StepEvent]:
        """Execute a batch of subtasks in parallel."""
        # Mark as running and emit start events
        for subtask in batch:
            subtask.status = SubTaskStatus.RUNNING
            yield StepEvent(
                event_type="step_started",
                plan_id=plan.id,
                subtask_id=subtask.id,
                data={"description": subtask.description},
            )
        
        # Build context from previous results
        history = self._build_history(plan)
        
        # Execute in parallel
        tasks = [
            self._executor.execute(subtask, context, history)
            for subtask in batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for subtask, result in zip(batch, results):
            if isinstance(result, Exception):
                subtask.status = SubTaskStatus.FAILED
                subtask.error = str(result)
                
                yield StepEvent(
                    event_type="step_failed",
                    plan_id=plan.id,
                    subtask_id=subtask.id,
                    data={"error": str(result)},
                )
            else:
                plan.results[subtask.id] = result
                
                if result.success:
                    subtask.status = SubTaskStatus.COMPLETED
                    subtask.result = result.result
                    
                    yield StepEvent(
                        event_type="step_completed",
                        plan_id=plan.id,
                        subtask_id=subtask.id,
                        data={"result": result.result[:200] if result.result else None},
                    )
                else:
                    subtask.status = SubTaskStatus.FAILED
                    subtask.error = result.error
                    
                    yield StepEvent(
                        event_type="step_failed",
                        plan_id=plan.id,
                        subtask_id=subtask.id,
                        data={"error": result.error},
                    )
    
    def _build_history(self, plan: ExecutionPlan) -> list[Message]:
        """Build conversation history from completed steps."""
        messages = []
        
        for subtask in plan.subtasks:
            if subtask.status == SubTaskStatus.COMPLETED and subtask.id in plan.results:
                result = plan.results[subtask.id]
                messages.extend(result.messages)
        
        return messages
    
    async def pause_plan(self, plan: ExecutionPlan) -> None:
        """Pause execution of a plan."""
        plan.status = PlanStatus.PAUSED
        
        if self._checkpoints:
            await self._checkpoints.save(plan)
    
    async def resume_plan(
        self,
        plan: ExecutionPlan,
        context: ExecutionContext,
    ) -> AsyncIterator[StepEvent]:
        """Resume execution of a paused plan."""
        if plan.status != PlanStatus.PAUSED:
            raise PlanningError(f"Cannot resume plan in state: {plan.status}")
        
        # Continue from where we left off
        async for event in self.execute_plan(plan, context):
            yield event
    
    async def cancel_plan(self, plan: ExecutionPlan) -> None:
        """Cancel execution of a plan."""
        plan.status = PlanStatus.CANCELLED
        plan.completed_at = datetime.utcnow()


# Avoid circular import
from src.planning.checkpoints import CheckpointManager
