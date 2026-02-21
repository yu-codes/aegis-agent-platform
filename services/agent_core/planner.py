"""
Task Planner

Task decomposition and multi-step planning.

Design decisions:
- Hierarchical task decomposition
- Dependency-aware planning
- Dynamic replanning on failure
- Integration with execution graph

Based on: src/planning/decomposer.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4


class SubTaskStatus(str, Enum):
    """Status of a subtask."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SubTaskType(str, Enum):
    """Types of subtasks."""

    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    RAG_QUERY = "rag_query"
    VALIDATION = "validation"
    AGGREGATION = "aggregation"


@dataclass
class SubTask:
    """A single subtask in a plan."""

    id: UUID = field(default_factory=uuid4)
    description: str = ""
    task_type: SubTaskType = SubTaskType.REASONING

    # Dependencies
    depends_on: list[UUID] = field(default_factory=list)

    # Status
    status: SubTaskStatus = SubTaskStatus.PENDING

    # Execution details
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    query: str | None = None

    # Results
    result: Any = None
    error: str | None = None

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def duration_ms(self) -> float:
        if not self.started_at or not self.completed_at:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds() * 1000


@dataclass
class Task:
    """A high-level task to be decomposed."""

    id: UUID = field(default_factory=uuid4)
    description: str = ""
    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Plan:
    """An execution plan for a task."""

    id: UUID = field(default_factory=uuid4)
    task: Task = field(default_factory=Task)
    subtasks: list[SubTask] = field(default_factory=list)
    levels: list[list[SubTask]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_subtasks(self) -> int:
        return len(self.subtasks)

    @property
    def completed_subtasks(self) -> int:
        return sum(1 for st in self.subtasks if st.status == SubTaskStatus.COMPLETED)

    @property
    def progress(self) -> float:
        if not self.subtasks:
            return 1.0
        return self.completed_subtasks / self.total_subtasks


class LLMProtocol(Protocol):
    """Protocol for LLM calls in planning."""

    async def complete(self, messages: list[dict], **kwargs) -> dict: ...


class TaskPlanner:
    """
    Task decomposition and planning engine.

    Decomposes complex tasks into executable subtasks.
    """

    def __init__(
        self,
        llm: LLMProtocol | None = None,
        available_tools: list[str] | None = None,
    ):
        self._llm = llm
        self._available_tools = available_tools or []

    async def decompose(
        self,
        task: Task,
        max_subtasks: int = 10,
    ) -> Plan:
        """
        Decompose a task into subtasks.

        Args:
            task: Task to decompose
            max_subtasks: Maximum number of subtasks

        Returns:
            Execution plan with subtasks
        """
        if not self._llm:
            # Simple fallback without LLM
            return self._simple_decompose(task)

        # Use LLM for intelligent decomposition
        return await self._llm_decompose(task, max_subtasks)

    def _simple_decompose(self, task: Task) -> Plan:
        """Simple decomposition without LLM."""
        subtask = SubTask(
            description=task.description,
            task_type=SubTaskType.REASONING,
            status=SubTaskStatus.READY,
        )

        return Plan(
            task=task,
            subtasks=[subtask],
            levels=[[subtask]],
        )

    async def _llm_decompose(self, task: Task, max_subtasks: int) -> Plan:
        """LLM-based task decomposition."""
        tool_list = ", ".join(self._available_tools) if self._available_tools else "none"

        system_prompt = f"""You are a task planning assistant.
Decompose the given task into clear, actionable subtasks.

Available tools: {tool_list}

For each subtask, provide:
1. A clear description
2. The type (reasoning, tool_call, rag_query, validation, aggregation)
3. Dependencies on other subtasks (by number)
4. For tool_call: tool name and arguments

Respond in JSON format:
{{
    "subtasks": [
        {{
            "description": "...",
            "type": "reasoning|tool_call|rag_query|validation|aggregation",
            "depends_on": [],
            "tool_name": null,
            "tool_args": {{}}
        }}
    ]
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task.description}\nGoal: {task.goal}"},
        ]

        try:
            response = await self._llm.complete(messages, response_format={"type": "json_object"})
            content = response.get("content", "{}")

            import json

            data = json.loads(content)

            subtasks = []
            subtask_map: dict[int, SubTask] = {}

            for i, st_data in enumerate(data.get("subtasks", [])[:max_subtasks]):
                subtask = SubTask(
                    description=st_data.get("description", ""),
                    task_type=SubTaskType(st_data.get("type", "reasoning")),
                    tool_name=st_data.get("tool_name"),
                    tool_args=st_data.get("tool_args", {}),
                )
                subtasks.append(subtask)
                subtask_map[i] = subtask

            # Resolve dependencies
            for i, st_data in enumerate(data.get("subtasks", [])[:max_subtasks]):
                deps = st_data.get("depends_on", [])
                for dep_idx in deps:
                    if dep_idx in subtask_map:
                        subtasks[i].depends_on.append(subtask_map[dep_idx].id)

            # Mark tasks with no dependencies as ready
            for subtask in subtasks:
                if not subtask.depends_on:
                    subtask.status = SubTaskStatus.READY

            # Build execution levels
            levels = self._build_levels(subtasks)

            return Plan(
                task=task,
                subtasks=subtasks,
                levels=levels,
            )

        except Exception:
            # Fallback to simple decomposition
            return self._simple_decompose(task)

    def _build_levels(self, subtasks: list[SubTask]) -> list[list[SubTask]]:
        """Build execution levels from dependencies."""
        if not subtasks:
            return []

        # Track which subtasks have been assigned to levels
        assigned: set[UUID] = set()
        levels: list[list[SubTask]] = []

        while len(assigned) < len(subtasks):
            level: list[SubTask] = []

            for subtask in subtasks:
                if subtask.id in assigned:
                    continue

                # Check if all dependencies are assigned
                deps_satisfied = all(dep in assigned for dep in subtask.depends_on)
                if deps_satisfied:
                    level.append(subtask)

            if not level:
                # Break cycle - add remaining subtasks
                for subtask in subtasks:
                    if subtask.id not in assigned:
                        level.append(subtask)
                        break

            for subtask in level:
                assigned.add(subtask.id)

            levels.append(level)

        return levels

    async def replan(
        self,
        plan: Plan,
        failed_subtask: SubTask,
        error: str,
    ) -> Plan | None:
        """
        Create a new plan after failure.

        Args:
            plan: Original plan
            failed_subtask: The failed subtask
            error: Error message

        Returns:
            New plan or None if replanning not possible
        """
        if not self._llm:
            return None

        # Create a new task that accounts for the failure
        new_task = Task(
            description=f"Continue from: {failed_subtask.description}",
            goal=plan.task.goal,
            context={
                "original_plan": str(plan.id),
                "failed_step": failed_subtask.description,
                "error": error,
                "completed_steps": [
                    st.description for st in plan.subtasks if st.status == SubTaskStatus.COMPLETED
                ],
            },
        )

        # Decompose with awareness of failure
        return await self.decompose(new_task)
