"""
Task Decomposition

Breaks complex tasks into manageable subtasks.

Design decisions:
- LLM-powered decomposition for complex tasks
- Rule-based decomposition for common patterns
- Dependency graph for parallel execution
- Iterative refinement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from src.core.types import ExecutionContext, Task


class SubTaskStatus(str, Enum):
    """Status of a subtask."""
    
    PENDING = "pending"
    READY = "ready"  # Dependencies met
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubTask:
    """
    A decomposed subtask.
    
    Subtasks form a DAG for execution ordering.
    """
    
    id: UUID = field(default_factory=uuid4)
    
    # Task content
    description: str = ""
    objective: str = ""
    
    # Dependencies
    depends_on: list[UUID] = field(default_factory=list)
    
    # Execution hints
    suggested_tools: list[str] = field(default_factory=list)
    estimated_complexity: int = 1  # 1-5 scale
    
    # State
    status: SubTaskStatus = SubTaskStatus.PENDING
    result: Any = None
    error: str | None = None
    
    # Metadata
    index: int = 0  # Order in original list
    parent_task_id: UUID | None = None
    
    @property
    def is_terminal(self) -> bool:
        """Check if subtask is in a terminal state."""
        return self.status in {
            SubTaskStatus.COMPLETED,
            SubTaskStatus.FAILED,
            SubTaskStatus.SKIPPED,
        }
    
    def can_execute(self, completed_ids: set[UUID]) -> bool:
        """Check if all dependencies are met."""
        return all(dep in completed_ids for dep in self.depends_on)


class DecompositionStrategy(ABC):
    """Base class for decomposition strategies."""
    
    @abstractmethod
    async def decompose(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> list[SubTask]:
        """Decompose a task into subtasks."""
        pass


class SimpleDecomposer(DecompositionStrategy):
    """
    Simple rule-based decomposition.
    
    Uses pattern matching for common task types.
    """
    
    async def decompose(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> list[SubTask]:
        """
        Rule-based decomposition.
        
        Looks for keywords and patterns to split tasks.
        """
        description = task.description.lower()
        
        # Split by "and" or "then"
        if " and " in description or " then " in description:
            import re
            parts = re.split(r'\s+(?:and|then)\s+', task.description, flags=re.IGNORECASE)
            
            subtasks = []
            prev_id = None
            
            for i, part in enumerate(parts):
                subtask = SubTask(
                    description=part.strip(),
                    objective=part.strip(),
                    depends_on=[prev_id] if prev_id else [],
                    index=i,
                    parent_task_id=task.id,
                )
                subtasks.append(subtask)
                prev_id = subtask.id
            
            return subtasks
        
        # Single task - no decomposition needed
        return [SubTask(
            description=task.description,
            objective=task.objective or task.description,
            parent_task_id=task.id,
        )]


class LLMDecomposer(DecompositionStrategy):
    """
    LLM-powered task decomposition.
    
    Uses an LLM to intelligently break down complex tasks.
    """
    
    DECOMPOSITION_PROMPT = """You are a task planning expert. Break down the following task into clear, executable subtasks.

Task: {task_description}

Objective: {objective}

Available tools: {tools}

Instructions:
1. Identify distinct steps needed to complete the task
2. Order steps by dependency (what must be done first)
3. Suggest appropriate tools for each step
4. Keep each subtask atomic and actionable

Respond in JSON format:
{{
    "subtasks": [
        {{
            "description": "Clear description of what to do",
            "objective": "The goal of this step",
            "depends_on": [0, 1],  // Indices of subtasks this depends on
            "suggested_tools": ["tool_name"],
            "complexity": 1  // 1-5, with 5 being most complex
        }}
    ]
}}"""
    
    def __init__(self, llm_adapter):
        from src.reasoning.llm import BaseLLMAdapter
        self._llm: BaseLLMAdapter = llm_adapter
    
    async def decompose(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> list[SubTask]:
        """Use LLM to decompose the task."""
        from src.core.types import Message
        import json
        
        # Format prompt
        tools_str = ", ".join(context.allowed_tools) if context.allowed_tools else "any"
        
        prompt = self.DECOMPOSITION_PROMPT.format(
            task_description=task.description,
            objective=task.objective or task.description,
            tools=tools_str,
        )
        
        messages = [Message(role="user", content=prompt)]
        
        try:
            response = await self._llm.complete(messages)
            
            # Parse JSON response
            content = response.content
            
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            # Build subtasks with proper UUIDs
            subtasks = []
            uuid_map: dict[int, UUID] = {}
            
            for i, item in enumerate(data.get("subtasks", [])):
                subtask_id = uuid4()
                uuid_map[i] = subtask_id
                
                subtask = SubTask(
                    id=subtask_id,
                    description=item.get("description", ""),
                    objective=item.get("objective", item.get("description", "")),
                    suggested_tools=item.get("suggested_tools", []),
                    estimated_complexity=item.get("complexity", 1),
                    index=i,
                    parent_task_id=task.id,
                )
                subtasks.append(subtask)
            
            # Resolve dependencies
            for i, item in enumerate(data.get("subtasks", [])):
                dep_indices = item.get("depends_on", [])
                subtasks[i].depends_on = [
                    uuid_map[idx] for idx in dep_indices if idx in uuid_map
                ]
            
            return subtasks
            
        except Exception as e:
            # Fallback to single task
            return [SubTask(
                description=task.description,
                objective=task.objective or task.description,
                parent_task_id=task.id,
            )]


class TaskDecomposer:
    """
    Main decomposer that coordinates strategies.
    
    Chooses the appropriate decomposition strategy
    based on task complexity and configuration.
    """
    
    def __init__(
        self,
        llm_strategy: LLMDecomposer | None = None,
        simple_strategy: SimpleDecomposer | None = None,
        complexity_threshold: int = 50,  # Word count threshold
    ):
        self._llm_strategy = llm_strategy
        self._simple_strategy = simple_strategy or SimpleDecomposer()
        self._complexity_threshold = complexity_threshold
    
    async def decompose(
        self,
        task: Task,
        context: ExecutionContext,
        force_llm: bool = False,
    ) -> list[SubTask]:
        """
        Decompose a task into subtasks.
        
        Automatically chooses strategy based on complexity.
        """
        word_count = len(task.description.split())
        
        # Use LLM for complex tasks
        if force_llm or (self._llm_strategy and word_count > self._complexity_threshold):
            return await self._llm_strategy.decompose(task, context)
        
        # Use simple decomposer for simple tasks
        return await self._simple_strategy.decompose(task, context)
    
    def build_execution_order(self, subtasks: list[SubTask]) -> list[list[SubTask]]:
        """
        Build execution order respecting dependencies.
        
        Returns list of "levels" - subtasks at same level can run in parallel.
        """
        if not subtasks:
            return []
        
        # Build dependency graph
        id_to_task = {t.id: t for t in subtasks}
        completed: set[UUID] = set()
        levels: list[list[SubTask]] = []
        remaining = set(id_to_task.keys())
        
        while remaining:
            # Find all tasks that can run now
            ready = [
                id_to_task[tid] for tid in remaining
                if id_to_task[tid].can_execute(completed)
            ]
            
            if not ready:
                # Circular dependency or error
                break
            
            levels.append(ready)
            
            for task in ready:
                remaining.discard(task.id)
                completed.add(task.id)
        
        return levels
