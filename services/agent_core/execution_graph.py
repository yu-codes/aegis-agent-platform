"""
Execution Graph

DAG-based task execution engine.

Design decisions:
- Directed acyclic graph for dependencies
- Parallel execution where possible
- Topological sorting for ordering
- Cycle detection
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import UUID, uuid4


class NodeStatus(str, Enum):
    """Status of a graph node."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GraphNode:
    """A node in the execution graph."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""

    # Execution
    executor: Callable[..., Awaitable[Any]] | None = None
    arguments: dict[str, Any] = field(default_factory=dict)

    # Dependencies
    depends_on: list[UUID] = field(default_factory=list)

    # Status
    status: NodeStatus = NodeStatus.PENDING

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
class GraphExecutionResult:
    """Result of graph execution."""

    success: bool
    completed_nodes: list[UUID] = field(default_factory=list)
    failed_nodes: list[UUID] = field(default_factory=list)
    skipped_nodes: list[UUID] = field(default_factory=list)
    results: dict[UUID, Any] = field(default_factory=dict)
    errors: dict[UUID, str] = field(default_factory=dict)
    duration_ms: float = 0.0


class ExecutionGraph:
    """
    DAG-based execution engine.

    Executes tasks in dependency order with parallelism.
    """

    def __init__(self, max_parallel: int = 5):
        self._nodes: dict[UUID, GraphNode] = {}
        self._edges: dict[UUID, list[UUID]] = defaultdict(list)  # node -> dependents
        self._max_parallel = max_parallel

    def add_node(
        self,
        name: str,
        executor: Callable[..., Awaitable[Any]],
        arguments: dict[str, Any] | None = None,
        depends_on: list[UUID] | None = None,
    ) -> GraphNode:
        """Add a node to the graph."""
        node = GraphNode(
            name=name,
            executor=executor,
            arguments=arguments or {},
            depends_on=depends_on or [],
        )

        self._nodes[node.id] = node

        # Build edge map
        for dep_id in node.depends_on:
            self._edges[dep_id].append(node.id)

        return node

    def get_node(self, node_id: UUID) -> GraphNode | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def validate(self) -> tuple[bool, str | None]:
        """
        Validate the graph.

        Checks for:
        - Missing dependencies
        - Cycles

        Returns:
            (valid, error_message)
        """
        # Check for missing dependencies
        for node in self._nodes.values():
            for dep_id in node.depends_on:
                if dep_id not in self._nodes:
                    return False, f"Node {node.name} depends on missing node {dep_id}"

        # Check for cycles using DFS
        visited: set[UUID] = set()
        rec_stack: set[UUID] = set()

        def has_cycle(node_id: UUID) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for dep_id in self._edges[node_id]:
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in self._nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    return False, "Graph contains a cycle"

        return True, None

    def topological_sort(self) -> list[UUID]:
        """Get nodes in topological order."""
        in_degree: dict[UUID, int] = {nid: 0 for nid in self._nodes}

        for node in self._nodes.values():
            for dep_id in node.depends_on:
                if dep_id in in_degree:
                    in_degree[node.id] += 1

        # Nodes with no dependencies
        queue = [nid for nid, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for dependent_id in self._edges[node_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        return result

    def get_execution_levels(self) -> list[list[UUID]]:
        """Get nodes grouped by execution level."""
        levels: list[list[UUID]] = []
        remaining = set(self._nodes.keys())
        completed: set[UUID] = set()

        while remaining:
            level = []

            for node_id in list(remaining):
                node = self._nodes[node_id]
                # Check if all dependencies are completed
                if all(dep in completed for dep in node.depends_on):
                    level.append(node_id)

            if not level:
                # Break deadlock - should not happen with valid DAG
                level.append(next(iter(remaining)))

            for node_id in level:
                remaining.remove(node_id)
                completed.add(node_id)

            levels.append(level)

        return levels

    async def execute(
        self,
        context: dict[str, Any] | None = None,
    ) -> GraphExecutionResult:
        """
        Execute the graph.

        Executes nodes in dependency order with parallelism.
        """
        start_time = datetime.utcnow()

        # Validate first
        valid, error = self.validate()
        if not valid:
            return GraphExecutionResult(
                success=False,
                errors={UUID(int=0): error or "Validation failed"},
            )

        result = GraphExecutionResult(success=True)
        context = context or {}

        # Get execution levels
        levels = self.get_execution_levels()

        # Execute level by level
        for level_nodes in levels:
            # Execute nodes in this level in parallel
            semaphore = asyncio.Semaphore(self._max_parallel)

            async def execute_node(node_id: UUID) -> None:
                async with semaphore:
                    node = self._nodes[node_id]

                    # Check dependencies completed successfully
                    for dep_id in node.depends_on:
                        dep_node = self._nodes[dep_id]
                        if dep_node.status != NodeStatus.COMPLETED:
                            node.status = NodeStatus.SKIPPED
                            result.skipped_nodes.append(node_id)
                            return

                    # Execute node
                    node.status = NodeStatus.RUNNING
                    node.started_at = datetime.utcnow()

                    try:
                        # Inject dependency results
                        args = node.arguments.copy()
                        args["_context"] = context
                        args["_dependency_results"] = {
                            dep_id: self._nodes[dep_id].result for dep_id in node.depends_on
                        }

                        if node.executor:
                            node.result = await node.executor(**args)
                        else:
                            node.result = None

                        node.status = NodeStatus.COMPLETED
                        node.completed_at = datetime.utcnow()
                        result.completed_nodes.append(node_id)
                        result.results[node_id] = node.result

                    except Exception as e:
                        node.status = NodeStatus.FAILED
                        node.error = str(e)
                        node.completed_at = datetime.utcnow()
                        result.failed_nodes.append(node_id)
                        result.errors[node_id] = str(e)
                        result.success = False

            # Execute all nodes in this level
            await asyncio.gather(*[execute_node(nid) for nid in level_nodes])

        result.duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        return result

    def reset(self) -> None:
        """Reset all nodes to pending state."""
        for node in self._nodes.values():
            node.status = NodeStatus.PENDING
            node.result = None
            node.error = None
            node.started_at = None
            node.completed_at = None

    def clear(self) -> None:
        """Clear all nodes."""
        self._nodes.clear()
        self._edges.clear()
