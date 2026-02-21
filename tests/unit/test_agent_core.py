"""
Unit Tests - Agent Core

Tests for agent orchestration components.
"""

import pytest
from dataclasses import dataclass


class TestAgentOrchestrator:
    """Tests for AgentOrchestrator."""

    def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized."""
        from services.agent_core import AgentOrchestrator

        orchestrator = AgentOrchestrator()
        assert orchestrator is not None

    @pytest.mark.asyncio
    async def test_orchestrator_run_basic(self):
        """Test basic orchestrator run."""
        from services.agent_core import AgentOrchestrator

        orchestrator = AgentOrchestrator()
        # Note: This would need mocked LLM in real tests
        # result = await orchestrator.run(message="Hello")
        # assert result.response is not None


class TestPlanner:
    """Tests for TaskPlanner."""

    def test_planner_initialization(self):
        """Test planner can be initialized."""
        from services.agent_core import TaskPlanner

        planner = TaskPlanner()
        assert planner is not None

    @pytest.mark.asyncio
    async def test_plan_creation(self):
        """Test plan creation."""
        from services.agent_core import TaskPlanner

        planner = TaskPlanner()
        # plan = await planner.create_plan("Analyze this data")
        # assert len(plan.steps) > 0


class TestStateManager:
    """Tests for StateManager."""

    def test_state_manager_initialization(self):
        """Test state manager can be initialized."""
        from services.agent_core import StateManager

        state_manager = StateManager()
        assert state_manager is not None

    @pytest.mark.asyncio
    async def test_state_create(self):
        """Test state creation."""
        from services.agent_core import StateManager

        state_manager = StateManager()
        state = await state_manager.create()

        assert state.status.value == "initialized"

    @pytest.mark.asyncio
    async def test_state_snapshot(self):
        """Test state snapshot."""
        from services.agent_core import StateManager

        state_manager = StateManager()
        state = await state_manager.create()

        snapshot = state_manager.snapshot(state)
        assert snapshot is not None
        assert snapshot.id == state.id


class TestExecutionGraph:
    """Tests for ExecutionGraph."""

    def test_execution_graph_creation(self):
        """Test execution graph can be created."""
        from services.agent_core import ExecutionGraph

        graph = ExecutionGraph()
        assert graph is not None

    def test_add_node(self):
        """Test adding nodes to graph."""
        from services.agent_core.execution_graph import ExecutionGraph, ExecutionNode

        graph = ExecutionGraph()
        node = ExecutionNode(id="node1", type="llm_call")
        graph.add_node(node)

        assert "node1" in [n.id for n in graph.nodes.values()]


class TestReflection:
    """Tests for ReflectionEngine."""

    def test_reflection_initialization(self):
        """Test reflection engine initialization."""
        from services.agent_core import ReflectionEngine

        engine = ReflectionEngine()
        assert engine is not None

    @pytest.mark.asyncio
    async def test_response_evaluation(self):
        """Test response evaluation."""
        from services.agent_core import ReflectionEngine

        engine = ReflectionEngine()
        # result = await engine.evaluate(
        #     query="What is 2+2?",
        #     response="The answer is 4.",
        # )
        # assert result.score > 0
