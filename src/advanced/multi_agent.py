"""
Multi-Agent System

Orchestration of multiple specialized agents.

Design decisions:
- Agent specialization
- Message routing
- Collaborative workflows
- Conflict resolution
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from src.core.types import ExecutionContext, Message


class AgentRole(str, Enum):
    """Predefined agent roles."""

    COORDINATOR = "coordinator"  # Orchestrates other agents
    RESEARCHER = "researcher"  # Gathers information
    ANALYST = "analyst"  # Analyzes data
    WRITER = "writer"  # Generates content
    CRITIC = "critic"  # Reviews and critiques
    EXECUTOR = "executor"  # Takes actions
    SPECIALIST = "specialist"  # Domain expert


@dataclass
class AgentDefinition:
    """
    Definition of an agent in a multi-agent system.
    """

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    role: AgentRole = AgentRole.SPECIALIST

    # Configuration
    system_prompt: str = ""
    model: str | None = None
    temperature: float = 0.7

    # Capabilities
    allowed_tools: list[str] = field(default_factory=list)
    can_delegate: bool = False

    # Metadata
    description: str = ""
    expertise: list[str] = field(default_factory=list)


@dataclass
class AgentMessage:
    """Message between agents."""

    id: UUID = field(default_factory=uuid4)

    # Routing
    from_agent: UUID | None = None
    to_agent: UUID | None = None  # None = broadcast

    # Content
    content: str = ""
    message_type: str = "message"  # message, request, response, error

    # Context
    in_reply_to: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Base class for agents in a multi-agent system."""

    def __init__(self, definition: AgentDefinition):
        self.definition = definition
        self._inbox: list[AgentMessage] = []

    @property
    def id(self) -> UUID:
        return self.definition.id

    @property
    def name(self) -> str:
        return self.definition.name

    @abstractmethod
    async def process(
        self,
        message: AgentMessage,
        context: ExecutionContext,
    ) -> AgentMessage | None:
        """Process an incoming message."""
        pass

    async def receive(self, message: AgentMessage) -> None:
        """Receive a message into inbox."""
        self._inbox.append(message)

    def has_messages(self) -> bool:
        """Check if there are pending messages."""
        return len(self._inbox) > 0

    def get_next_message(self) -> AgentMessage | None:
        """Get next message from inbox."""
        if self._inbox:
            return self._inbox.pop(0)
        return None


class LLMAgent(Agent):
    """
    Agent powered by an LLM.

    Uses the reasoning system for processing.
    """

    def __init__(
        self,
        definition: AgentDefinition,
        llm_adapter,
        tool_executor=None,
    ):
        super().__init__(definition)
        self._llm = llm_adapter
        self._tools = tool_executor

    async def process(
        self,
        message: AgentMessage,
        context: ExecutionContext,
    ) -> AgentMessage | None:
        """Process message using LLM."""
        # Build prompt
        system = (
            self.definition.system_prompt
            or f"You are {self.definition.name}, a {self.definition.role.value}."
        )

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=message.content),
        ]

        # Get response from LLM
        response = await self._llm.complete(messages)

        return AgentMessage(
            from_agent=self.id,
            to_agent=message.from_agent,
            content=response.content,
            message_type="response",
            in_reply_to=message.id,
        )


class AgentPool:
    """
    Pool of available agents.

    Manages agent lifecycle and discovery.
    """

    def __init__(self):
        self._agents: dict[UUID, Agent] = {}
        self._by_role: dict[AgentRole, list[UUID]] = {role: [] for role in AgentRole}
        self._by_name: dict[str, UUID] = {}

    def register(self, agent: Agent) -> None:
        """Register an agent in the pool."""
        self._agents[agent.id] = agent
        self._by_role[agent.definition.role].append(agent.id)
        self._by_name[agent.name] = agent.id

    def unregister(self, agent_id: UUID) -> bool:
        """Remove an agent from the pool."""
        if agent_id in self._agents:
            agent = self._agents.pop(agent_id)
            self._by_role[agent.definition.role].remove(agent_id)
            del self._by_name[agent.name]
            return True
        return False

    def get(self, agent_id: UUID) -> Agent | None:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def get_by_name(self, name: str) -> Agent | None:
        """Get agent by name."""
        agent_id = self._by_name.get(name)
        if agent_id:
            return self._agents.get(agent_id)
        return None

    def get_by_role(self, role: AgentRole) -> list[Agent]:
        """Get all agents with a role."""
        return [self._agents[aid] for aid in self._by_role.get(role, []) if aid in self._agents]

    def list_all(self) -> list[Agent]:
        """List all agents."""
        return list(self._agents.values())


class RoutingStrategy(ABC):
    """Strategy for routing messages between agents."""

    @abstractmethod
    def route(
        self,
        message: AgentMessage,
        pool: AgentPool,
    ) -> list[UUID]:
        """Determine which agents should receive a message."""
        pass


class RoleBasedRouting(RoutingStrategy):
    """Route based on agent roles."""

    def __init__(self, role_map: dict[str, AgentRole] | None = None):
        self._role_map = role_map or {}

    def route(
        self,
        message: AgentMessage,
        pool: AgentPool,
    ) -> list[UUID]:
        # If specific target, return that
        if message.to_agent:
            return [message.to_agent]

        # Route based on message type
        message_type = message.metadata.get("target_role")
        if message_type:
            role = self._role_map.get(message_type) or AgentRole.SPECIALIST
            agents = pool.get_by_role(role)
            return [a.id for a in agents]

        # Broadcast to all
        return [a.id for a in pool.list_all()]


class ContentBasedRouting(RoutingStrategy):
    """Route based on message content analysis."""

    def __init__(self, keywords: dict[str, list[UUID]] | None = None):
        self._keywords = keywords or {}

    def route(
        self,
        message: AgentMessage,
        pool: AgentPool,
    ) -> list[UUID]:
        if message.to_agent:
            return [message.to_agent]

        content_lower = message.content.lower()
        matched_agents = []

        for keyword, agent_ids in self._keywords.items():
            if keyword in content_lower:
                matched_agents.extend(agent_ids)

        if matched_agents:
            return list(set(matched_agents))

        # Default to coordinator
        coordinators = pool.get_by_role(AgentRole.COORDINATOR)
        if coordinators:
            return [coordinators[0].id]

        return []


class AgentOrchestrator:
    """
    Orchestrates multi-agent workflows.

    Features:
    - Message routing
    - Turn management
    - Conflict resolution
    - Workflow coordination
    """

    def __init__(
        self,
        pool: AgentPool,
        routing_strategy: RoutingStrategy | None = None,
        max_turns: int = 10,
    ):
        self._pool = pool
        self._routing = routing_strategy or RoleBasedRouting()
        self._max_turns = max_turns

        # Conversation history
        self._history: list[AgentMessage] = []

    async def run(
        self,
        initial_message: str,
        context: ExecutionContext,
    ) -> AsyncIterator[AgentMessage]:
        """
        Run a multi-agent conversation.

        Yields messages as they're generated.
        """
        # Create initial message
        message = AgentMessage(
            content=initial_message,
            message_type="request",
        )

        self._history.append(message)
        yield message

        # Route to initial agents
        target_ids = self._routing.route(message, self._pool)

        for agent_id in target_ids:
            agent = self._pool.get(agent_id)
            if agent:
                await agent.receive(message)

        # Process turns
        for _turn in range(self._max_turns):
            # Collect responses from all agents with pending messages
            responses = []

            for agent in self._pool.list_all():
                while agent.has_messages():
                    msg = agent.get_next_message()
                    if msg:
                        response = await agent.process(msg, context)
                        if response:
                            responses.append(response)
                            self._history.append(response)
                            yield response

            if not responses:
                break

            # Route responses to next agents
            for response in responses:
                target_ids = self._routing.route(response, self._pool)

                for agent_id in target_ids:
                    if agent_id != response.from_agent:  # Don't send to self
                        agent = self._pool.get(agent_id)
                        if agent:
                            await agent.receive(response)

    async def run_workflow(
        self,
        workflow: list[dict[str, Any]],
        context: ExecutionContext,
    ) -> list[AgentMessage]:
        """
        Run a predefined workflow.

        Workflow is a list of steps:
        [
            {"agent": "researcher", "input": "Find information about X"},
            {"agent": "analyst", "input": "{previous_result}"},
            ...
        ]
        """
        results = []
        previous_result = ""

        for step in workflow:
            agent_name = step.get("agent")
            input_template = step.get("input", "")

            # Substitute previous result
            input_text = input_template.replace("{previous_result}", previous_result)

            # Get agent
            agent = self._pool.get_by_name(agent_name)
            if not agent:
                continue

            # Create and process message
            message = AgentMessage(
                to_agent=agent.id,
                content=input_text,
            )

            response = await agent.process(message, context)

            if response:
                results.append(response)
                previous_result = response.content

        return results

    def get_history(self) -> list[AgentMessage]:
        """Get conversation history."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
