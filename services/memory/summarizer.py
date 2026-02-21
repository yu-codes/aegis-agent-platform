"""
Summarizer

Conversation and memory summarization.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class SummaryConfig:
    """Configuration for summarization."""

    max_summary_length: int = 500
    summary_model: str = "claude-3-haiku"
    include_key_points: bool = True
    include_entities: bool = True
    include_decisions: bool = True


@dataclass
class Summary:
    """A summary result."""

    content: str = ""
    key_points: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    token_reduction: float = 0.0  # Original tokens / Summary tokens


class LLMProtocol(Protocol):
    """Protocol for LLM services."""

    async def complete(self, prompt: str, max_tokens: int = 500) -> str: ...


class Summarizer:
    """
    Conversation and memory summarizer.

    Uses LLM to create concise summaries.
    """

    def __init__(
        self,
        llm: LLMProtocol | None = None,
        config: SummaryConfig | None = None,
    ):
        self._llm = llm
        self._config = config or SummaryConfig()

    async def summarize_conversation(
        self,
        messages: list[dict],
        context: str | None = None,
    ) -> Summary:
        """
        Summarize a conversation.

        Args:
            messages: List of messages with 'role' and 'content'
            context: Additional context

        Returns:
            Summary of the conversation
        """
        if not messages:
            return Summary()

        # Calculate original token count (rough estimate)
        original_tokens = sum(len(m.get("content", "")) // 4 for m in messages)

        if self._llm:
            summary = await self._llm_summarize(messages, context)
        else:
            summary = self._extractive_summarize(messages)

        # Calculate reduction
        summary_tokens = len(summary.content) // 4
        if summary_tokens > 0:
            summary.token_reduction = original_tokens / summary_tokens

        return summary

    async def _llm_summarize(
        self,
        messages: list[dict],
        context: str | None = None,
    ) -> Summary:
        """Summarize using LLM."""
        # Format conversation
        conversation = "\n".join(
            [f"{m.get('role', 'unknown').upper()}: {m.get('content', '')}" for m in messages]
        )

        prompt = f"""Summarize this conversation concisely.

{f'Context: {context}' if context else ''}

Conversation:
{conversation}

Provide:
1. A brief summary (max {self._config.max_summary_length} chars)
2. Key points (bullet list)
3. Important entities/topics mentioned
4. Any decisions or conclusions reached

Format as:
SUMMARY: <your summary>
KEY_POINTS:
- point 1
- point 2
ENTITIES: entity1, entity2, entity3
DECISIONS:
- decision 1"""

        response = await self._llm.complete(prompt, max_tokens=800)

        return self._parse_summary_response(response)

    def _extractive_summarize(self, messages: list[dict]) -> Summary:
        """Simple extractive summarization fallback."""
        # Get user questions
        user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]

        # Get assistant responses
        assistant_messages = [
            m.get("content", "") for m in messages if m.get("role") == "assistant"
        ]

        # Extract key sentences
        key_points = []
        for msg in assistant_messages[:3]:
            sentences = msg.split(". ")
            if sentences:
                key_points.append(
                    sentences[0] + "." if not sentences[0].endswith(".") else sentences[0]
                )

        # Build summary
        summary_parts = []
        if user_messages:
            summary_parts.append(f"User asked about: {user_messages[0][:100]}...")
        if assistant_messages:
            summary_parts.append(f"Assistant responded with information about the topic.")

        return Summary(
            content=" ".join(summary_parts)[: self._config.max_summary_length],
            key_points=key_points[:5],
            entities=self._extract_entities(messages),
            decisions=[],
        )

    def _parse_summary_response(self, response: str) -> Summary:
        """Parse LLM summary response."""
        summary = Summary()

        lines = response.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("SUMMARY:"):
                summary.content = line.replace("SUMMARY:", "").strip()
                current_section = None
            elif line.startswith("KEY_POINTS:"):
                current_section = "key_points"
            elif line.startswith("ENTITIES:"):
                entities = line.replace("ENTITIES:", "").strip()
                summary.entities = [e.strip() for e in entities.split(",") if e.strip()]
                current_section = None
            elif line.startswith("DECISIONS:"):
                current_section = "decisions"
            elif line.startswith("- "):
                item = line[2:].strip()
                if current_section == "key_points":
                    summary.key_points.append(item)
                elif current_section == "decisions":
                    summary.decisions.append(item)

        return summary

    def _extract_entities(self, messages: list[dict]) -> list[str]:
        """Simple entity extraction."""
        text = " ".join(m.get("content", "") for m in messages)
        words = text.split()

        # Find capitalized words that might be entities
        entities = set()
        for word in words:
            cleaned = word.strip(".,!?()[]{}\"'")
            if cleaned and cleaned[0].isupper() and len(cleaned) > 2:
                entities.add(cleaned)

        return list(entities)[:10]

    async def summarize_memories(
        self,
        memories: list[dict],
    ) -> Summary:
        """Summarize a collection of memories."""
        if not memories:
            return Summary()

        content = "\n".join([f"- {m.get('content', '')}" for m in memories])

        if self._llm:
            prompt = f"""Consolidate these memory entries into a coherent summary:

{content}

Provide a concise summary that captures the key information."""

            response = await self._llm.complete(prompt, max_tokens=500)
            return Summary(content=response)

        # Fallback
        return Summary(
            content=f"Collection of {len(memories)} memories covering various topics.",
            key_points=[m.get("content", "")[:50] + "..." for m in memories[:5]],
        )
