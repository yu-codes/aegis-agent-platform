"""
Short-Term Memory

Manages recent conversation history within context limits.
Provides strategies for handling context overflow.

Design decisions:
- Multiple strategies: window, summarization, hybrid
- Strategy pattern for easy swapping
- Works with token limits, not just message counts
- Integrates with LLM for summarization
"""

from abc import ABC, abstractmethod
from typing import Any

from src.core.types import Message, MessageRole
from src.reasoning.llm.base import BaseLLMAdapter
from src.reasoning.prompts.template import get_prompt_registry


class ShortTermMemory(ABC):
    """
    Abstract base for short-term memory strategies.
    
    Short-term memory determines which messages to include
    in the LLM context. Different strategies balance between:
    - Context utilization
    - Computational cost
    - Information preservation
    """
    
    @abstractmethod
    async def process(
        self,
        messages: list[Message],
        max_tokens: int,
    ) -> list[Message]:
        """
        Process messages to fit within token limit.
        
        Args:
            messages: Full message history
            max_tokens: Maximum tokens allowed for messages
            
        Returns:
            Processed messages that fit the limit
        """
        pass
    
    @abstractmethod
    async def estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate token count for messages."""
        pass


class WindowMemory(ShortTermMemory):
    """
    Sliding window memory strategy.
    
    Keeps the N most recent messages that fit within the token limit.
    Simple and fast, but loses older context entirely.
    
    Pros:
    - Fast (no LLM calls needed)
    - Predictable behavior
    - Good for short conversations
    
    Cons:
    - Older context is completely lost
    - May miss important early context
    """
    
    def __init__(
        self,
        llm: BaseLLMAdapter | None = None,
        chars_per_token: float = 4.0,
    ):
        self._llm = llm
        self._chars_per_token = chars_per_token
    
    async def estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate tokens in messages."""
        if not messages:
            return 0
        
        # If we have an LLM, use its tokenizer
        if self._llm:
            total_text = "\n".join(
                f"{msg.role.value}: {msg.content}"
                for msg in messages
            )
            return await self._llm.count_tokens(total_text)
        
        # Otherwise, estimate from character count
        total_chars = sum(
            len(msg.content) + len(msg.role.value) + 2
            for msg in messages
        )
        return int(total_chars / self._chars_per_token)
    
    async def process(
        self,
        messages: list[Message],
        max_tokens: int,
    ) -> list[Message]:
        """Keep recent messages within token limit."""
        if not messages:
            return []
        
        # Always keep system message(s) if present
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in messages if m.role != MessageRole.SYSTEM]
        
        # Calculate tokens for system messages
        system_tokens = await self.estimate_tokens(system_messages)
        remaining_tokens = max_tokens - system_tokens
        
        if remaining_tokens <= 0:
            # Only room for system messages
            return system_messages
        
        # Add messages from most recent, going back
        result_messages = []
        current_tokens = 0
        
        for msg in reversed(other_messages):
            msg_tokens = await self.estimate_tokens([msg])
            if current_tokens + msg_tokens > remaining_tokens:
                break
            result_messages.insert(0, msg)
            current_tokens += msg_tokens
        
        return system_messages + result_messages


class SummarizingMemory(ShortTermMemory):
    """
    Summarizing memory strategy.
    
    When messages exceed the limit, older messages are
    summarized into a compact form. Preserves key information
    while reducing token count.
    
    Pros:
    - Preserves important context from older messages
    - Better for long conversations
    
    Cons:
    - Requires LLM calls for summarization
    - Summarization may lose nuance
    - Higher latency
    """
    
    def __init__(
        self,
        llm: BaseLLMAdapter,
        summary_threshold: float = 0.75,  # Trigger summary at 75% capacity
        recent_messages_to_keep: int = 4,  # Always keep these unsummarized
    ):
        self._llm = llm
        self._threshold = summary_threshold
        self._keep_recent = recent_messages_to_keep
        self._registry = get_prompt_registry()
    
    async def estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate tokens in messages."""
        if not messages:
            return 0
        
        total_text = "\n".join(
            f"{msg.role.value}: {msg.content}"
            for msg in messages
        )
        return await self._llm.count_tokens(total_text)
    
    async def _summarize_messages(self, messages: list[Message]) -> str:
        """Generate a summary of messages."""
        template = self._registry.get("conversation_summary")
        if not template:
            # Fallback to simple concatenation
            return "Previous context: " + " | ".join(
                f"{m.role.value}: {m.content[:100]}"
                for m in messages
            )
        
        prompt = template.render(messages=messages)
        
        response = await self._llm.complete(
            [Message(role=MessageRole.USER, content=prompt)],
            max_tokens=500,  # Keep summary concise
            temperature=0.3,  # More deterministic
        )
        
        return response.content or ""
    
    async def process(
        self,
        messages: list[Message],
        max_tokens: int,
    ) -> list[Message]:
        """Process messages with summarization if needed."""
        if not messages:
            return []
        
        # Check current token count
        current_tokens = await self.estimate_tokens(messages)
        threshold_tokens = int(max_tokens * self._threshold)
        
        if current_tokens <= threshold_tokens:
            # Under threshold, return as-is
            return messages
        
        # Separate system messages and others
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in messages if m.role != MessageRole.SYSTEM]
        
        if len(other_messages) <= self._keep_recent:
            # Not enough to summarize, use window
            window = WindowMemory(self._llm)
            return await window.process(messages, max_tokens)
        
        # Split into messages to summarize and messages to keep
        to_summarize = other_messages[:-self._keep_recent]
        to_keep = other_messages[-self._keep_recent:]
        
        # Generate summary
        summary = await self._summarize_messages(to_summarize)
        
        # Create summary message
        summary_message = Message(
            role=MessageRole.SYSTEM,
            content=f"[Conversation Summary]\n{summary}",
            metadata={"is_summary": True, "summarized_count": len(to_summarize)},
        )
        
        # Combine: system messages + summary + recent messages
        result = system_messages + [summary_message] + to_keep
        
        # Verify we're under the limit
        result_tokens = await self.estimate_tokens(result)
        if result_tokens > max_tokens:
            # Still too long, fall back to window
            window = WindowMemory(self._llm)
            return await window.process(result, max_tokens)
        
        return result


class HybridMemory(ShortTermMemory):
    """
    Hybrid memory strategy.
    
    Uses window memory for speed, falls back to summarization
    only when significant context would be lost.
    
    Strategy:
    1. Try window memory first
    2. If too many messages dropped, trigger summarization
    3. Cache summaries to avoid repeated computation
    """
    
    def __init__(
        self,
        llm: BaseLLMAdapter,
        drop_threshold: int = 10,  # Summarize if dropping more than this
    ):
        self._llm = llm
        self._drop_threshold = drop_threshold
        self._window = WindowMemory(llm)
        self._summarizing = SummarizingMemory(llm)
    
    async def estimate_tokens(self, messages: list[Message]) -> int:
        return await self._window.estimate_tokens(messages)
    
    async def process(
        self,
        messages: list[Message],
        max_tokens: int,
    ) -> list[Message]:
        """Process with hybrid strategy."""
        if not messages:
            return []
        
        # Try window first
        windowed = await self._window.process(messages, max_tokens)
        
        # Calculate how many messages were dropped
        dropped_count = len(messages) - len(windowed)
        
        if dropped_count <= self._drop_threshold:
            # Acceptable drop, use window result
            return windowed
        
        # Too much dropped, use summarization
        return await self._summarizing.process(messages, max_tokens)
