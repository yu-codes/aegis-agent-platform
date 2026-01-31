# Aegis System Integration Hardening

**Date**: 2026-01-31  
**Author**: Principal Engineer  
**Purpose**: Validate, harden, and clarify the end-to-end execution flow

---

## 1. Canonical Agent Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CANONICAL EXECUTION FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

   HTTP Request
        │
        ▼
┌───────────────────┐
│   API Layer       │  1. Parse request, validate input
│   (FastAPI)       │  2. Apply middleware chain (trace, auth, rate-limit)
└────────┬──────────┘  3. Route to handler
         │
         ▼
┌───────────────────┐
│  Session Layer    │  4. Load or create session
│  (SessionManager) │  5. Retrieve conversation history
└────────┬──────────┘  6. Build ExecutionContext
         │
         ▼
┌───────────────────┐
│  AgentRuntime     │  7. SINGLE orchestration point (NEW)
│  [ORCHESTRATOR]   │  8. Owns the execution lifecycle
└────────┬──────────┘
         │
    ┌────┴────────────────────────┐
    ▼                             ▼
┌─────────────┐           ┌─────────────────┐
│ Safety      │◄─────────►│  Planning       │
│ (Input)     │           │  (Decomposer)   │
└─────┬───────┘           └────────┬────────┘
      │                            │
      ▼                            ▼
┌───────────────────────────────────────────┐
│              Reasoning Loop               │
│  ┌─────────────────────────────────────┐  │
│  │  9. Build prompt (RAG + Memory)     │  │
│  │  10. LLM call                       │  │
│  │  11. Parse response                 │  │
│  │  12. Execute tools (if any)         │  │
│  │  13. Loop until terminal            │  │
│  └─────────────────────────────────────┘  │
└───────────────────┬───────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────┐   ┌──────────┐   ┌──────────────┐
│ RAG     │   │ Memory   │   │ Tools        │
│(Context)│   │(History) │   │(Execution)   │
└─────────┘   └──────────┘   └──────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│              Output Path                   │
│  14. Apply guardrails                      │
│  15. Persist to session                    │
│  16. Emit response (stream or batch)       │
└───────────────────┬───────────────────────┘
                    │
                    ▼
   HTTP Response (SSE stream or JSON)
```

---

## 2. Sequence Diagram: Control and Data Flow

```
┌──────────┐ ┌─────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐ ┌─────────┐ ┌───────┐ ┌─────────┐
│  Client  │ │   API   │ │  Session    │ │AgentRuntime │ │Reasoning │ │ Memory  │ │  RAG  │ │  Tools  │
└────┬─────┘ └────┬────┘ └──────┬──────┘ └──────┬──────┘ └────┬─────┘ └────┬────┘ └───┬───┘ └────┬────┘
     │            │             │               │             │            │          │          │
     │ POST /chat │             │               │             │            │          │          │
     │───────────►│             │               │             │            │          │          │
     │            │             │               │             │            │          │          │
     │            │ get_session │               │             │            │          │          │
     │            │────────────►│               │             │            │          │          │
     │            │             │               │             │            │          │          │
     │            │   Session   │               │             │            │          │          │
     │            │◄────────────│               │             │            │          │          │
     │            │             │               │             │            │          │          │
     │            │ build_context               │             │            │          │          │
     │            │─────────────────────────────┤             │            │          │          │
     │            │             │               │             │            │          │          │
     │            │        run(message, ctx)    │             │            │          │          │
     │            │────────────────────────────►│             │            │          │          │
     │            │             │               │             │            │          │          │
     │            │             │               │ validate_input           │          │          │
     │            │             │               │─────────────────────────►│          │          │
     │            │             │               │             │            │          │          │
     │            │             │               │ get_history │            │          │          │
     │            │             │               │────────────────────────────────────►│          │
     │            │             │               │             │            │          │          │
     │            │             │               │         retrieve_context │          │          │
     │            │             │               │─────────────────────────────────────►          │
     │            │             │               │             │            │          │          │
     │            │             │               │  reason(messages, tools) │          │          │
     │            │             │               │────────────►│            │          │          │
     │            │             │               │             │            │          │          │
     │            │             │               │             │  LLM.complete         │          │
     │            │             │               │             │───────────►│          │          │
     │            │             │               │             │            │          │          │
     │            │             │               │             │◄───────────│          │          │
     │            │             │               │             │            │          │          │
     │            │             │               │             │  execute_tool         │          │
     │            │             │               │             │──────────────────────────────────►
     │            │             │               │             │            │          │          │
     │            │             │               │             │◄──────────────────────────────────
     │            │             │               │             │            │          │          │
     │            │             │               │   result    │            │          │          │
     │            │             │               │◄────────────│            │          │          │
     │            │             │               │             │            │          │          │
     │            │             │  apply_guardrails          │            │          │          │
     │            │             │               │─────────────────────────►│          │          │
     │            │             │               │             │            │          │          │
     │            │             │  save_to_session           │            │          │          │
     │            │             │◄──────────────│             │            │          │          │
     │            │             │               │             │            │          │          │
     │            │   AgentResult               │             │            │          │          │
     │            │◄────────────────────────────│             │            │          │          │
     │            │             │               │             │            │          │          │
     │  Response  │             │               │             │            │          │          │
     │◄───────────│             │               │             │            │          │          │
     │            │             │               │             │            │          │          │
```

### Ownership at Each Step

| Step | Component | Owns | Never Touches |
|------|-----------|------|---------------|
| 1-3 | API Layer | HTTP parsing, middleware, routing | LLM, tools, memory internals |
| 4-6 | Session | Session state, message history | LLM configuration, tool execution |
| 7-8 | **AgentRuntime** | Execution lifecycle, orchestration | HTTP, persistence implementation |
| 9 | RAG + Memory | Context assembly | LLM calls, tool execution |
| 10-11 | Reasoning | LLM interaction, response parsing | Persistence, HTTP |
| 12 | Tools | Tool execution, sandboxing | Session state, LLM |
| 14 | Safety | Output filtering | Everything else |
| 15-16 | Session/API | Persistence, response serialization | LLM, reasoning |

---

## 3. Identified Issues

### 3.1 Circular Dependencies

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| CD-1 | `tools/executor.py` → `reasoning/strategies/base.py` | 🟡 Medium | ToolExecutor imports `ToolExecutor` protocol from reasoning. Creates tight coupling. |
| CD-2 | `memory/long_term.py` → `reasoning/prompts` | 🟡 Medium | Memory module imports prompt templates for summarization. |
| CD-3 | `planning/controller.py` → `reasoning` | 🟢 Low | Controller uses reasoning, but this is expected one-way dependency. |

### 3.2 Implicit Shared State

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| SS-1 | `reasoning/prompts/template.py` | 🔴 High | Global singleton `_registry` with non-thread-safe lazy init |
| SS-2 | `api/app.py` | 🟡 Medium | `app.state.components` is mutable dict passed around |
| SS-3 | `memory/session.py` | 🟡 Medium | Redis client lazy-initialized per instance, not injected |

### 3.3 Leaky Abstractions

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| LA-1 | `api/routes/chat.py` | 🔴 High | Route handler has placeholder logic; no real orchestration |
| LA-2 | `api/dependencies.py` | 🔴 High | `get_agent()` is empty placeholder; no unified agent entry point |
| LA-3 | `planning/controller.py` | 🟡 Medium | `StepExecutor` directly calls `strategy.execute()` which doesn't exist on base |
| LA-4 | `tools/executor.py` | 🟡 Medium | Returns `ToolResult` but base schema differs from `core/types.py` |

### 3.4 Missing Glue Logic

| Issue | Location | Description |
|-------|----------|-------------|
| GL-1 | API → Reasoning | No component bridges chat route to reasoning strategy |
| GL-2 | Reasoning → RAG | No integration point for RAG context injection |
| GL-3 | Reasoning → Safety | Guardrails not wired into response path |
| GL-4 | Session → Memory | Short-term memory not synchronized with session |

---

## 4. Proposed AgentRuntime Orchestration Boundary

### 4.1 What AgentRuntime OWNS

```python
class AgentRuntime:
    """
    SINGLE orchestration point for agent execution.
    
    Responsibilities:
    - Execution lifecycle management
    - Component coordination (reasoning, memory, tools, RAG)
    - Safety enforcement (input validation, output guardrails)
    - Event emission for observability
    - Error handling and recovery
    """
    
    # OWNS these concerns:
    - Execution state machine (IDLE → THINKING → EXECUTING → COMPLETED)
    - Timeout enforcement at execution level
    - Tool call budget tracking
    - Token budget tracking
    - Iteration limiting
    - Context assembly (memory + RAG → prompt)
    - Response post-processing (guardrails)
    - Trace span lifecycle
```

### 4.2 What AgentRuntime MUST NEVER KNOW

```
- HTTP/transport layer details
- Session persistence implementation (Redis vs in-memory)
- LLM provider specifics (OpenAI vs Anthropic internals)
- Vector store implementation (FAISS vs Milvus)
- Tool implementation details (only interface)
- Authentication/authorization (receives already-validated context)
```

### 4.3 Dependency Injection Contract

```python
@dataclass
class AgentDependencies:
    """All dependencies injected into AgentRuntime."""
    
    llm: BaseLLMAdapter              # Required
    tool_executor: ToolExecutor       # Required
    memory: MemoryManager | None      # Optional
    retriever: Retriever | None       # Optional
    input_validator: InputValidator   # Required
    output_guardrails: GuardrailChain # Required
    tracer: Tracer | None             # Optional
    metrics: MetricsCollector | None  # Optional
```

---

## 5. Concrete Refactors Required

### 5.1 Extract ToolExecutor Protocol to Core

**Current**: `reasoning/strategies/base.py` defines `ToolExecutor` protocol  
**Problem**: Creates circular dependency tools → reasoning  
**Fix**: Move protocol to `core/interfaces.py`

```python
# src/core/interfaces.py (NEW FILE)
from abc import ABC, abstractmethod
from typing import Any, Protocol
from src.core.types import ExecutionContext, ToolResult

class ToolExecutorProtocol(Protocol):
    """Interface for tool execution."""
    
    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult: ...
    
    def get_tool_definitions(
        self,
        context: ExecutionContext,
    ) -> list[dict[str, Any]]: ...
```

### 5.2 Create AgentRuntime Module

**File**: `src/runtime/__init__.py`, `src/runtime/agent.py`

```python
# src/runtime/agent.py
class AgentRuntime:
    async def run(
        self,
        message: str,
        session: Session,
        context: ExecutionContext,
    ) -> AgentResult:
        """Execute a single turn."""
        
    async def run_stream(
        self,
        message: str,
        session: Session,
        context: ExecutionContext,
    ) -> AsyncIterator[AgentEvent]:
        """Execute with streaming."""
```

### 5.3 Fix ToolResult Schema Inconsistency

**Current**: `tools/executor.py` uses different field names than `core/types.py`  
**Fix**: Align on single ToolResult in core/types.py

```python
# core/types.py - CANONICAL
class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    result: Any           # Changed from 'output'
    error: str | None
    duration_ms: float
    success: bool = True  # Derived property
```

### 5.4 Eliminate Prompt Registry Singleton

**Current**: Global `_registry` in `reasoning/prompts/template.py`  
**Fix**: Dependency injection via factory

```python
# BEFORE (singleton)
def get_prompt_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry

# AFTER (factory)
def create_prompt_registry(templates: dict[str, str] | None = None) -> PromptRegistry:
    registry = PromptRegistry()
    # Load templates from config
    return registry
```

### 5.5 Wire Guardrails into Response Path

**Current**: Guardrails exist but not connected  
**Fix**: AgentRuntime applies guardrails before returning

```python
# In AgentRuntime.run()
async def run(self, ...):
    # ... reasoning ...
    
    # Apply output guardrails
    guard_result = await self._guardrails.check(response.content)
    if guard_result.blocked:
        return AgentResult(
            content="I cannot provide that response.",
            blocked=True,
            blocked_reason=guard_result.reason,
        )
    
    return AgentResult(content=response.content)
```

### 5.6 Integrate RAG into Context Assembly

**Current**: RAG exists but not wired to reasoning  
**Fix**: AgentRuntime assembles context with RAG

```python
# In AgentRuntime
async def _build_context(
    self,
    query: str,
    history: list[Message],
) -> list[Message]:
    messages = []
    
    # System prompt
    messages.append(Message(role="system", content=self._system_prompt))
    
    # RAG context (if enabled)
    if self._retriever and self._context.enable_rag:
        docs = await self._retriever.retrieve(query, top_k=5)
        if docs:
            context_text = "\n\n".join(d.content for d in docs)
            messages.append(Message(
                role="system",
                content=f"Relevant context:\n{context_text}",
            ))
    
    # Conversation history
    messages.extend(history)
    
    return messages
```

---

## 6. Implementation Priority

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | Create `AgentRuntime` orchestrator | High | Critical - enables deterministic flow |
| P0 | Extract `ToolExecutorProtocol` to core | Low | Critical - breaks circular dependency |
| P1 | Fix `ToolResult` schema alignment | Low | High - prevents runtime errors |
| P1 | Wire guardrails into AgentRuntime | Medium | High - safety enforcement |
| P1 | Wire RAG into AgentRuntime | Medium | High - feature completeness |
| P2 | Eliminate PromptRegistry singleton | Low | Medium - testability |
| P2 | Add proper error types | Low | Medium - debuggability |
| P3 | Add execution replay capability | High | Medium - debugging |

---

## 7. Quality Assurance Checklist

### Deterministic Execution
- [ ] Same input + same state = same output
- [ ] No hidden randomness (seeded where needed)
- [ ] No time-dependent behavior in core logic

### Replayable Behavior
- [ ] All inputs captured in trace
- [ ] All tool calls logged with args/results
- [ ] Checkpoints at defined boundaries

### Traceable Decisions
- [ ] Every LLM call has span
- [ ] Every tool call has span
- [ ] Decision points emit events

### Zero Cross-Module Reach-Ins
- [ ] No module imports another's private functions
- [ ] All cross-module communication via defined interfaces
- [ ] No shared mutable state between modules

---

## 8. 3AM Production Debug Playbook

When things break at 3AM, follow this trace:

1. **Check trace_id** in error logs → correlate all operations
2. **Find session_id** → load session state from Redis
3. **Identify which span failed** → pinpoint component
4. **Check AgentRuntime state** → understand execution phase
5. **Review tool calls** → check for external failures
6. **Check guardrail logs** → was response blocked?
7. **Replay from checkpoint** → reproduce issue

```bash
# Quick debug commands
redis-cli GET "aegis:session:$SESSION_ID" | jq .
redis-cli GET "aegis:trace:$TRACE_ID" | jq .
curl -s localhost:8000/admin/traces/$TRACE_ID | jq .
```
