# Aegis Agent Platform Architecture

> **Last Updated:** January 2026
> **Version:** 1.0.0

## Overview

Aegis is a production-ready AI agent platform designed for enterprise deployments. It provides a modular, extensible architecture with domain-aware routing, multi-LLM support, and comprehensive observability.

## System Architecture

```mermaid
flowchart TB
    subgraph Client ["Client Layer"]
        REST[REST API Client]
        SSE[SSE Streaming Client]
    end
    
    subgraph API ["API Layer (FastAPI)"]
        Routes[API Routes]
        Middleware[Middleware Stack]
        Auth[AuthMiddleware]
        Tracing[TracingMiddleware]
        RateLimit[RateLimitMiddleware]
    end
    
    subgraph Core ["Core Runtime"]
        DomainRuntime[DomainAwareRuntime]
        AgentRuntime[AgentRuntime]
        Resolver[DomainResolver]
    end
    
    subgraph Reasoning ["Reasoning Layer"]
        LLMAdapter[LLM Adapters]
        Strategies[Reasoning Strategies]
        Prompts[Prompt Templates]
    end
    
    subgraph Memory ["Memory Layer"]
        SessionMgr[SessionManager]
        ShortTerm[ShortTermMemory]
        LongTerm[LongTermMemory]
    end
    
    subgraph Tools ["Tools Layer"]
        Registry[ToolRegistry]
        Executor[ToolExecutor]
        Builtin[Built-in Tools]
    end
    
    subgraph Safety ["Safety Layer"]
        Guardrails[GuardrailChain]
        InputVal[InputValidator]
        RBAC[RBACManager]
        Audit[AuditLogger]
    end
    
    subgraph Knowledge ["Knowledge Layer"]
        VectorStore[VectorStore]
        Embeddings[EmbeddingService]
        Retriever[Retriever]
    end
    
    subgraph Observability ["Observability"]
        Logger[StructuredLogger]
        Metrics[MetricsCollector]
        Traces[Tracer]
    end
    
    Client --> API
    Routes --> Middleware
    Middleware --> Auth
    Middleware --> Tracing
    Middleware --> RateLimit
    
    API --> Core
    Core --> DomainRuntime
    DomainRuntime --> AgentRuntime
    DomainRuntime --> Resolver
    
    AgentRuntime --> Reasoning
    AgentRuntime --> Memory
    AgentRuntime --> Tools
    AgentRuntime --> Safety
    AgentRuntime --> Knowledge
    
    Reasoning --> LLMAdapter
    Reasoning --> Strategies
    Strategies --> Prompts
    
    Memory --> SessionMgr
    Memory --> ShortTerm
    Memory --> LongTerm
    
    Tools --> Registry
    Tools --> Executor
    Registry --> Builtin
    
    Safety --> Guardrails
    Safety --> InputVal
    Safety --> RBAC
    Safety --> Audit
    
    Knowledge --> VectorStore
    Knowledge --> Embeddings
    Knowledge --> Retriever
    
    Core --> Observability
```

## Module Structure

```
src/
├── api/                    # FastAPI REST API
│   ├── app.py              # Application factory & lifespan
│   ├── dependencies.py     # Dependency injection
│   ├── middleware.py       # Rate limiting, auth, tracing
│   ├── streaming.py        # SSE streaming support
│   └── routes/             # API endpoints
│       ├── chat.py         # Chat completions
│       ├── sessions.py     # Session management
│       ├── tools.py        # Tool introspection
│       ├── domains.py      # Domain profiles
│       ├── health.py       # Health checks
│       └── admin.py        # Administrative endpoints
│
├── config/                 # Configuration management
│   ├── settings.py         # Application settings
│   ├── model_routing.py    # LLM model routing
│   └── secrets.py          # Secrets management
│
├── core/                   # Core types & interfaces
│   ├── types.py            # Pydantic models & types
│   ├── interfaces.py       # Protocol definitions
│   └── exceptions.py       # Custom exceptions
│
├── domains/                # Domain profile system
│   ├── profile.py          # DomainProfile model
│   ├── registry.py         # DomainRegistry
│   ├── resolver.py         # Domain resolution logic
│   └── runtime.py          # DomainAwareRuntime
│
├── knowledge/              # RAG & knowledge base
│   ├── vector_store.py     # FAISS/Milvus adapters
│   ├── embeddings.py       # Embedding services
│   ├── retriever.py        # Context retrieval
│   ├── chunking.py         # Text chunking strategies
│   └── ingestion.py        # Document ingestion
│
├── memory/                 # State & memory management
│   ├── session.py          # Session management
│   ├── short_term.py       # Conversation window
│   ├── long_term.py        # Persistent memory
│   └── retrieval.py        # Memory retrieval
│
├── observability/          # Monitoring & logging
│   ├── logging.py          # Structured logging
│   ├── metrics.py          # Prometheus metrics
│   ├── tracing.py          # Distributed tracing
│   └── evaluation.py       # Evaluation harness
│
├── planning/               # Task planning
│   ├── controller.py       # Execution controller
│   ├── decomposer.py       # Task decomposition
│   └── checkpoints.py      # Checkpoint management
│
├── reasoning/              # LLM & reasoning
│   ├── llm/                # LLM adapters
│   │   ├── base.py         # Base adapter interface
│   │   ├── openai_adapter.py
│   │   ├── anthropic_adapter.py
│   │   └── stub_adapter.py # Offline mode stub
│   ├── prompts/            # Prompt management
│   │   └── template.py     # Jinja2 templates
│   └── strategies/         # Reasoning strategies
│       ├── base.py         # Strategy interface
│       ├── react.py        # ReAct pattern
│       └── tool_calling.py # Native tool calling
│
├── runtime/                # Agent runtime
│   ├── agent.py            # AgentRuntime core
│   └── factory.py          # Runtime factory
│
├── safety/                 # Security & compliance
│   ├── guardrails.py       # Input/output guardrails
│   ├── input_validation.py # Injection detection
│   ├── rbac.py             # Role-based access
│   └── audit.py            # Audit logging
│
├── tools/                  # Tool system
│   ├── registry.py         # Tool registration
│   ├── executor.py         # Tool execution
│   ├── permissions.py      # Tool permissions
│   ├── builtin.py          # Built-in tools
│   └── tracing.py          # Tool tracing
│
└── advanced/               # Advanced features
    ├── multi_agent.py      # Multi-agent orchestration
    ├── plugins.py          # Plugin system
    └── critic.py           # Self-critique loop
```

## Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant MW as Middleware
    participant DAR as DomainAwareRuntime
    participant DR as DomainResolver
    participant AR as AgentRuntime
    participant LLM as LLMAdapter
    participant Tools as ToolExecutor
    participant Mem as SessionManager

    Client->>API: POST /api/v1/chat
    API->>MW: Process request
    MW->>MW: Auth check
    MW->>MW: Rate limit check
    MW->>MW: Start trace
    
    API->>DAR: run(message, context)
    DAR->>DR: resolve_domain(content)
    DR-->>DAR: DomainProfile
    
    DAR->>AR: run_with_profile(profile, message)
    AR->>LLM: generate(messages, tools)
    
    loop Tool Calls
        LLM-->>AR: tool_call request
        AR->>Tools: execute(tool_name, args)
        Tools-->>AR: ToolResult
        AR->>LLM: continue with result
    end
    
    LLM-->>AR: final response
    AR->>Mem: update_session(messages)
    AR-->>DAR: AgentResult
    DAR-->>API: ChatResponse
    
    API->>MW: Finalize trace
    API-->>Client: JSON/SSE Response
```

## Domain Resolution Flow

```mermaid
flowchart TD
    Start([Incoming Message]) --> ExplicitCheck{Explicit<br/>Domain?}
    
    ExplicitCheck -->|Yes| ValidateDomain{Domain<br/>Exists?}
    ValidateDomain -->|Yes| UseDomain[Use Specified Domain]
    ValidateDomain -->|No| DefaultDomain[Use Default Domain]
    
    ExplicitCheck -->|No| InferDomain[DomainResolver.infer]
    InferDomain --> AnalyzeContent[Analyze Message Content]
    
    AnalyzeContent --> KeywordMatch{Keyword<br/>Match?}
    KeywordMatch -->|Yes| ScoreMatches[Score All Matches]
    KeywordMatch -->|No| DefaultDomain
    
    ScoreMatches --> ThresholdCheck{Confidence ><br/>Threshold?}
    ThresholdCheck -->|Yes| UseBestMatch[Use Best Match]
    ThresholdCheck -->|No| DefaultDomain
    
    UseDomain --> ApplyProfile[Apply Profile Constraints]
    UseBestMatch --> ApplyProfile
    DefaultDomain --> ApplyProfile
    
    ApplyProfile --> ConfigureRuntime[Configure Runtime]
    ConfigureRuntime --> End([Execute with Domain])
```

## LLM Adapter System

```mermaid
classDiagram
    class BaseLLMAdapter {
        <<abstract>>
        +model: str
        +generate(messages, tools) LLMResponse
        +stream(messages, tools) AsyncIterator
        +count_tokens(text) int
    }
    
    class OpenAIAdapter {
        +client: AsyncOpenAI
        +generate(messages, tools) LLMResponse
        +stream(messages, tools) AsyncIterator
    }
    
    class AnthropicAdapter {
        +client: AsyncAnthropic
        +generate(messages, tools) LLMResponse
        +stream(messages, tools) AsyncIterator
    }
    
    class StubLLMAdapter {
        +response_delay: float
        +generate(messages, tools) LLMResponse
        +stream(messages, tools) AsyncIterator
    }
    
    BaseLLMAdapter <|-- OpenAIAdapter
    BaseLLMAdapter <|-- AnthropicAdapter
    BaseLLMAdapter <|-- StubLLMAdapter
```

## Tool Execution Flow

```mermaid
flowchart LR
    subgraph Request ["Tool Request"]
        LLM[LLM Response]
        Parse[Parse Tool Calls]
    end
    
    subgraph Validation ["Validation"]
        Exists{Tool<br/>Exists?}
        Allowed{Tool<br/>Allowed?}
        ArgCheck{Args<br/>Valid?}
    end
    
    subgraph Execution ["Execution"]
        Execute[Execute Tool]
        Trace[Record Trace]
        Result[Format Result]
    end
    
    LLM --> Parse
    Parse --> Exists
    Exists -->|No| Error1[Tool Not Found]
    Exists -->|Yes| Allowed
    Allowed -->|No| Error2[Permission Denied]
    Allowed -->|Yes| ArgCheck
    ArgCheck -->|No| Error3[Invalid Args]
    ArgCheck -->|Yes| Execute
    
    Execute --> Trace
    Trace --> Result
    Result --> Return[Return to LLM]
```

## Memory Architecture

```mermaid
flowchart TB
    subgraph SessionLayer ["Session Layer"]
        SM[SessionManager]
        IMB[InMemoryBackend]
        RB[RedisBackend]
    end
    
    subgraph ShortTermLayer ["Short-Term Memory"]
        Window[WindowMemory]
        Summarizing[SummarizingMemory]
    end
    
    subgraph LongTermLayer ["Long-Term Memory"]
        LTM[LongTermMemory]
        VectorMem[Vector Store]
    end
    
    SM --> IMB
    SM --> RB
    
    SM --> Window
    SM --> Summarizing
    
    Window --> LTM
    Summarizing --> LTM
    
    LTM --> VectorMem
```

## Safety & Guardrails

```mermaid
flowchart LR
    subgraph Input ["Input Processing"]
        Receive[Receive Input]
        InputVal[InputValidator]
        Injection[InjectionDetector]
        ContentFilt[ContentFilter]
    end
    
    subgraph Processing ["Core Processing"]
        Agent[Agent Runtime]
    end
    
    subgraph Output ["Output Processing"]
        OutputGuard[OutputGuardrail]
        PolicyCheck[PolicyEvaluator]
        Audit[AuditLogger]
    end
    
    Receive --> InputVal
    InputVal --> Injection
    Injection --> ContentFilt
    ContentFilt --> Agent
    
    Agent --> OutputGuard
    OutputGuard --> PolicyCheck
    PolicyCheck --> Audit
    Audit --> Response[Response]
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_OFFLINE_MODE` | Use stub adapter | `false` |
| `REDIS_ENABLED` | Enable Redis backend | `true` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `LOG_LEVEL` | Logging level | `INFO` |
| `RATE_LIMIT_REQUESTS` | Requests per window | `100` |
| `RATE_LIMIT_WINDOW` | Window in seconds | `60` |

### Domain Profile Configuration

Domain profiles are defined in YAML files under `config/domains/`:

```yaml
name: financial_analysis
version: "1.0.0"
display_name: "Financial Analysis Assistant"
description: |
  Financial analyst assistant for portfolio analysis,
  market research, and investment insights.

tags:
  - finance
  - investment
  - analysis

system_prompt: |
  You are a financial analyst assistant...

inference:
  keywords:
    - stock
    - investment
    - portfolio
  priority: 80
  confidence_threshold: 0.7

tools:
  allowed:
    - calculate
    - json_parse
  denied:
    - http_get

guardrails:
  max_output_tokens: 4000
  block_code_execution: true
```

## Deployment

### Docker Profiles

| Profile | Description | Services |
|---------|-------------|----------|
| default | Standard deployment | aegis, redis |
| dev | Development with hot reload | aegis-dev, redis |
| offline | No external dependencies | aegis-offline |

### Running Offline Mode

```bash
# Build and run without external LLM dependencies
docker compose --profile offline up -d aegis-offline

# Verify health
curl http://localhost:8002/health

# Test chat (uses StubLLMAdapter)
curl -X POST http://localhost:8002/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |
| `/api/v1/chat` | POST | Send chat message |
| `/api/v1/sessions` | POST | Create session |
| `/api/v1/sessions` | GET | List sessions |
| `/api/v1/sessions/{id}` | GET | Get session |
| `/api/v1/sessions/{id}` | DELETE | Delete session |
| `/api/v1/tools` | GET | List available tools |
| `/api/v1/domains` | GET | List domain profiles |
| `/api/v1/domains/{name}` | GET | Get domain details |

## References

- [Domain System Documentation](DOMAIN_SYSTEM.md)
- [System Integration Guide](SYSTEM_INTEGRATION.md)
- [Consolidation Report](CONSOLIDATION.md)
