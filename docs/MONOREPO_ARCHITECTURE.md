# AEGIS Agent Platform - Monorepo Architecture

Production-grade AI Agent Platform reimagined as a modular monorepo.

## Architecture Overview

```
aegis-agent-platform/
├── services/           # Core service modules
│   ├── agent_core/     # Agent orchestration & state management
│   ├── reasoning/      # LLM routing & prompt engineering
│   ├── rag/            # Retrieval-augmented generation
│   ├── memory/         # Session & long-term memory
│   ├── tools/          # Tool registry & execution
│   ├── governance/     # Safety policies & content filtering
│   ├── evaluation/     # Quality metrics & testing
│   └── observability/  # Tracing, metrics, logging
├── apps/               # Application entry points
│   ├── api_server/     # FastAPI REST API
│   └── worker/         # Background task processor
├── configs/            # Configuration files
│   ├── model_config.yaml
│   ├── rag_config.yaml
│   ├── policy_rules.yaml
│   ├── tool_manifest.yaml
│   └── domains.yaml
├── infra/              # Infrastructure
│   └── docker/         # Docker configurations
│       ├── Dockerfile
│       ├── docker-compose.dev.yaml
│       ├── docker-compose.prod.yaml
│       └── docker-compose.offline.yaml
└── tests/              # Test suite
```

## Quick Start

### Development Mode

```bash
# Clone and setup
cd aegis-agent-platform

# Install dependencies
pip install -e ".[dev]"

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run with Docker Compose
docker-compose -f infra/docker/docker-compose.dev.yaml up
```

### Offline Mode (No API Keys Required)

```bash
# Run in offline mode with stub responses
docker-compose -f infra/docker/docker-compose.offline.yaml up
```

### Production Mode

```bash
# Build and deploy
docker-compose -f infra/docker/docker-compose.prod.yaml up -d
```

## Services

### Agent Core (`services/agent_core/`)
- **Orchestrator**: Main agent execution loop with ReAct/Tool-calling
- **Planner**: Task decomposition and planning
- **State Manager**: Execution state and context management
- **Execution Graph**: DAG-based task execution
- **Reflection**: Self-improvement and error correction

### Reasoning (`services/reasoning/`)
- **LLM Router**: Intelligent model selection
- **Prompt Builder**: Template-based prompt construction
- **Response Parser**: Structured output parsing
- **Model Adapters**: Anthropic, OpenAI, Stub adapters

### RAG (`services/rag/`)
- **Index Manager**: Document indexing and management
- **Retriever**: Semantic search and retrieval
- **Hybrid Search**: Combined vector + keyword search
- **Reranker**: Cross-encoder reranking
- **Domain Registry**: Domain-specific configurations

### Memory (`services/memory/`)
- **Session Memory**: Short-term conversation context
- **Long-term Memory**: Persistent semantic memory
- **Summarizer**: Context compression
- **Vector Memory**: Embedding-based recall

### Tools (`services/tools/`)
- **Tool Registry**: Central tool management
- **Tool Executor**: Safe sandboxed execution
- **Tool Validator**: Input/output validation
- **Builtins**: Calculator, web search, datetime, etc.

### Governance (`services/governance/`)
- **Policy Engine**: Rule-based policy evaluation
- **Content Filter**: PII detection and filtering
- **Injection Guard**: Prompt injection detection
- **Permission Checker**: RBAC enforcement

### Evaluation (`services/evaluation/`)
- **RAG Metrics**: Retrieval quality metrics
- **Hallucination Check**: Factual consistency validation
- **Regression Tests**: Automated testing framework
- **Benchmark Runner**: Performance benchmarking

### Observability (`services/observability/`)
- **Tracing**: Distributed tracing (OpenTelemetry-compatible)
- **Metrics**: Prometheus-style metrics
- **Logging**: Structured logging
- **Audit Log**: Security audit trail

## API Endpoints

### Chat
- `POST /api/v1/chat` - Send chat message
- `POST /api/v1/chat/stream` - Streaming chat (SSE)
- `GET /api/v1/chat/{session_id}/history` - Get chat history

### Sessions
- `POST /api/v1/sessions` - Create session
- `GET /api/v1/sessions/{id}` - Get session
- `DELETE /api/v1/sessions/{id}` - Delete session

### Tools
- `GET /api/v1/tools` - List tools
- `POST /api/v1/tools/execute` - Execute tool
- `PUT /api/v1/tools/{name}/enable` - Enable tool
- `PUT /api/v1/tools/{name}/disable` - Disable tool

### Domains
- `GET /api/v1/domains` - List domains
- `GET /api/v1/domains/{id}` - Get domain config

### Health
- `GET /health` - Health check
- `GET /health/ready` - Readiness check
- `GET /health/live` - Liveness check
- `GET /metrics` - Prometheus metrics

## Configuration

### Environment Variables

```bash
# Core
AEGIS_ENV=development|production|offline
AEGIS_DEBUG=true|false
AEGIS_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR

# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Storage
AEGIS_REDIS_URL=redis://localhost:6379/0
AEGIS_DATABASE_URL=postgresql://...

# Features
AEGIS_OFFLINE_MODE=false
AEGIS_DEFAULT_MODEL=claude-3-5-sonnet
```

### Domain Configuration

Domains are defined in `configs/domains.yaml`:

```yaml
technical_support:
  model:
    default: claude-3-5-sonnet
    temperature: 0.3
  system_prompt: |
    You are a technical support specialist...
  tools:
    enabled: [calculator, web_search]
  rag:
    enabled: true
    namespace: technical_kb
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=services --cov=apps

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m e2e
```

## Development

### Code Style

```bash
# Format code
black services apps tests
ruff check --fix services apps tests

# Type checking
mypy services apps
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## License

MIT License - see [LICENSE](LICENSE) for details.
