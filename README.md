# Aegis — Enterprise-Grade Modular AI Agent Platform

[![CI](https://github.com/aegis-ai/aegis-agent-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/aegis-ai/aegis-agent-platform/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/aegis-ai/aegis-agent-platform/branch/main/graph/badge.svg)](https://codecov.io/gh/aegis-ai/aegis-agent-platform)

A production-ready, modular AI agent platform designed for enterprise deployments. Built with Python, FastAPI, and modern async patterns. Supports **development**, **production**, and **offline** modes.

## ✨ Features

- **🧠 Agent Core** — Task orchestration, planning, state management, execution graphs, and self-reflection
- **💡 Reasoning** — Provider-agnostic LLM integration (OpenAI, Anthropic, Stub) with model routing
- **📚 RAG Pipeline** — Document indexing, hybrid search (vector + keyword), reranking, and domain-specific retrieval
- **💾 Memory System** — Session management, long-term memory with decay, vector memory, and summarization
- **🔧 Tool System** — Extensible tool registry with validation, execution, rate limiting, and built-in tools
- **🛡️ Governance** — Policy engine, content filtering, prompt injection detection, and RBAC
- **📊 Evaluation** — RAG metrics, hallucination detection, regression testing, and benchmarking
- **👀 Observability** — Distributed tracing, Prometheus metrics, structured logging, and audit logging
- **🚀 API Server** — FastAPI with streaming (SSE), middleware composition, and dependency injection
- **⚙️ Worker** — Background task processing for document ingestion, indexing, and evaluation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (FastAPI)                            │
│   ┌─────────┐  ┌─────────────┐  ┌───────────┐  ┌─────────┐   ┌─────────────┐│
│   │  Chat   │  │  Sessions   │  │   Tools   │  │  Admin  │   │   Health    ││
│   └────┬────┘  └──────┬──────┘  └─────┬─────┘  └────┬────┘   └──────┬──────┘│
└────────┼──────────────┼───────────────┼─────────────┼───────────────┼───────┘
         │              │               │             │               │
┌────────┴──────────────┴───────────────┴─────────────┴───────────────┴───────┐
│                           MIDDLEWARE STACK                                  │
│     [Tracing] → [RateLimit] → [Auth] → [ErrorHandling] → [Streaming]        │
└─────────────────────────────────────────────────────────────────────────────┘
         │              │               │             │               │
┌────────┴──────────────┴───────────────┴─────────────┴───────────────┴───────┐
│                              CORE SERVICES                                  │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │
│  │ Agent Core │ │ Reasoning  │ │    RAG     │ │   Memory   │ │   Tools    │ │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ │
│        │              │              │              │              │        │
│  ┌─────┴──────────────┴──────────────┴──────────────┴──────────────┴───────┐│
│  │                    GOVERNANCE & OBSERVABILITY                           ││
│  │  [PolicyEngine] [ContentFilter] [Tracing] [Metrics] [AuditLog]          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
         │              │               │             │               │
┌────────┴──────────────┴───────────────┴─────────────┴───────────────┴───────┐
│                           DATA LAYER                                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
│  │   Memory   │  │  Knowledge │  │  Sessions  │  │   Vector Store         │ │
│  │   (Redis)  │  │  (RAG)     │  │  (Redis)   │  │   (FAISS/In-Memory)    │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure (Monorepo)

```
aegis-agent-platform/
├── services/                    # Core service modules
│   ├── agent_core/              # Agent orchestration (6 files)
│   │   ├── orchestrator.py      # Main agent execution loop
│   │   ├── planner.py           # Task decomposition
│   │   ├── state_manager.py     # Execution state management
│   │   ├── execution_graph.py   # DAG-based task execution
│   │   └── reflection.py        # Self-improvement engine
│   ├── reasoning/               # LLM integration (8 files)
│   │   ├── llm_router.py        # Model routing & fallback
│   │   ├── prompt_builder.py    # Prompt construction
│   │   ├── response_parser.py   # Response parsing
│   │   └── model_adapters/      # OpenAI, Anthropic, Stub adapters
│   ├── rag/                     # RAG pipeline (10 files)
│   │   ├── index_manager.py     # Document indexing
│   │   ├── retriever.py         # Semantic retrieval
│   │   ├── hybrid_search.py     # Vector + keyword search with RRF
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   ├── domain_registry.py   # Domain-specific configs
│   │   └── chunking/            # Recursive & semantic chunking
│   ├── memory/                  # Memory system (4 files)
│   │   ├── session_memory.py    # Conversation memory
│   │   ├── long_term_memory.py  # Persistent memory with decay
│   │   ├── summarizer.py        # LLM/extractive summarization
│   │   └── vector_memory.py     # Semantic memory retrieval
│   ├── tools/                   # Tool system (8 files)
│   │   ├── tool_registry.py     # Registration & schema generation
│   │   ├── tool_executor.py     # Execution with timeout/retry
│   │   ├── tool_validator.py    # Input validation & injection detection
│   │   └── builtins/            # Web, code, file, math tools
│   ├── governance/              # Safety & policy (4 files)
│   │   ├── policy_engine.py     # YAML-based policy rules
│   │   ├── content_filter.py    # Content filtering levels
│   │   ├── injection_guard.py   # Prompt injection detection
│   │   └── permission_checker.py # RBAC with role hierarchy
│   ├── evaluation/              # Testing & metrics (4 files)
│   │   ├── rag_metrics.py       # Context/answer relevance
│   │   ├── hallucination_check.py # Claim verification
│   │   ├── regression_tests.py  # JSON-based test suites
│   │   └── benchmark_runner.py  # Performance benchmarking
│   └── observability/           # Monitoring (4 files)
│       ├── tracing.py           # Distributed tracing
│       ├── metrics.py           # Prometheus metrics
│       ├── logging.py           # Structured logging
│       └── audit_log.py         # Audit trail
├── apps/                        # Applications
│   ├── api_server/              # FastAPI server
│   │   ├── app.py               # Application factory
│   │   ├── dependencies.py      # Dependency injection
│   │   ├── middleware.py        # Custom middleware
│   │   └── routes/              # API endpoints
│   └── worker/                  # Background worker
│       ├── worker.py            # Task processor
│       └── tasks.py             # Task definitions
├── configs/                     # Configuration files
│   ├── model_config.yaml        # LLM model settings
│   ├── rag_config.yaml          # RAG parameters
│   ├── policy_rules.yaml        # Safety policies
│   └── tool_manifest.yaml       # Tool definitions
├── infra/                       # Infrastructure
│   ├── docker/                  # Additional Docker configs
│   │   └── nginx/               # Nginx configuration
│   ├── kubernetes/              # K8s manifests
│   └── terraform/               # IaC templates
├── tests/                       # Test suites
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── e2e/                     # End-to-end tests
├── Dockerfile                   # Multi-stage Docker build (dev/prod/offline)
├── docker-compose.yml           # Docker Compose with profiles
└── data/                        # Runtime data
    └── vector_store/            # FAISS indices
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis (optional, for production)
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/aegis-ai/aegis-agent-platform.git
cd aegis-agent-platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install optional providers
pip install openai anthropic  # LLM providers
pip install faiss-cpu         # Vector store

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys
```

### Running

```bash
# Development mode (with hot reload)
uvicorn apps.api_server.app:create_app --factory --reload --port 8000

# Production mode
uvicorn apps.api_server.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 4

# Using Docker Compose
docker-compose up -d

# Offline mode (no API keys required)
docker-compose --profile offline up -d aegis-offline
```

### First API Call

```bash
# Health check
curl http://localhost:8080/health

# Create a session
curl -X POST http://localhost:8080/api/v1/sessions

# Send a message
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "session_id": "<session-id>"}'
```

## 🐳 Docker Modes

All modes use the same `docker-compose.yml` with different profiles:

### Development Mode
```bash
# Hot reload enabled, debug logging (port 8001)
docker-compose --profile dev up -d aegis-dev redis

# Or without Redis
docker-compose --profile dev up -d aegis-dev
```

### Production Mode
```bash
# Standard production deployment (port 8000)
docker-compose up -d
```

### Offline Mode (No External APIs)
```bash
# Air-gapped deployment with stub LLM (port 8002) 
docker-compose --profile offline up -d aegis-offline

# Verify health
curl http://localhost:8002/health

# Test chat (uses stub responses)
curl -X POST http://localhost:8002/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# Access Swagger UI
open http://localhost:8002/docs
```

Offline mode is useful for:
- Local development without API keys
- CI/CD pipeline testing
- Demo environments
- Air-gapped deployments
- Architecture validation

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Basic health check |
| GET | `/health/ready` | Readiness probe |
| GET | `/health/live` | Liveness probe |
| GET | `/health/metrics` | Prometheus metrics |
| POST | `/api/v1/chat` | Send message (supports streaming) |
| POST | `/api/v1/sessions` | Create session |
| GET | `/api/v1/sessions` | List sessions |
| GET | `/api/v1/sessions/{id}` | Get session history |
| DELETE | `/api/v1/sessions/{id}` | Delete session |
| GET | `/api/v1/tools` | List available tools |
| GET | `/api/v1/tools/{name}` | Get tool details |
| POST | `/api/v1/tools/call` | Execute a tool |
| GET | `/api/v1/admin/stats` | Platform statistics |
| GET | `/api/v1/admin/audit` | Audit logs |
| GET | `/api/v1/admin/config` | Current configuration |

## 🔧 Configuration

Configuration is managed through environment variables and YAML files:

```bash
# Core settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# LLM providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Offline mode
OFFLINE_MODE=true
DEFAULT_MODEL=stub

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
ENABLE_RATE_LIMIT=true
RATE_LIMIT_RPM=60
```

### YAML Configuration Files

- `configs/model_config.yaml` - Model definitions, routing rules, rate limits
- `configs/rag_config.yaml` - Embedding, chunking, retrieval settings
- `configs/policy_rules.yaml` - Safety policies, content filters, RBAC roles
- `configs/tool_manifest.yaml` - Tool definitions and permissions

## 🧪 Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=services --cov=apps --cov-report=html

# Run specific test types
pytest tests/unit/ -v           # Unit tests
pytest tests/integration/ -v    # Integration tests
pytest tests/e2e/ -v -m e2e     # E2E tests

# Run offline tests (no external APIs)
pytest tests/ -m "not requires_llm"
```

### Test Structure

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_agent_core.py   # Orchestrator, planner, state
│   ├── test_rag.py          # Indexing, retrieval, chunking
│   ├── test_memory.py       # Session, long-term memory
│   ├── test_tools.py        # Registry, executor, validators
│   └── test_governance.py   # Policies, filters, permissions
├── integration/             # Service integration tests
│   ├── test_api.py          # API endpoint tests
│   └── test_rag_pipeline.py # Full RAG pipeline
├── e2e/                     # End-to-end scenarios
│   └── test_chat_flow.py    # Complete chat workflows
└── fixtures.py              # Shared test fixtures
```

## 🔌 Extending

### Adding a Custom Tool

```python
from services.tools import ToolRegistry

registry = ToolRegistry()

@registry.tool(
    name="my_custom_tool",
    description="Description of what the tool does",
)
def my_custom_tool(query: str, limit: int = 10) -> dict:
    """Execute custom logic."""
    return {"result": f"Processed: {query}", "limit": limit}
```

### Adding an LLM Provider

```python
from services.reasoning.model_adapters.base import BaseModelAdapter, LLMResponse

class CustomAdapter(BaseModelAdapter):
    async def complete(self, messages: list, **kwargs) -> LLMResponse:
        # Implementation
        pass
    
    async def stream(self, messages: list, **kwargs):
        # Streaming implementation
        async for chunk in self._call_api(messages):
            yield chunk
```

### Creating a Background Task

```python
from apps.worker.tasks import TASKS

async def my_task(payload: dict) -> dict:
    """Process background task."""
    # Implementation
    return {"status": "completed"}

TASKS["my_task"] = my_task
```

## 📈 CI/CD

The project uses GitHub Actions for continuous integration:

- **Lint**: Ruff, Black, MyPy checks
- **Test**: Unit, integration, and E2E tests with coverage
- **Build**: Docker image building
- **Security**: Dependency scanning
- **Deploy**: Automated deployments (on release)

```yaml
# Trigger CI
git push origin main

# View workflows
https://github.com/aegis-ai/aegis-agent-platform/actions
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📚 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Domain System](docs/DOMAIN_SYSTEM.md)
- [System Integration](docs/SYSTEM_INTEGRATION.md)
- [API Documentation](http://localhost:8080/docs) (when running)
