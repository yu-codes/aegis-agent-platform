# Aegis — Enterprise-Grade Modular AI Agent Platform

[![CI](https://github.com/aegis-ai/aegis-agent-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/aegis-ai/aegis-agent-platform/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A production-ready, modular AI agent platform designed for enterprise deployments. Built with Python, FastAPI, and modern async patterns.

## ✨ Features

- **🧠 Reasoning Core** — Provider-agnostic LLM integration (OpenAI, Anthropic) with ReAct and tool-calling strategies
- **💾 State & Memory** — Session management, short-term context, long-term retrieval with Redis backend
- **📚 Knowledge/RAG** — Document ingestion, chunking, embeddings, and vector store integration
- **🔧 Tool System** — Extensible tool registry with permissions, rate limiting, and execution tracing
- **📋 Planning & Orchestration** — Task decomposition, step control, and checkpoint management
- **🛡️ Safety & Governance** — Input validation, guardrails, RBAC, and comprehensive audit logging
- **📊 Observability** — OpenTelemetry-compatible tracing, Prometheus metrics, structured logging
- **🚀 API Layer** — FastAPI with streaming (SSE), middleware composition, and dependency injection
- **🤖 Multi-Agent** — Agent orchestration, critic/reflection patterns, plugin architecture

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (FastAPI)                            │
│   ┌─────────┐  ┌─────────────┐  ┌───────────┐  ┌─────────┐  ┌─────────────┐│
│   │  Chat   │  │  Sessions   │  │   Tools   │  │  Admin  │  │   Health    ││
│   └────┬────┘  └──────┬──────┘  └─────┬─────┘  └────┬────┘  └──────┬──────┘│
└────────┼──────────────┼───────────────┼─────────────┼───────────────┼───────┘
         │              │               │             │               │
┌────────┴──────────────┴───────────────┴─────────────┴───────────────┴───────┐
│                           MIDDLEWARE STACK                                   │
│     [Tracing] → [RateLimit] → [Auth] → [ErrorHandling] → [Streaming]        │
└─────────────────────────────────────────────────────────────────────────────┘
         │              │               │             │               │
┌────────┴──────────────┴───────────────┴─────────────┴───────────────┴───────┐
│                              CORE MODULES                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │  Reasoning     │  │  Planning &    │  │  Multi-Agent   │                 │
│  │  (LLM + Tools) │  │  Orchestration │  │  Coordination  │                 │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                 │
│          │                   │                   │                          │
│  ┌───────┴───────────────────┴───────────────────┴────────┐                 │
│  │                    SAFETY & GOVERNANCE                  │                 │
│  │  [Validation] [Guardrails] [RBAC] [Audit] [Plugins]    │                 │
│  └─────────────────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
         │              │               │             │               │
┌────────┴──────────────┴───────────────┴─────────────┴───────────────┴───────┐
│                           DATA LAYER                                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
│  │   Memory   │  │  Knowledge │  │  Sessions  │  │   Vector Store         │ │
│  │   (Redis)  │  │  (RAG)     │  │  (Redis)   │  │   (FAISS/Milvus)       │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis (for session storage)
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
# Start Redis (if not using Docker)
redis-server

# Start the API server
uvicorn src.api.app:create_app --factory --reload

# Or use Docker Compose
docker-compose up -d
```

### First API Call

```bash
# Create a session
curl -X POST http://localhost:8000/sessions

# Send a message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session-id>", "message": "Hello!"}'
```

## 📁 Project Structure

```
src/
├── api/                    # FastAPI application
│   ├── app.py             # App factory & lifespan
│   ├── middleware.py      # Request/response middleware
│   ├── streaming.py       # SSE streaming utilities
│   ├── dependencies.py    # Dependency injection
│   └── routes/            # API endpoints
├── config/                 # Configuration management
│   ├── settings.py        # Application settings
│   ├── secrets.py         # Secrets handling
│   └── model_routing.py   # LLM model routing
├── core/                   # Core types & exceptions
│   ├── types.py           # Domain models
│   └── exceptions.py      # Custom exceptions
├── reasoning/              # LLM & reasoning
│   ├── llm/               # LLM adapters
│   ├── prompts/           # Prompt templates
│   └── strategies/        # Reasoning strategies
├── memory/                 # Memory management
│   ├── session.py         # Session state
│   ├── short_term.py      # Working memory
│   └── long_term.py       # Persistent memory
├── knowledge/              # RAG pipeline
│   ├── ingestion.py       # Document ingestion
│   ├── chunking.py        # Text chunking
│   ├── embeddings.py      # Embedding generation
│   └── retriever.py       # Knowledge retrieval
├── tools/                  # Tool system
│   ├── registry.py        # Tool registration
│   ├── executor.py        # Tool execution
│   ├── permissions.py     # Access control
│   └── builtin.py         # Built-in tools
├── planning/               # Task planning
│   ├── decomposer.py      # Task decomposition
│   ├── controller.py      # Execution control
│   └── checkpoints.py     # Checkpoint management
├── safety/                 # Safety & governance
│   ├── input_validation.py
│   ├── guardrails.py
│   ├── rbac.py
│   └── audit.py
├── observability/          # Monitoring
│   ├── tracing.py         # Distributed tracing
│   ├── metrics.py         # Prometheus metrics
│   ├── logging.py         # Structured logging
│   └── evaluation.py      # Evaluation harness
└── advanced/               # Advanced features
    ├── multi_agent.py     # Multi-agent orchestration
    ├── critic.py          # Self-critique
    └── plugins.py         # Plugin system
```

## 🔧 Configuration

Configuration is managed through environment variables and Pydantic settings:

```bash
# Core settings
AEGIS_ENV=production
AEGIS_DEBUG=false
AEGIS_LOG_LEVEL=INFO

# LLM providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Redis
AEGIS_REDIS_URL=redis://localhost:6379/0

# Security
AEGIS_API_KEY=your-api-key
AEGIS_JWT_SECRET=your-jwt-secret
```

## 🧪 Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_tools.py -v
```

## 🐳 Docker

```bash
# Build image
docker build -t aegis:latest .

# Run with Docker Compose
docker-compose up -d

# Development mode (with hot reload)
docker-compose --profile dev up

# View logs
docker-compose logs -f aegis
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Send message (supports streaming) |
| POST | `/sessions` | Create session |
| GET | `/sessions/{id}` | Get session |
| DELETE | `/sessions/{id}` | Delete session |
| GET | `/tools` | List available tools |
| POST | `/tools/{name}/execute` | Execute a tool |
| GET | `/admin/stats` | Platform statistics |
| GET | `/admin/metrics` | Prometheus metrics |

## 🔌 Extending

### Adding a Custom Tool

```python
from src.tools import tool_registry

@tool_registry.register
def my_custom_tool(query: str) -> str:
    """
    Description of what the tool does.
    
    Args:
        query: The search query
        
    Returns:
        The result
    """
    return f"Result for: {query}"
```

### Adding an LLM Provider

```python
from src.reasoning.llm.base import LLMAdapter, LLMResponse

class CustomAdapter(LLMAdapter):
    async def complete(self, messages, **kwargs) -> LLMResponse:
        # Implementation
        pass
    
    async def stream(self, messages, **kwargs):
        # Implementation
        pass
```

### Creating a Plugin

```python
from src.advanced.plugins import Plugin, PluginMetadata, HookType

class MyPlugin(Plugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            description="My custom plugin",
        )
    
    def get_hooks(self):
        return {
            HookType.PRE_REQUEST: self.on_request,
        }
    
    async def on_request(self, context):
        print(f"Request: {context.request_id}")
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

## 📚 Documentation

Full documentation available at [https://aegis-ai.github.io/aegis-agent-platform](https://aegis-ai.github.io/aegis-agent-platform)