# Aegis Agent Platform - Architecture Consolidation

> **Version**: 1.0.0  
> **Date**: Post-Integration Consolidation  
> **Status**: Production-Ready with Offline Verification

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Phase Integration Summary](#phase-integration-summary)
4. [Offline Execution Mode](#offline-execution-mode)
5. [Docker Deployment Guide](#docker-deployment-guide)
6. [API Verification Checklist](#api-verification-checklist)
7. [Configuration Reference](#configuration-reference)
8. [Known Risks & Gaps](#known-risks--gaps)
9. [Appendix](#appendix)

---

## Executive Summary

The Aegis Agent Platform has completed three major integration phases, resulting in a **production-ready, domain-agnostic AI agent framework**. Key achievements:

| Capability | Status | Description |
|------------|--------|-------------|
| **Unified AgentRuntime** | ✅ Complete | Single execution engine with layered timeout handling |
| **Behavioral Evaluation** | ✅ Complete | Hermetic replay harness for regression testing |
| **Domain Profiles** | ✅ Complete | Dynamic behavior configuration without code changes |
| **Offline Mode** | ✅ Complete | Zero-dependency execution for CI/testing |

**Critical Guarantee**: The system builds and runs WITHOUT any external API keys.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AEGIS AGENT PLATFORM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   API Layer │───▶│  AgentRuntime   │───▶│     Reasoning Engine        │  │
│  │  (FastAPI)  │    │  (Unified Core) │    │  (LLM Adapters + Planner)   │  │
│  └─────────────┘    └─────────────────┘    └─────────────────────────────┘  │
│         │                   │                           │                    │
│         │                   │                           ▼                    │
│         │                   │              ┌─────────────────────────────┐  │
│         │                   │              │      LLM Adapter Layer      │  │
│         │                   │              ├─────────────────────────────┤  │
│         │                   │              │  OpenAI │ Anthropic │ Stub  │  │
│         │                   │              └─────────────────────────────┘  │
│         │                   │                                               │
│         ▼                   ▼                                               │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │  Sessions   │───▶│  Domain Profile │───▶│     Memory Backends         │  │
│  │  Management │    │  Configuration  │    │  (Redis │ InMemory)         │  │
│  └─────────────┘    └─────────────────┘    └─────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | Path | Purpose |
|--------|------|---------|
| **AgentRuntime** | `src/core/agent_runtime.py` | Unified execution engine |
| **Interfaces** | `src/core/interfaces.py` | Protocol definitions (LLMAdapterProtocol, etc.) |
| **Settings** | `src/config/settings.py` | Pydantic configuration with env support |
| **API App** | `src/api/app.py` | FastAPI application with lifespan management |
| **LLM Adapters** | `src/reasoning/llm/` | OpenAI, Anthropic, Stub adapters |
| **Memory** | `src/memory/` | Session backends (Redis, InMemory) |
| **Domain Profiles** | `src/domains/` | Behavior configuration system |

---

## Phase Integration Summary

### Phase 1: AgentRuntime Unification

**Goal**: Single execution engine replacing scattered agent implementations.

**Key Changes**:
- Unified `AgentRuntime` class with standardized lifecycle
- Layered timeout handling (step, turn, session levels)
- Consistent error propagation and logging
- Protocol-based adapter injection

**Files Modified**:
- `src/core/agent_runtime.py` - Core runtime implementation
- `src/core/interfaces.py` - Protocol definitions
- `src/api/app.py` - Runtime wiring in lifespan

### Phase 2: Behavioral Evaluation Harness

**Goal**: Deterministic replay testing without external dependencies.

**Key Changes**:
- Hermetic evaluation with mocked LLM responses
- Behavioral regression detection
- Test fixture recording and replay
- Performance baseline comparison

**Files Created**:
- `tests/evaluation/` - Evaluation framework
- `tests/fixtures/` - Recorded behaviors

### Phase 3: DomainProfile System

**Goal**: Dynamic behavior configuration without code changes.

**Key Changes**:
- YAML-based domain definitions
- Runtime profile switching
- Guardrail and constraint configuration
- Tool availability per domain

**Files Created**:
- `src/domains/profile.py` - Profile loader
- `src/domains/profiles/` - Domain YAML files

---

## Offline Execution Mode

### Design Principles

1. **Zero External Dependencies**: No API keys required for startup
2. **Deterministic Responses**: StubLLMAdapter provides reproducible outputs
3. **CI/Testing Ready**: All verification runs without secrets
4. **Graceful Fallback**: Missing keys auto-fallback to stub mode

### Configuration

```bash
# Enable offline mode via environment variables
export LLM_OFFLINE_MODE=true
export LLM_DEFAULT_PROVIDER=stub
export REDIS_ENABLED=false
```

### StubLLMAdapter

Located at: `src/reasoning/llm/stub_adapter.py`

**Features**:
- Full `LLMAdapterProtocol` compliance
- Pattern-based response selection
- Simulated streaming with configurable delay
- Multi-turn scripted conversations via `ScriptedLLMAdapter`

**Usage**:
```python
from src.reasoning.llm.stub_adapter import StubLLMAdapter

adapter = StubLLMAdapter(
    model="stub-model-v1",
    stream_delay_ms=20,
)

response = await adapter.complete([
    {"role": "user", "content": "Hello"}
])
```

### InMemorySessionBackend

Fallback when Redis is disabled:
```python
from src.memory.session import InMemorySessionBackend

backend = InMemorySessionBackend()
```

---

## Docker Deployment Guide

### Available Profiles

| Profile | Command | Use Case |
|---------|---------|----------|
| **Production** | `docker-compose up` | Full stack with Redis |
| **Development** | `docker-compose --profile dev up` | Hot reload enabled |
| **Offline** | `docker-compose --profile offline up` | Zero API keys |

### Build Arguments

```dockerfile
# Dockerfile ARGs for optional dependencies
ARG INSTALL_OPENAI=true     # Include OpenAI SDK
ARG INSTALL_ANTHROPIC=true  # Include Anthropic SDK
ARG INSTALL_FAISS=false     # Include FAISS vector search
```

### Offline Build

```bash
# Build minimal image without LLM SDKs
docker build \
  --target production \
  --build-arg INSTALL_OPENAI=false \
  --build-arg INSTALL_ANTHROPIC=false \
  -t aegis:offline .

# Run with offline configuration
docker run -d \
  -e LLM_OFFLINE_MODE=true \
  -e LLM_DEFAULT_PROVIDER=stub \
  -e REDIS_ENABLED=false \
  -p 8000:8000 \
  aegis:offline
```

### Docker Compose Offline

```yaml
# In docker-compose.yml - use the offline profile
# docker-compose --profile offline up

aegis-offline:
  profiles:
    - offline
  environment:
    - LLM_OFFLINE_MODE=true
    - LLM_DEFAULT_PROVIDER=stub
    - REDIS_ENABLED=false
```

---

## API Verification Checklist

### Core Endpoints

| Endpoint | Method | Description | Offline Safe |
|----------|--------|-------------|--------------|
| `/health` | GET | Health check | ✅ |
| `/api/v1/status` | GET | System status | ✅ |
| `/api/v1/chat` | POST | Chat completion | ✅ (stub) |
| `/api/v1/stream` | POST | Streaming chat | ✅ (stub) |
| `/api/v1/domains` | GET | List domain profiles | ✅ |
| `/api/v1/domains/{id}` | GET | Get domain profile | ✅ |
| `/docs` | GET | OpenAPI documentation | ✅ |
| `/redoc` | GET | ReDoc documentation | ✅ |

### Verification Script

```bash
# Run full verification
./scripts/verify_offline.sh

# Quick mode (skip Docker tests)
./scripts/verify_offline.sh --quick
```

### Manual Verification Commands

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Status check
curl http://localhost:8000/api/v1/status

# 3. Chat (with stub adapter)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# 4. List domains
curl http://localhost:8000/api/v1/domains
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_ENV` | `development` | Environment (development/production) |
| `AEGIS_DEBUG` | `false` | Enable debug mode |
| `AEGIS_LOG_LEVEL` | `INFO` | Logging level |
| `AEGIS_PORT` | `8000` | Server port |
| **LLM Settings** | | |
| `LLM_DEFAULT_PROVIDER` | `openai` | LLM provider (openai/anthropic/stub) |
| `LLM_OFFLINE_MODE` | `false` | Force stub adapter |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `LLM_STUB_MODEL_NAME` | `stub-model-v1` | Stub model identifier |
| `LLM_STUB_STREAM_DELAY_MS` | `20` | Simulated stream delay |
| **Redis Settings** | | |
| `REDIS_ENABLED` | `true` | Enable Redis backend |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |

### Settings Classes

```python
# src/config/settings.py

class LLMSettings(BaseSettings):
    default_provider: Literal["openai", "anthropic", "stub"] = "openai"
    offline_mode: bool = False
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    stub_model_name: str = "stub-model-v1"
    stub_stream_delay_ms: int = 20
    
    @property
    def effective_provider(self) -> str:
        if self.offline_mode:
            return "stub"
        return self.default_provider

class RedisSettings(BaseSettings):
    enabled: bool = True
    url: str = "redis://localhost:6379/0"
```

---

## Known Risks & Gaps

### Current Limitations

| Area | Status | Mitigation |
|------|--------|------------|
| **Tool Execution** | Stub tools only | Extend StubLLMAdapter for tool mocking |
| **Memory Persistence** | InMemory loses data on restart | Document for users; Redis for production |
| **Rate Limiting** | Not implemented | Add before production deployment |
| **Authentication** | Basic API key only | Implement JWT for production |

### Technical Debt

1. **Adapter Factory Pattern**: Currently in `app.py`, should be separate module
2. **Error Handling**: Some adapters may leak implementation details
3. **Logging**: Inconsistent log levels across modules
4. **Tests**: Need more integration tests for offline mode

### Security Considerations

- API keys should NEVER be committed to version control
- Use `.env` files with `.gitignore` for local development
- In production, use secrets management (AWS Secrets Manager, HashiCorp Vault)

---

## Appendix

### File Structure

```
aegis-agent-platform/
├── src/
│   ├── api/
│   │   └── app.py                 # FastAPI application
│   ├── config/
│   │   └── settings.py            # Pydantic settings
│   ├── core/
│   │   ├── agent_runtime.py       # Unified runtime
│   │   └── interfaces.py          # Protocol definitions
│   ├── reasoning/
│   │   └── llm/
│   │       ├── base.py            # Base adapter
│   │       ├── openai_adapter.py  # OpenAI implementation
│   │       ├── anthropic_adapter.py # Anthropic implementation
│   │       └── stub_adapter.py    # Offline stub adapter
│   ├── memory/
│   │   └── session.py             # Session backends
│   └── domains/
│       ├── profile.py             # Profile loader
│       └── profiles/              # YAML definitions
├── scripts/
│   └── verify_offline.sh          # Offline verification
├── tests/
│   └── evaluation/                # Behavioral tests
├── docker-compose.yml
├── Dockerfile
└── docs/
    └── CONSOLIDATION.md           # This document
```

### Quick Start Commands

```bash
# Development with hot reload
docker-compose --profile dev up

# Production with full stack
docker-compose up -d

# Offline/CI mode
docker-compose --profile offline up

# Run verification
./scripts/verify_offline.sh

# Local development (no Docker)
export LLM_OFFLINE_MODE=true
export REDIS_ENABLED=false
python -m uvicorn src.api.app:app --reload
```

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Current | Initial consolidation complete |

---

*This document is the authoritative reference for Aegis platform architecture and deployment.*
