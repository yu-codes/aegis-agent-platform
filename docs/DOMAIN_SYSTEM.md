# Domain-Aware Agent Configuration System

## Architecture Overview

The Domain System enables the Aegis agent to dynamically adapt its behavior based on task domain **without modifying application code**. Domains are defined declaratively via YAML configuration and resolved at runtime.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           API Request                                    │
│  POST /api/v1/chat                                                       │
│  { "message": "...", "domain": "financial_analysis" }                    │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DomainResolver                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │   EXPLICIT   │→ │   CONTEXT    │→ │   INFERRED   │→ │  FALLBACK   │  │
│  │ API param    │  │ Session/User │  │ Classifier   │  │  Default    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ DomainProfile (frozen, read-only)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       DomainAwareRuntime                                 │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Configuration Applied                            │ │
│  │  • System prompt from profile.prompt                               │ │
│  │  • RAG config from profile.rag (collection, filters, top_k)        │ │
│  │  • Memory scope from profile.memory                                │ │
│  │  • Tools filtered by profile.tools (allow/deny lists)              │ │
│  │  • Reasoning strategy from profile.reasoning                       │ │
│  │  • Safety guardrails from profile.safety                           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                  │                                       │
│                                  ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     AgentRuntime.run()                             │ │
│  │  (Existing execution with domain-configured components)            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Abstraction: DomainProfile

A `DomainProfile` is a **first-class, declarative configuration** that completely describes agent behavior for a task domain.

### Key Properties

| Property | Description |
| -------- | ----------- |
| **Declarative** | All configuration is data (YAML), not code |
| **Versionable** | Semantic versioning (`1.2.3`) |
| **Immutable** | Frozen at runtime (read-only) |
| **Composable** | Can extend base profiles (`extends: base_domain`) |
| **Auditable** | Includes metadata, creation timestamps |

### Schema

```python
@dataclass
class DomainProfile:
    # Identity
    name: str                    # Unique identifier (e.g., "financial_analysis")
    version: str                 # Semantic version (e.g., "1.0.0")
    extends: str | None          # Base profile to extend
    
    # Configuration sections
    prompt: PromptConfig         # System prompt, persona, templates
    rag: RAGConfig               # Retrieval settings
    memory: MemoryConfig         # Memory scope and policies
    tools: ToolsConfig           # Tool access control
    reasoning: ReasoningConfig   # Strategy and limits
    safety: SafetyConfig         # Guardrails and compliance
    
    # Custom domain data
    custom: dict[str, Any]       # Domain-specific settings
```

---

## Integration Points

### 1. src/reasoning/ - Prompt & Strategy

The domain profile controls:

- **System prompt**: `profile.prompt.system_prompt`
- **Persona**: `profile.prompt.persona`  
- **Strategy selection**: `profile.reasoning.strategy` (ReAct, ToolCalling, etc.)
- **Model override**: `profile.reasoning.model`
- **Temperature**: `profile.reasoning.temperature`

```yaml
# Example: technical_support domain
prompt:
  system_prompt: |
    You are an expert IT support specialist...
    Your approach:
    1. GATHER INFORMATION
    2. DIAGNOSE
    3. GUIDE
    4. VERIFY

reasoning:
  strategy: "react"           # Step-by-step for troubleshooting
  max_iterations: 15          # Allow more iterations for complex issues
  temperature: 0.5            # Lower for more focused responses
```

### 2. src/knowledge/ - RAG Configuration

The domain profile controls:

- **Collection/index**: `profile.rag.collection`
- **Retrieval parameters**: `profile.rag.top_k`, `profile.rag.min_score`
- **Metadata filters**: `profile.rag.metadata_filters`
- **Context budget**: `profile.rag.max_context_tokens`

```yaml
# Example: financial_analysis domain
rag:
  enabled: true
  collection: "financial_research"
  top_k: 10
  min_score: 0.7              # Higher threshold for accuracy
  require_sources: true       # MUST cite sources
  metadata_filters:
    content_type: "financial"
    verified: true
```

### 3. src/memory/ - Memory Policies

The domain profile controls:

- **Scope**: `profile.memory.scope` (session, user, domain, global)
- **History limits**: `profile.memory.max_turns`
- **Summarization**: `profile.memory.summarize_after`
- **Long-term memory**: `profile.memory.long_term_enabled`

```yaml
# Example: technical_support domain
memory:
  enabled: true
  scope: "session"            # Keep troubleshooting context
  max_turns: 30               # Longer sessions for complex issues
  summarize_after: 20
  long_term_enabled: true     # Remember past issues
```

### 4. src/tools/ - Tool Access Control

The domain profile controls:

- **Allowed tools**: `profile.tools.allowed_tools`
- **Denied tools**: `profile.tools.denied_tools`
- **Category restrictions**: `profile.tools.allowed_categories`
- **Confirmation requirements**: `profile.tools.require_confirmation`

```yaml
# Example: financial_analysis domain
tools:
  enabled: true
  allowed_tools:
    - "search_financial_news"
    - "get_market_data"
    - "portfolio_analyzer"
  denied_tools:
    - "execute_trade"         # NEVER execute trades
    - "transfer_funds"        # NEVER move money
  require_confirmation:
    - "portfolio_analyzer"    # Confirm before analyzing personal data
```

### 5. src/planning/ - Planning Integration

The domain profile controls:

- **Planning enablement**: `profile.reasoning.enable_planning`
- **Planning threshold**: `profile.reasoning.planning_threshold`

```yaml
reasoning:
  enable_planning: true       # Use planning for complex tasks
  planning_threshold: 5       # Trigger after 5 steps
```

### 6. src/safety/ - Safety Constraints

The domain profile controls:

- **Input guardrails**: `profile.safety.input_guardrails_enabled`
- **Output guardrails**: `profile.safety.output_guardrails_enabled`
- **Blocked topics**: `profile.safety.blocked_topics`
- **PII redaction**: `profile.safety.redact_pii_input/output`
- **Disclaimers**: `profile.safety.disclaimer`

```yaml
# Example: financial_analysis domain (strict compliance)
safety:
  input_guardrails_enabled: true
  output_guardrails_enabled: true
  blocked_topics:
    - "insider_trading"
    - "market_manipulation"
  output_blocked_patterns:
    - "(?i)you\\s+should\\s+(buy|sell)"   # Never give advice
  redact_pii_input: true
  redact_pii_output: true
  require_citations: true
  disclaimer: |
    ⚠️ This information is for educational purposes only...
```

### 7. src/api/ - API Integration

The chat endpoint accepts an optional `domain` parameter:

```python
class ChatRequest(BaseModel):
    message: str
    domain: str | None = None  # Explicit domain override

# Response includes domain info
class ChatResponse(BaseModel):
    domain: str | None
    domain_resolution_method: str  # "explicit", "inferred", "fallback"
```

---

## Domain Resolution Flow

```
1. EXPLICIT (highest priority)
   ├─ Check: request.domain provided?
   ├─ Check: domain exists in registry?
   └─ Result: Use specified domain or continue to next step

2. CONTEXT
   ├─ Check: session.domain set?
   ├─ Check: user.default_domain set?
   └─ Result: Use context domain or continue to next step

3. INFERRED
   ├─ Run: KeywordClassifier.classify(message)
   ├─ Check: confidence >= threshold (0.6)?
   └─ Result: Use inferred domain or continue to next step

4. FALLBACK (always succeeds)
   └─ Result: Use default "general" domain
```

---

## Example Domain Profiles

### technical_support.yaml

```yaml
name: technical_support
version: "1.0.0"
display_name: "Technical Support Agent"

prompt:
  system_prompt: |
    You are an expert IT support specialist...
    1. GATHER INFORMATION
    2. DIAGNOSE
    3. GUIDE
    4. VERIFY

rag:
  collection: "support_docs"
  top_k: 8
  rerank: true

memory:
  scope: "session"
  max_turns: 30
  long_term_enabled: true

tools:
  allowed_tools:
    - "search_knowledge_base"
    - "create_support_ticket"
    - "check_system_status"
  denied_tools:
    - "execute_code"
    - "shell_command"

reasoning:
  strategy: "react"
  max_iterations: 15

safety:
  blocked_topics:
    - "hacking"
    - "bypassing_security"
  log_all_interactions: true
```

### financial_analysis.yaml

```yaml
name: financial_analysis
version: "1.0.0"
display_name: "Financial Analysis Assistant"

prompt:
  system_prompt: |
    You are a financial analyst. CRITICAL:
    - NEVER provide investment advice
    - ALWAYS cite sources
    - INCLUDE disclaimers

rag:
  collection: "financial_research"
  top_k: 10
  min_score: 0.7
  require_sources: true

memory:
  scope: "user"
  retention_days: 365

tools:
  allowed_tools:
    - "search_financial_news"
    - "get_market_data"
  denied_tools:
    - "execute_trade"
    - "transfer_funds"

reasoning:
  strategy: "tool_calling"
  temperature: 0.3

safety:
  blocked_topics:
    - "insider_trading"
  require_citations: true
  disclaimer: "⚠️ Not investment advice..."
```

---

## Same Input, Different Domain, Different Behavior

### Input

```
"What should I do about my stock portfolio?"
```

### With `domain: "general_chat"`

```
Response: "I can help you think about your portfolio! What specific aspects
are you wondering about? I can discuss general investment concepts, though
for specific advice you should consult a financial advisor."

Configuration:
- Generic system prompt
- No specialized RAG collection
- All tools available
- Standard guardrails
```

### With `domain: "financial_analysis"`

```
Response: "📊 I can help analyze your portfolio structure. Based on current
market data [1], here are some observations...

⚠️ **Disclaimer**: This information is for educational purposes only and 
does not constitute financial advice. Please consult a licensed advisor."

Configuration:
- Financial-specific system prompt
- RAG from "financial_research" collection
- Only analysis tools (no trading)
- PII redaction enabled
- Mandatory disclaimer appended
- Citations required

Sources:
[1] Market data from XYZ (15min delay)
```

---

## Guardrails Against Domain Confusion/Abuse

### 1. Domain Profile is READ-ONLY

```python
class DomainProfile(BaseModel):
    class Config:
        frozen = True  # Immutable at runtime
```

### 2. Tool Filtering is Enforced

```python
class DomainAwareToolExecutor:
    async def execute(self, name: str, args: dict, context):
        if not self._profile.is_tool_allowed(name):
            return ToolResult(
                error=f"Tool '{name}' not available in domain '{self._profile.name}'"
            )
```

### 3. Safe Fallback is Guaranteed

```python
async def resolve(self, explicit_domain, content, context) -> ResolutionResult:
    # ... resolution logic ...
    
    # 4. FALLBACK: Always succeeds
    return ResolutionResult(
        profile=self._registry.fallback,
        method=ResolutionMethod.FALLBACK,
        confidence=1.0,
    )
```

### 4. Domain Resolution is Auditable

```python
@dataclass
class ResolutionResult:
    profile: DomainProfile
    method: ResolutionMethod      # explicit, inferred, fallback
    confidence: float             # 0.0-1.0
    matched_patterns: list[str]   # What triggered inference
    candidates: list[tuple]       # All candidates considered
```

### 5. No Domain Logic in Application Code

All domain-specific behavior comes from YAML configuration:

```yaml
# ❌ WRONG: Hard-coded domain logic
if domain == "financial":
    add_disclaimer()
    block_trading_tools()

# ✅ RIGHT: Declarative configuration
safety:
  disclaimer: "..."
tools:
  denied_tools: ["execute_trade"]
```

---

## Adding New Domains

Teams can add domains independently by:

1. **Create YAML file**: `config/domains/my_domain.yaml`
2. **Define profile**: Follow the schema
3. **Deploy**: No code changes required

```yaml
# config/domains/legal_research.yaml
name: legal_research
version: "1.0.0"
extends: general_chat  # Inherit defaults

prompt:
  system_prompt: |
    You are a legal research assistant...
    NEVER provide legal advice.

rag:
  collection: "legal_documents"

tools:
  allowed_tools:
    - "search_case_law"
    - "search_statutes"

safety:
  require_citations: true
  disclaimer: "This is legal research, not legal advice."
```

---

## Quality Checklist

| Requirement | Status |
| ----------- | ------ |
| No domain-specific logic scattered in code | ✅ All in YAML |
| Domains are declarative, not procedural | ✅ DomainProfile schema |
| Behavior differences are explainable | ✅ ResolutionResult audit trail |
| Safe default behavior is guaranteed | ✅ Fallback domain always succeeds |
| Multiple teams can add domains independently | ✅ YAML files, no code changes |
| Domain profiles are versionable | ✅ Semantic versioning |
| Domain profiles are testable | ✅ Pydantic validation |
| Domain profiles are auditable | ✅ Metadata and resolution tracking |
