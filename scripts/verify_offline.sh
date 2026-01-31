#!/bin/bash
# =============================================================================
# Aegis Platform - Offline Verification Script
# =============================================================================
# This script verifies that the Aegis platform runs correctly without any
# external dependencies (no LLM API keys, no external Redis).
#
# Usage:
#   ./scripts/verify_offline.sh           # Full verification
#   ./scripts/verify_offline.sh --quick   # Skip Docker tests
# =============================================================================

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_subheader() {
    echo ""
    echo -e "${YELLOW}▸ $1${NC}"
}

pass() {
    echo -e "  ${GREEN}✓${NC} $1"
    ((PASSED++))
}

fail() {
    echo -e "  ${RED}✗${NC} $1"
    ((FAILED++))
}

warn() {
    echo -e "  ${YELLOW}!${NC} $1"
    ((WARNINGS++))
}

info() {
    echo -e "  ${BLUE}ℹ${NC} $1"
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

print_header "AEGIS OFFLINE VERIFICATION"
echo "  Starting at: $(date)"
echo "  Working dir: $(pwd)"

# Ensure we're in the project root
if [[ ! -f "pyproject.toml" ]]; then
    fail "Must run from project root directory"
    exit 1
fi

# Check for quick mode
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    info "Quick mode: skipping Docker tests"
fi

# =============================================================================
# 1. Environment Verification
# =============================================================================

print_header "1. ENVIRONMENT VERIFICATION"

print_subheader "Checking Python environment"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    pass "Python available: $PYTHON_VERSION"
else
    fail "Python3 not found"
fi

print_subheader "Checking dependencies"
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    pass "Virtual environment found"
else
    warn "No virtual environment found - using system Python"
fi

# Check core dependencies
for pkg in fastapi uvicorn pydantic; do
    if python3 -c "import $pkg" 2>/dev/null; then
        pass "Package '$pkg' available"
    else
        fail "Package '$pkg' not found"
    fi
done

print_subheader "Ensuring NO API keys are set"
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY
export LLM_OFFLINE_MODE=true
export LLM_DEFAULT_PROVIDER=stub
export REDIS_ENABLED=false

pass "API keys cleared"
pass "Offline mode enabled (LLM_OFFLINE_MODE=true)"
pass "Stub provider set (LLM_DEFAULT_PROVIDER=stub)"
pass "Redis disabled (REDIS_ENABLED=false)"

# =============================================================================
# 2. Module Import Verification
# =============================================================================

print_header "2. MODULE IMPORT VERIFICATION"

print_subheader "Core modules"
MODULES=(
    "src.core.interfaces"
    "src.core.agent_runtime"
    "src.config.settings"
    "src.api.app"
)

for module in "${MODULES[@]}"; do
    if python3 -c "import $module" 2>/dev/null; then
        pass "Import: $module"
    else
        fail "Import failed: $module"
    fi
done

print_subheader "LLM adapters"
LLM_MODULES=(
    "src.reasoning.llm.base"
    "src.reasoning.llm.stub_adapter"
)

for module in "${LLM_MODULES[@]}"; do
    if python3 -c "import $module" 2>/dev/null; then
        pass "Import: $module"
    else
        fail "Import failed: $module"
    fi
done

print_subheader "Memory backends"
if python3 -c "from src.memory.session import InMemorySessionBackend" 2>/dev/null; then
    pass "Import: InMemorySessionBackend"
else
    fail "Import failed: InMemorySessionBackend"
fi

# =============================================================================
# 3. StubLLMAdapter Unit Tests
# =============================================================================

print_header "3. STUB LLM ADAPTER VERIFICATION"

print_subheader "Testing StubLLMAdapter"
python3 << 'PYTEST'
import asyncio
import sys
sys.path.insert(0, '.')

from src.reasoning.llm.stub_adapter import StubLLMAdapter, ScriptedLLMAdapter

async def test_stub_adapter():
    adapter = StubLLMAdapter()
    
    # Test basic completion
    response = await adapter.complete([{"role": "user", "content": "Hello"}])
    assert response is not None
    assert len(response.content) > 0
    print(f"  ✓ Basic completion works: {len(response.content)} chars")
    
    # Test streaming
    chunks = []
    async for chunk in adapter.stream([{"role": "user", "content": "Hi"}]):
        chunks.append(chunk)
    assert len(chunks) > 0
    print(f"  ✓ Streaming works: {len(chunks)} chunks")
    
    # Test pattern matching (tools)
    response = await adapter.complete([{"role": "user", "content": "What tools are available?"}])
    assert "tool" in response.content.lower() or "stub" in response.content.lower()
    print(f"  ✓ Pattern matching works for 'tools'")
    
    # Test model property
    assert adapter.model == "stub-model-v1"
    print(f"  ✓ Model property: {adapter.model}")

asyncio.run(test_stub_adapter())
PYTEST

if [[ $? -eq 0 ]]; then
    pass "StubLLMAdapter tests passed"
else
    fail "StubLLMAdapter tests failed"
fi

print_subheader "Testing ScriptedLLMAdapter"
python3 << 'PYTEST'
import asyncio
import sys
sys.path.insert(0, '.')

from src.reasoning.llm.stub_adapter import ScriptedLLMAdapter

async def test_scripted_adapter():
    responses = [
        "First scripted response",
        "Second scripted response",
        "Third scripted response",
    ]
    adapter = ScriptedLLMAdapter(responses=responses)
    
    for i, expected in enumerate(responses):
        response = await adapter.complete([{"role": "user", "content": f"Message {i}"}])
        assert response.content == expected, f"Expected '{expected}', got '{response.content}'"
        print(f"  ✓ Scripted response {i+1}: correct")
    
    # Test cycling
    response = await adapter.complete([{"role": "user", "content": "Message 4"}])
    assert response.content == responses[0]
    print(f"  ✓ Response cycling works")

asyncio.run(test_scripted_adapter())
PYTEST

if [[ $? -eq 0 ]]; then
    pass "ScriptedLLMAdapter tests passed"
else
    fail "ScriptedLLMAdapter tests failed"
fi

# =============================================================================
# 4. Configuration Verification
# =============================================================================

print_header "4. CONFIGURATION VERIFICATION"

print_subheader "Testing settings loading"
python3 << 'PYTEST'
import sys
sys.path.insert(0, '.')

from src.config.settings import get_settings

settings = get_settings()

# Check offline mode
assert hasattr(settings.llm, 'offline_mode'), "LLMSettings missing offline_mode"
print(f"  ✓ LLM offline_mode: {settings.llm.offline_mode}")

# Check effective provider
effective = settings.llm.effective_provider
print(f"  ✓ Effective provider: {effective}")

# Check Redis enabled
assert hasattr(settings.redis, 'enabled'), "RedisSettings missing enabled"
print(f"  ✓ Redis enabled: {settings.redis.enabled}")
PYTEST

if [[ $? -eq 0 ]]; then
    pass "Settings verification passed"
else
    fail "Settings verification failed"
fi

# =============================================================================
# 5. Server Startup Test
# =============================================================================

print_header "5. SERVER STARTUP VERIFICATION"

print_subheader "Starting server in background"
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')

async def test_app_creation():
    from src.api.app import create_app
    app = create_app()
    print('  ✓ Application created successfully')
    print(f'  ✓ App title: {app.title}')
    print(f'  ✓ App version: {app.version}')

asyncio.run(test_app_creation())
"

if [[ $? -eq 0 ]]; then
    pass "Application creation succeeded"
else
    fail "Application creation failed"
fi

print_subheader "Testing server with uvicorn (5 second smoke test)"
timeout 5 python3 -c "
import uvicorn
import sys
sys.path.insert(0, '.')
from src.api.app import create_app
app = create_app()
uvicorn.run(app, host='127.0.0.1', port=8099, log_level='warning')
" &> /dev/null &
SERVER_PID=$!

sleep 2

if kill -0 $SERVER_PID 2>/dev/null; then
    pass "Server started successfully on port 8099"
    
    # Test health endpoint
    if command -v curl &> /dev/null; then
        if curl -s http://127.0.0.1:8099/health | grep -q "ok\|healthy"; then
            pass "Health endpoint responding"
        else
            warn "Health endpoint not responding as expected"
        fi
    fi
    
    kill $SERVER_PID 2>/dev/null || true
else
    fail "Server failed to start"
fi

wait 2>/dev/null

# =============================================================================
# 6. Docker Verification (unless --quick)
# =============================================================================

if [[ "$QUICK_MODE" != "true" ]]; then
    print_header "6. DOCKER VERIFICATION"
    
    if command -v docker &> /dev/null; then
        print_subheader "Building offline Docker image"
        
        if docker build \
            --target production \
            --build-arg INSTALL_OPENAI=false \
            --build-arg INSTALL_ANTHROPIC=false \
            --build-arg INSTALL_FAISS=false \
            -t aegis:offline-test \
            . 2>&1 | tail -5; then
            pass "Docker build succeeded"
        else
            fail "Docker build failed"
        fi
        
        print_subheader "Testing Docker container startup"
        CONTAINER_ID=$(docker run -d \
            -e LLM_OFFLINE_MODE=true \
            -e LLM_DEFAULT_PROVIDER=stub \
            -e REDIS_ENABLED=false \
            -p 8098:8000 \
            aegis:offline-test 2>/dev/null || echo "")
        
        if [[ -n "$CONTAINER_ID" ]]; then
            sleep 3
            
            if docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
                pass "Container running"
                
                if curl -s http://127.0.0.1:8098/health 2>/dev/null | grep -q "ok\|healthy"; then
                    pass "Container health check passed"
                else
                    warn "Container health endpoint not responding"
                fi
            else
                fail "Container exited unexpectedly"
                docker logs "$CONTAINER_ID" 2>&1 | tail -10
            fi
            
            docker stop "$CONTAINER_ID" &>/dev/null || true
            docker rm "$CONTAINER_ID" &>/dev/null || true
        else
            fail "Failed to start container"
        fi
        
        # Cleanup test image
        docker rmi aegis:offline-test &>/dev/null || true
    else
        warn "Docker not available - skipping Docker tests"
    fi
else
    print_header "6. DOCKER VERIFICATION (SKIPPED - Quick Mode)"
    info "Use './scripts/verify_offline.sh' without --quick to run Docker tests"
fi

# =============================================================================
# 7. API Endpoint Verification Checklist
# =============================================================================

print_header "7. API ENDPOINT CHECKLIST"

print_subheader "Core endpoints to verify"
info "GET  /health           - Health check"
info "GET  /api/v1/status    - System status"
info "POST /api/v1/chat      - Chat completion"
info "POST /api/v1/stream    - Streaming chat"
info "GET  /api/v1/domains   - List domain profiles"
info "GET  /docs             - OpenAPI documentation"
info "GET  /redoc            - ReDoc documentation"

# =============================================================================
# Summary
# =============================================================================

print_header "VERIFICATION SUMMARY"

echo ""
echo -e "  ${GREEN}Passed:${NC}   $PASSED"
echo -e "  ${RED}Failed:${NC}   $FAILED"
echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✓ ALL VERIFICATION CHECKS PASSED${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  ✗ VERIFICATION FAILED - $FAILED check(s) did not pass${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi
