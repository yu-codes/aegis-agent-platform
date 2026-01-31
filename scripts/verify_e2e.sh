#!/bin/bash
# =============================================================================
# Aegis Platform End-to-End Verification Script
# =============================================================================
# This script validates all major components of the Aegis platform are working
# correctly. It tests API endpoints, container health, and system integration.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_BASE="${AEGIS_API_URL:-http://localhost:8002}"
TIMEOUT=5

# Counters
PASSED=0
FAILED=0
TOTAL=0

# Helper functions
log_test() {
    TOTAL=$((TOTAL + 1))
    echo -n "  [$TOTAL] $1... "
}

log_pass() {
    PASSED=$((PASSED + 1))
    echo -e "${GREEN}PASS${NC}"
}

log_fail() {
    FAILED=$((FAILED + 1))
    echo -e "${RED}FAIL${NC}"
    if [ -n "$1" ]; then
        echo "      Error: $1"
    fi
}

log_section() {
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  $1${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
}

# Check if API is reachable
check_api_available() {
    curl -s -o /dev/null -w "%{http_code}" "${API_BASE}/health" 2>/dev/null
}

# =============================================================================
# TESTS
# =============================================================================

log_section "CONTAINER STATUS"

log_test "Docker container running"
if docker ps --format '{{.Names}}' | grep -q "aegis-offline"; then
    log_pass
else
    log_fail "Container 'aegis-offline' not found"
fi

log_test "Container health status"
HEALTH=$(docker inspect --format='{{.State.Health.Status}}' aegis-offline 2>/dev/null || echo "unknown")
if [ "$HEALTH" = "healthy" ]; then
    log_pass
else
    log_fail "Health: $HEALTH"
fi

log_section "API ENDPOINTS"

log_test "Health endpoint (/health)"
RESP=$(curl -s "${API_BASE}/health" 2>/dev/null)
if echo "$RESP" | grep -q "healthy"; then
    log_pass
else
    log_fail "$RESP"
fi

log_test "Swagger UI (/docs)"
CODE=$(curl -s -o /dev/null -w "%{http_code}" "${API_BASE}/docs" 2>/dev/null)
if [ "$CODE" = "200" ]; then
    log_pass
else
    log_fail "HTTP $CODE"
fi

log_test "OpenAPI schema (/openapi.json)"
CODE=$(curl -s -o /dev/null -w "%{http_code}" "${API_BASE}/openapi.json" 2>/dev/null)
if [ "$CODE" = "200" ]; then
    log_pass
else
    log_fail "HTTP $CODE"
fi

log_section "DOMAIN SYSTEM"

log_test "List domains (/api/v1/domains)"
DOMAINS=$(curl -s "${API_BASE}/api/v1/domains" 2>/dev/null)
if echo "$DOMAINS" | grep -q '"total":'; then
    DOMAIN_COUNT=$(echo "$DOMAINS" | grep -o '"total":[0-9]*' | grep -o '[0-9]*')
    echo -e "${GREEN}PASS${NC} ($DOMAIN_COUNT domains loaded)"
    PASSED=$((PASSED + 1))
else
    log_fail
fi

log_test "Domain: financial_analysis"
if echo "$DOMAINS" | grep -q "financial_analysis"; then
    log_pass
else
    log_fail "Domain not found"
fi

log_test "Domain: general_chat"
if echo "$DOMAINS" | grep -q "general_chat"; then
    log_pass
else
    log_fail "Domain not found"
fi

log_test "Domain: technical_support"
if echo "$DOMAINS" | grep -q "technical_support"; then
    log_pass
else
    log_fail "Domain not found"
fi

log_section "TOOLS SYSTEM"

log_test "List tools (/api/v1/tools)"
TOOLS=$(curl -s "${API_BASE}/api/v1/tools" 2>/dev/null)
if echo "$TOOLS" | grep -q '"total":'; then
    TOOL_COUNT=$(echo "$TOOLS" | grep -o '"total":[0-9]*' | grep -o '[0-9]*')
    echo -e "${GREEN}PASS${NC} ($TOOL_COUNT tools registered)"
    PASSED=$((PASSED + 1))
else
    log_fail
fi

log_test "Tool: get_current_time"
if echo "$TOOLS" | grep -q "get_current_time"; then
    log_pass
else
    log_fail "Tool not found"
fi

log_test "Tool: calculate"
if echo "$TOOLS" | grep -q "calculate"; then
    log_pass
else
    log_fail "Tool not found"
fi

log_test "Tool: web_search"
if echo "$TOOLS" | grep -q "web_search"; then
    log_pass
else
    log_fail "Tool not found"
fi

log_section "SESSION MANAGEMENT"

log_test "Create session"
SESSION=$(curl -s -X POST "${API_BASE}/api/v1/sessions" \
    -H "Content-Type: application/json" \
    -d '{}' 2>/dev/null)
if echo "$SESSION" | grep -q '"id":'; then
    SESSION_ID=$(echo "$SESSION" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    echo -e "${GREEN}PASS${NC} (ID: ${SESSION_ID:0:8}...)"
    PASSED=$((PASSED + 1))
else
    log_fail "$SESSION"
fi

log_test "Get session"
if [ -n "$SESSION_ID" ]; then
    GET_SESSION=$(curl -s "${API_BASE}/api/v1/sessions/${SESSION_ID}" 2>/dev/null)
    if echo "$GET_SESSION" | grep -q "$SESSION_ID"; then
        log_pass
    else
        log_fail "$GET_SESSION"
    fi
else
    log_fail "No session ID"
fi

log_test "List sessions"
SESSIONS=$(curl -s "${API_BASE}/api/v1/sessions" 2>/dev/null)
if echo "$SESSIONS" | grep -q '"sessions":'; then
    log_pass
else
    log_fail "$SESSIONS"
fi

log_section "CHAT FUNCTIONALITY"

log_test "Chat completion (Stub LLM)"
CHAT_RESP=$(curl -s -X POST "${API_BASE}/api/v1/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello, how are you?"}' 2>/dev/null)
if echo "$CHAT_RESP" | grep -q "offline mode\|STUB\|Aegis"; then
    log_pass
else
    log_fail "$CHAT_RESP"
fi

log_test "Chat includes domain resolution"
if echo "$CHAT_RESP" | grep -q "domain"; then
    log_pass
else
    log_fail "No domain info in response"
fi

log_test "Chat includes streaming events"
if echo "$CHAT_RESP" | grep -q "event:"; then
    log_pass
else
    log_fail "No events in response"
fi

log_section "CLEANUP"

log_test "Delete session"
if [ -n "$SESSION_ID" ]; then
    DEL_RESP=$(curl -s -X DELETE "${API_BASE}/api/v1/sessions/${SESSION_ID}" 2>/dev/null)
    if echo "$DEL_RESP" | grep -q "deleted"; then
        log_pass
    else
        log_fail "$DEL_RESP"
    fi
else
    log_fail "No session to delete"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  VERIFICATION SUMMARY${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Total Tests:  $TOTAL"
echo -e "  Passed:       ${GREEN}$PASSED${NC}"
echo -e "  Failed:       ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All verification tests passed!${NC}"
    echo ""
    echo "The Aegis platform is running correctly in offline mode."
    echo "Access the API at: ${API_BASE}"
    echo "Swagger UI at: ${API_BASE}/docs"
    exit 0
else
    echo -e "${RED}✗ Some verification tests failed.${NC}"
    echo ""
    echo "Please check the container logs for details:"
    echo "  docker logs aegis-offline"
    exit 1
fi
