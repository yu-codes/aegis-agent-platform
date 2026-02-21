#!/bin/bash
# =============================================================================
# Test API Endpoints Script
# =============================================================================
# Usage:
#   ./scripts/test_api.sh [BASE_URL]
#
# Default BASE_URL: http://localhost:8080
# =============================================================================

set -e

BASE_URL="${1:-http://localhost:8080}"
PASS_COUNT=0
FAIL_COUNT=0

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Testing Aegis API Endpoints"
echo "Base URL: $BASE_URL"
echo "========================================"

# Function to test an endpoint
test_endpoint() {
    local method="$1"
    local endpoint="$2"
    local expected_code="$3"
    local data="$4"
    local description="$5"

    echo -n "Testing: $description... "

    if [ "$method" == "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint" 2>/dev/null)
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$BASE_URL$endpoint" 2>/dev/null)
    fi

    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "$status_code" == "$expected_code" ]; then
        echo -e "${GREEN}PASS${NC} (HTTP $status_code)"
        ((PASS_COUNT++))
    else
        echo -e "${RED}FAIL${NC} (Expected $expected_code, got $status_code)"
        echo "  Response: $body"
        ((FAIL_COUNT++))
    fi
}

echo ""
echo "--- Health Endpoints ---"
test_endpoint "GET" "/health" "200" "" "Health check"
test_endpoint "GET" "/health/live" "200" "" "Liveness probe"
test_endpoint "GET" "/health/ready" "200" "" "Readiness probe"

echo ""
echo "--- API Documentation ---"
test_endpoint "GET" "/docs" "200" "" "Swagger UI"
test_endpoint "GET" "/openapi.json" "200" "" "OpenAPI spec"

echo ""
echo "--- Session Endpoints ---"
test_endpoint "GET" "/api/v1/sessions" "200" "" "List sessions"
test_endpoint "POST" "/api/v1/sessions" "200" '{"user_id": "test-user"}' "Create session"

echo ""
echo "--- Chat Endpoints ---"
test_endpoint "POST" "/api/v1/chat" "200" '{"message": "Hello!", "stream": false}' "Simple chat"

echo ""
echo "--- Tool Endpoints ---"
test_endpoint "GET" "/api/v1/tools" "200" "" "List tools"

echo ""
echo "--- Admin Endpoints ---"
test_endpoint "GET" "/api/v1/admin/stats" "200" "" "Admin stats"
test_endpoint "GET" "/api/v1/admin/config" "200" "" "Admin config"

echo ""
echo "========================================"
echo "Results: ${GREEN}$PASS_COUNT passed${NC}, ${RED}$FAIL_COUNT failed${NC}"
echo "========================================"

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi

exit 0
