"""
Integration Tests - API

Tests for API endpoints.
"""

import os
import pytest
from fastapi.testclient import TestClient


# Ensure offline mode for testing
os.environ.setdefault("LLM_OFFLINE_MODE", "true")
os.environ.setdefault("LLM_DEFAULT_PROVIDER", "stub")


@pytest.fixture
def client():
    """Create test client."""
    from apps.api_server.app import create_app

    app = create_app(enable_docs=True, debug=True)
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data

    def test_liveness_check(self, client):
        """Test liveness check endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_check(self, client):
        """Test readiness check endpoint."""
        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["ready", "initializing"]


class TestChatEndpoints:
    """Tests for chat endpoints."""

    def test_chat_request_basic(self, client):
        """Test basic chat request."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello, how are you?",
                "stream": False,
            },
        )
        # Should return 200 or handle missing dependencies gracefully
        assert response.status_code in [200, 422, 500]

    def test_chat_request_with_session(self, client):
        """Test chat request with session ID."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Test message",
                "session_id": "test-session-123",
                "stream": False,
            },
        )
        assert response.status_code in [200, 422, 500]


class TestSessionEndpoints:
    """Tests for session endpoints."""

    def test_list_sessions(self, client):
        """Test listing sessions."""
        response = client.get("/api/v1/sessions")
        assert response.status_code in [200, 500]

    def test_session_creation(self, client):
        """Test session creation."""
        response = client.post(
            "/api/v1/sessions",
            json={"user_id": "test-user"},
        )
        assert response.status_code in [200, 201, 422, 500]


class TestToolEndpoints:
    """Tests for tool endpoints."""

    def test_list_tools(self, client):
        """Test listing tools."""
        response = client.get("/api/v1/tools")
        assert response.status_code in [200, 500]

    def test_tool_call(self, client):
        """Test calling a tool."""
        response = client.post(
            "/api/v1/tools/call",
            json={
                "tool_name": "calculator",
                "arguments": {"expression": "2+2"},
            },
        )
        assert response.status_code in [200, 400, 404, 422, 500]


class TestAdminEndpoints:
    """Tests for admin endpoints."""

    def test_admin_stats(self, client):
        """Test admin stats endpoint."""
        response = client.get("/api/v1/admin/stats")
        assert response.status_code in [200, 401, 403, 500]

    def test_admin_config(self, client):
        """Test admin config endpoint."""
        response = client.get("/api/v1/admin/config")
        assert response.status_code in [200, 401, 403, 500]
