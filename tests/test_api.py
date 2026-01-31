"""
Tests for API Endpoints
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test main health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]

    def test_liveness_check(self, client):
        """Test liveness probe."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_check(self, client):
        """Test readiness probe."""
        response = client.get("/health/ready")
        # May be 200 or 503 depending on dependencies
        assert response.status_code in [200, 503]


class TestToolsEndpoints:
    """Tests for tools endpoints."""

    def test_list_tools(self, client):
        """Test listing available tools."""
        response = client.get("/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)

    def test_list_categories(self, client):
        """Test listing tool categories."""
        response = client.get("/tools/categories")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data


class TestSessionEndpoints:
    """Tests for session endpoints."""

    def test_create_session(self, client):
        """Test creating a session."""
        response = client.post("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    def test_get_nonexistent_session(self, client):
        """Test getting a session that doesn't exist."""
        import uuid
        fake_id = str(uuid.uuid4())
        response = client.get(f"/sessions/{fake_id}")
        assert response.status_code == 404
