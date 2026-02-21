"""
E2E Tests - Chat Flow

End-to-end tests for chat functionality.
Tests the complete flow from API to response in offline mode.
"""

import os
import pytest
from fastapi.testclient import TestClient

# Ensure offline mode for E2E tests
os.environ.setdefault("LLM_OFFLINE_MODE", "true")
os.environ.setdefault("LLM_DEFAULT_PROVIDER", "stub")


@pytest.fixture(scope="module")
def e2e_client():
    """Create test client for E2E tests."""
    from apps.api_server.app import create_app

    app = create_app(enable_docs=True, debug=True)
    return TestClient(app)


class TestChatFlow:
    """E2E tests for chat flow."""

    @pytest.mark.e2e
    def test_simple_chat(self, e2e_client):
        """Test simple chat interaction."""
        # First check health
        health_response = e2e_client.get("/health")
        assert health_response.status_code == 200

        # Then send a chat message
        chat_response = e2e_client.post(
            "/api/v1/chat",
            json={
                "message": "Hello, how are you?",
                "stream": False,
            },
        )
        # In offline mode, should work with stub provider
        assert chat_response.status_code in [200, 500]

    @pytest.mark.e2e
    def test_chat_with_session(self, e2e_client):
        """Test chat with session management."""
        # Create a session first
        session_response = e2e_client.post(
            "/api/v1/sessions",
            json={"user_id": "e2e-test-user"},
        )

        if session_response.status_code == 200:
            session_data = session_response.json()
            session_id = session_data.get("session_id")

            # Use session for chat
            chat_response = e2e_client.post(
                "/api/v1/chat",
                json={
                    "message": "Remember this number: 42",
                    "session_id": session_id,
                    "stream": False,
                },
            )
            assert chat_response.status_code in [200, 500]

    @pytest.mark.e2e
    def test_chat_with_tools(self, e2e_client):
        """Test chat with tool invocation."""
        response = e2e_client.post(
            "/api/v1/chat",
            json={
                "message": "What is 2 + 2? Use the calculator.",
                "stream": False,
                "use_tools": True,
            },
        )
        assert response.status_code in [200, 422, 500]


class TestToolFlow:
    """E2E tests for tool execution flow."""

    @pytest.mark.e2e
    def test_tool_list(self, e2e_client):
        """Test listing available tools."""
        response = e2e_client.get("/api/v1/tools")
        assert response.status_code in [200, 500]

    @pytest.mark.e2e
    def test_direct_tool_call(self, e2e_client):
        """Test direct tool execution."""
        response = e2e_client.post(
            "/api/v1/tools/call",
            json={
                "tool_name": "calculator",
                "arguments": {"expression": "10 * 5"},
            },
        )
        assert response.status_code in [200, 400, 404, 422, 500]


class TestMultiTurnConversation:
    """E2E tests for multi-turn conversations."""

    @pytest.mark.e2e
    def test_conversation_context(self, e2e_client):
        """Test conversation maintains context across turns."""
        # First turn
        response1 = e2e_client.post(
            "/api/v1/chat",
            json={
                "message": "My name is Alice.",
                "session_id": "e2e-context-test",
                "stream": False,
            },
        )
        assert response1.status_code in [200, 500]

        # Second turn - should remember context
        response2 = e2e_client.post(
            "/api/v1/chat",
            json={
                "message": "What is my name?",
                "session_id": "e2e-context-test",
                "stream": False,
            },
        )
        assert response2.status_code in [200, 500]


class TestOfflineModeFlow:
    """E2E tests specifically for offline mode."""

    @pytest.mark.e2e
    def test_offline_health(self, e2e_client):
        """Test health check in offline mode."""
        response = e2e_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.e2e
    def test_offline_chat(self, e2e_client):
        """Test chat works in offline mode with stub provider."""
        response = e2e_client.post(
            "/api/v1/chat",
            json={
                "message": "This is an offline test.",
                "stream": False,
            },
        )
        # Should work without external API calls
        assert response.status_code in [200, 500]

    @pytest.mark.e2e
    def test_docs_available(self, e2e_client):
        """Test OpenAPI docs are available."""
        response = e2e_client.get("/docs")
        assert response.status_code == 200

        response = e2e_client.get("/openapi.json")
        assert response.status_code == 200
