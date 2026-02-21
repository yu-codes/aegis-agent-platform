"""
Health Routes

Health check endpoints.

Based on: src/api/routes/health.py
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Response

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Basic health check.

    Returns service status.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "aegis-agent-platform",
    }


@router.get("/health/live")
async def liveness_check() -> dict[str, str]:
    """
    Kubernetes liveness probe.

    Returns 200 if service is alive.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_check() -> dict[str, Any]:
    """
    Kubernetes readiness probe.

    Returns 200 if service is ready to accept requests.
    """
    from apps.api_server.dependencies import (
        get_agent_orchestrator,
        get_session_manager,
        get_tool_registry,
    )

    checks = {
        "orchestrator": False,
        "memory": False,
        "tools": False,
    }

    try:
        get_agent_orchestrator()
        checks["orchestrator"] = True
    except Exception:
        pass

    try:
        get_session_manager()
        checks["memory"] = True
    except Exception:
        pass

    try:
        get_tool_registry()
        checks["tools"] = True
    except Exception:
        pass

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/metrics")
async def metrics_endpoint() -> Response:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format.
    """
    from apps.api_server.dependencies import get_metrics

    try:
        metrics = get_metrics()
        content = metrics.export_prometheus()
    except Exception:
        content = ""

    return Response(
        content=content,
        media_type="text/plain; charset=utf-8",
    )
