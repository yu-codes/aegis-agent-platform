"""
Health Check Routes
"""

from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Basic health check."""
    return {"status": "healthy"}


@router.get("/health/ready")
async def readiness_check() -> dict[str, Any]:
    """Readiness check for Kubernetes."""
    # Check if all required components are ready
    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check() -> dict[str, Any]:
    """Liveness check for Kubernetes."""
    return {"status": "alive"}
