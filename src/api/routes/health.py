"""
Health Check Routes
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy"}


@router.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    # Check if all required components are ready
    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive"}
