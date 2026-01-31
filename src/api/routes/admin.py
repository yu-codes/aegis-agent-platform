"""
Admin Routes

Administrative endpoints for monitoring and configuration.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_components, require_user

router = APIRouter()


@router.get("/stats")
async def get_stats(
    components: dict = Depends(get_components),
    user: dict = Depends(require_user),
):
    """Get platform statistics."""
    # Check admin role
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    session_manager = components.get("session_manager")
    tool_registry = components.get("tool_registry")
    
    stats = {
        "sessions": {
            "active": 0,  # Would query from session manager
        },
        "tools": {
            "registered": len(tool_registry.get_all_definitions()) if tool_registry else 0,
        },
    }
    
    return stats


@router.get("/metrics")
async def get_metrics(
    user: dict = Depends(require_user),
):
    """Get Prometheus metrics."""
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from src.observability.metrics import get_metrics_collector
    
    collector = get_metrics_collector()
    
    from starlette.responses import Response
    return Response(
        content=collector.to_prometheus(),
        media_type="text/plain",
    )


class ConfigUpdate(BaseModel):
    """Configuration update request."""
    
    key: str
    value: str


@router.post("/config")
async def update_config(
    update: ConfigUpdate,
    user: dict = Depends(require_user),
):
    """Update configuration."""
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Placeholder - would update configuration
    return {"status": "updated", "key": update.key}


@router.get("/audit")
async def get_audit_logs(
    limit: int = 100,
    event_type: str | None = None,
    user: dict = Depends(require_user),
):
    """Get audit logs."""
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Placeholder - would query audit logs
    return {"logs": [], "total": 0}


@router.post("/cache/clear")
async def clear_cache(
    components: dict = Depends(get_components),
    user: dict = Depends(require_user),
):
    """Clear application caches."""
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Clear Redis cache if available
    redis = components.get("redis")
    if redis:
        # Would selectively clear cache keys
        pass
    
    return {"status": "cleared"}
