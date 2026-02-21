"""
Admin Routes

Administration endpoints.

Based on: src/api/routes/admin.py
"""

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/stats")
async def get_stats():
    """
    Get system statistics.

    Returns metrics and usage statistics.
    """
    from apps.api_server.dependencies import get_metrics

    metrics = get_metrics()

    return {
        "metrics": metrics.collect(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/audit")
async def get_audit_logs(
    event_type: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    hours: int = Query(default=24, ge=1, le=168),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Get audit logs.

    Args:
        event_type: Filter by event type
        user_id: Filter by user ID
        session_id: Filter by session ID
        hours: Time window in hours
        limit: Maximum results
    """
    from apps.api_server.dependencies import get_audit_log

    audit_log = get_audit_log()

    start_time = datetime.utcnow() - timedelta(hours=hours)

    entries = await audit_log.query(
        event_type=event_type,
        user_id=user_id,
        session_id=session_id,
        start_time=start_time,
        limit=limit,
    )

    return {
        "entries": [e.to_dict() for e in entries],
        "total": len(entries),
        "filters": {
            "event_type": event_type,
            "user_id": user_id,
            "session_id": session_id,
            "start_time": start_time.isoformat(),
        },
    }


@router.get("/config")
async def get_config():
    """
    Get current configuration.

    Returns non-sensitive configuration values.
    """
    from apps.api_server.dependencies import get_settings

    settings = get_settings()

    return {
        "environment": settings.get("environment"),
        "debug": settings.get("debug"),
        "log_level": settings.get("log_level"),
        "max_concurrent_sessions": settings.get("max_concurrent_sessions"),
        "default_model": settings.get("default_model"),
    }


@router.post("/config/reload")
async def reload_config():
    """
    Reload configuration.

    Reloads configuration from environment.
    """
    # Clear cached settings
    from apps.api_server.dependencies import get_settings

    get_settings.cache_clear()

    return {"status": "reloaded"}


@router.get("/domains")
async def list_domains():
    """
    List available domains.

    Returns configured domain profiles.
    """
    from services.rag import DomainRegistry

    registry = DomainRegistry()
    domains = registry.list_domains()

    return {
        "domains": [
            {
                "name": d.name,
                "description": d.description,
                "retrieval_config": d.retrieval_config,
            }
            for d in domains
        ],
        "total": len(domains),
    }


@router.post("/cache/clear")
async def clear_cache(
    cache_type: str = Query(default="all", description="Cache type: all, session, rag"),
):
    """
    Clear caches.

    Args:
        cache_type: Type of cache to clear
    """
    from apps.api_server.dependencies import get_session_manager, get_rag_retriever

    cleared = []

    if cache_type in ("all", "session"):
        memory = get_session_manager()
        # Clear session cache if method exists
        if hasattr(memory, "clear_cache"):
            memory.clear_cache()
        cleared.append("session")

    if cache_type in ("all", "rag"):
        retriever = get_rag_retriever()
        if hasattr(retriever, "clear_cache"):
            retriever.clear_cache()
        cleared.append("rag")

    return {"status": "cleared", "caches": cleared}


@router.get("/sessions/active")
async def get_active_sessions():
    """
    Get active session count.

    Returns number of active sessions.
    """
    from apps.api_server.dependencies import get_session_manager

    memory = get_session_manager()
    sessions = await memory.list_sessions(limit=1000)

    return {
        "active_sessions": len(sessions),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/shutdown")
async def shutdown():
    """
    Graceful shutdown.

    Initiates graceful shutdown of the service.
    """
    import asyncio
    import signal
    import os

    # Send SIGTERM to self
    async def delayed_shutdown():
        await asyncio.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(delayed_shutdown())

    return {"status": "shutting_down"}
