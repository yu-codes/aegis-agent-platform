"""
Middleware

Custom middleware for the API.

Based on: src/api/middleware.py
"""

import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Request tracing middleware.

    Adds trace context to requests.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tracing."""
        from apps.api_server.dependencies import get_tracer, get_metrics

        try:
            tracer = get_tracer()
            metrics = get_metrics()
        except RuntimeError:
            return await call_next(request)

        # Extract trace context from headers
        parent_context = tracer.extract_context(dict(request.headers))

        # Start span
        span = tracer.start_span(
            name=f"{request.method} {request.url.path}",
            parent=parent_context,
            kind="server",
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.route": request.url.path,
            },
        )

        # Store span in request state
        request.state.span = span

        start_time = time.time()

        try:
            response = await call_next(request)

            # Record metrics
            duration = time.time() - start_time
            span.set_attribute("http.status_code", response.status_code)

            metrics.counter("requests_total").inc(
                labels={
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status": str(response.status_code),
                }
            )
            metrics.histogram("request_duration_seconds").observe(duration)

            tracer.end_span(span)

            # Add trace headers to response
            response.headers["X-Trace-ID"] = str(span.context.trace_id)

            return response

        except Exception as e:
            tracer.end_span(span, status="error", error=str(e))
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.

    Limits requests per client.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_limit: int = 10,
    ):
        super().__init__(app)
        self._rpm = requests_per_minute
        self._burst = burst_limit
        self._requests: dict[str, list[datetime]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Get client identifier
        client_id = self._get_client_id(request)

        # Check rate limit
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)

        # Clean old requests
        self._requests[client_id] = [t for t in self._requests[client_id] if t > window_start]

        # Check limit
        if len(self._requests[client_id]) >= self._rpm:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit of {self._rpm} requests per minute exceeded",
                    "retry_after": 60,
                },
            )

        # Record request
        self._requests[client_id].append(now)

        response = await call_next(request)

        # Add rate limit headers
        remaining = self._rpm - len(self._requests[client_id])
        response.headers["X-RateLimit-Limit"] = str(self._rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier."""
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:8]}"

        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Error handling middleware.

    Catches and formats errors.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with error handling."""
        try:
            return await call_next(request)
        except Exception as e:
            return self._handle_error(request, e)

    def _handle_error(self, request: Request, error: Exception) -> JSONResponse:
        """Handle an error."""
        # Log error
        from services.observability import Logger

        logger = Logger.get_logger("api.error")
        logger.error(
            f"Request error: {error}",
            error=error,
            path=str(request.url.path),
            method=request.method,
        )

        # Determine status code
        status_code = 500
        error_type = "internal_error"

        if isinstance(error, ValueError):
            status_code = 400
            error_type = "validation_error"
        elif isinstance(error, PermissionError):
            status_code = 403
            error_type = "permission_denied"
        elif isinstance(error, FileNotFoundError):
            status_code = 404
            error_type = "not_found"

        return JSONResponse(
            status_code=status_code,
            content={
                "error": error_type,
                "message": str(error),
            },
        )


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware.

    Validates API keys and JWT tokens.
    """

    PUBLIC_PATHS = {"/health", "/health/ready", "/health/live", "/docs", "/redoc", "/openapi.json"}

    def __init__(self, app, api_keys: set[str] | None = None):
        super().__init__(app)
        self._api_keys = api_keys or set()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with authentication."""
        # Skip public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Check API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            if api_key in self._api_keys or not self._api_keys:
                request.state.user_id = f"api_key:{api_key[:8]}"
                return await call_next(request)

        # Check Bearer token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user_id = await self._validate_token(token)
            if user_id:
                request.state.user_id = user_id
                return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={
                "error": "unauthorized",
                "message": "Valid API key or token required",
            },
        )

    async def _validate_token(self, token: str) -> str | None:
        """Validate JWT token."""
        # TODO: Implement JWT validation
        return None
