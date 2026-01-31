"""
API Middleware

Custom middleware for cross-cutting concerns.
"""

import time
from collections.abc import Callable
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Adds tracing headers and context to requests.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        # Generate or extract trace ID
        trace_id = request.headers.get("X-Trace-Id", str(uuid4()))
        request_id = request.headers.get("X-Request-Id", str(uuid4()))

        # Store in request state
        request.state.trace_id = trace_id
        request.state.request_id = request_id

        # Start timing
        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Add headers to response
        response.headers["X-Trace-Id"] = trace_id
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting.

    For production, use Redis-based rate limiting.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        key_func: Callable[[Request], str] | None = None,
    ):
        super().__init__(app)
        self._rpm = requests_per_minute
        self._key_func = key_func or self._default_key
        self._requests: dict[str, list[float]] = {}

    def _default_key(self, request: Request) -> str:
        """Default rate limit key: IP address."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        key = self._key_func(request)
        now = time.time()

        # Clean old entries
        if key in self._requests:
            self._requests[key] = [t for t in self._requests[key] if now - t < 60]
        else:
            self._requests[key] = []

        # Check rate limit
        if len(self._requests[key]) >= self._rpm:
            return Response(
                content='{"error": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self._rpm),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Record request
        self._requests[key].append(now)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = self._rpm - len(self._requests[key])
        response.headers["X-RateLimit-Limit"] = str(self._rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware.

    Validates API keys and JWT tokens.
    """

    def __init__(
        self,
        app,
        api_key_header: str = "X-API-Key",
        bearer_header: str = "Authorization",
        public_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self._api_key_header = api_key_header
        self._bearer_header = bearer_header
        self._public_paths = public_paths or ["/health", "/docs", "/openapi.json"]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        # Skip auth for public paths
        if request.url.path in self._public_paths:
            return await call_next(request)

        # Check API key
        api_key = request.headers.get(self._api_key_header)
        if api_key:
            user = await self._validate_api_key(api_key)
            if user:
                request.state.user = user
                return await call_next(request)

        # Check Bearer token
        auth_header = request.headers.get(self._bearer_header)
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user = await self._validate_token(token)
            if user:
                request.state.user = user
                return await call_next(request)

        # No valid auth
        return Response(
            content='{"error": "Unauthorized"}',
            status_code=401,
            media_type="application/json",
        )

    async def _validate_api_key(self, api_key: str) -> dict | None:
        """Validate API key and return user info."""
        # Implement API key validation
        # This is a placeholder
        return None

    async def _validate_token(self, token: str) -> dict | None:
        """Validate JWT token and return user info."""
        # Implement JWT validation
        # This is a placeholder
        return None


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            # Log the error
            import traceback

            traceback.print_exc()

            # Return appropriate error response
            from src.core.exceptions import AegisError

            if isinstance(e, AegisError):
                return Response(
                    content=f'{{"error": "{e.message}", "code": "{e.code}"}}',
                    status_code=e.status_code,
                    media_type="application/json",
                )

            return Response(
                content='{"error": "Internal server error"}',
                status_code=500,
                media_type="application/json",
            )
