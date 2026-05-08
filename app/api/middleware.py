"""
app/api/middleware.py — FastAPI middleware.

Adds:
1. Request/response logging with latency tracking
2. Correlation ID header injection (for distributed tracing)
3. Global exception handler
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api.middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request and response with timing information.
    Adds X-Request-ID header for correlation with downstream logs.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        # Attach request ID to request state for access in route handlers
        request.state.request_id = request_id

        logger.info(
            "→ %s %s [%s]",
            request.method,
            request.url.path,
            request_id,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled exception in request [%s]: %s", request_id, exc)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "error": str(exc), "request_id": request_id},
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

        logger.info(
            "← %s %s [%s] %dms",
            request.method,
            request.url.path,
            request_id,
            int(elapsed_ms),
        )
        return response
